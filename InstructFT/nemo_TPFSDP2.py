#!/usr/bin/python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from nemo.utils.decorators import deprecated_warning

deprecated_warning(
    old_method="Automodel on NVIDIA/NeMo", new_method="https://github.com/NVIDIA-NeMo/Automodel repo", wait_seconds=2
)

import os
import tempfile
from pathlib import Path

import fiddle as fdl
import lightning.pytorch as pl

from nemo import lightning as nl
from nemo.automodel.dist_utils import FirstRankPerNode
from nemo.automodel.loss import chunked_cross_entropy, masked_cross_entropy
from nemo.automodel.misc_utils import calculate_valid_accumulate_grad_batches
from nemo.collections import llm
from nemo.collections.llm.gpt.data.hf_dataset import HFMockDataModule, HFDatasetDataModule
from nemo.lightning.pytorch.callbacks import JitConfig, JitTransform

from transformers import AutoTokenizer
import torch.distributed as dist

import datasets
from utils import (
    make_sft_collate,
    Chronometer,
    MyChronoCallback
)



# Run this example with torchrun, for example:
# torchrun --nproc-per-node=8 \
#   examples/llm/finetune/automodel.py \
#   --strategy fsdp2 \
#   --devices 8 \
#   --model meta-llama/Llama-3.2-1B \
#   --ckpt-folder "output"
#
# For PEFT please also pass --lora
# Note: ensure that the --nproc-per-node and --devices values match.


def make_strategy(
    rank,
    strategy,
    model,
    devices,
    num_nodes,
    adapter_only=False,
    enable_cpu_offload=False,
    dp_size=None,
    tp_size=None,
    cp_size=None,
    sequence_parallel=False,
    use_hf_tp_plan=False,
):
    if strategy == 'auto':
        return pl.strategies.SingleDeviceStrategy(
            device='cuda:0',
            checkpoint_io=model.make_checkpoint_io(adapter_only=adapter_only),
        )
    elif strategy == 'ddp':
        return pl.strategies.DDPStrategy(
            checkpoint_io=model.make_checkpoint_io(adapter_only=adapter_only),
            find_unused_parameters=True,
        )
        if rank == 0: print(f"Using DDP with DP={dp_size}, TP={tp_size}, CP={cp_size}")
    elif strategy == 'fsdp2':

        offload_policy = None
        if enable_cpu_offload:
            from nemo.lightning.pytorch.strategies.fsdp2_strategy import HAS_CPU_OFFLOAD_POLICY, CPUOffloadPolicy

            assert HAS_CPU_OFFLOAD_POLICY, "Could not import offload policy"
            offload_policy = CPUOffloadPolicy()

        assert (
            dp_size * tp_size * cp_size == devices * num_nodes
        ), "Data Parallel size * Tensor Parallel size * Context Parallel size must equal to devices * num_nodes"
        if rank == 0: print(f"Using FSDP2 with DP={dp_size}, TP={tp_size}, CP={cp_size}")

        return nl.FSDP2Strategy(
            data_parallel_size=dp_size,
            tensor_parallel_size=tp_size,
            context_parallel_size=cp_size,
            sequence_parallel=sequence_parallel,
            checkpoint_io=model.make_checkpoint_io(adapter_only=adapter_only),
            offload_policy=offload_policy,
            use_hf_tp_plan=use_hf_tp_plan,
        )
    else:
        raise NotImplementedError("Encountered unknown strategy")


def logger(ckpt_folder, save_every_n_train_steps) -> nl.NeMoLogger:
    ckpt = nl.ModelCheckpoint(
        save_last=True,
        every_n_train_steps=save_every_n_train_steps,
        monitor="reduced_train_loss",
        save_top_k=1,
        save_on_train_epoch_end=True,
        save_optim_on_train_end=True,
    )

    return nl.NeMoLogger(
        name="nemo2_finetune",
        log_dir=ckpt_folder,
        use_datetime_version=False,  # must be false if using auto resume
        ckpt=ckpt,
        wandb=None,
    )


def main():
    """Example script to run SFT/PEFT with a HF transformers-instantiated model on squad."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-72B-Instruct', help='Hugging Face model-id to use')
    parser.add_argument(
        '--strategy',
        type=str,
        default='fsdp2',
        choices=['auto', 'ddp', 'fsdp2'],
        help='Training strategy e.g. ddp/fsdp2/single-gpu',
    )
    parser.add_argument('--devices', type=int, default=2, help='Number of GPUs to use')
    parser.add_argument('--num-nodes', type=int, default=1, help='Number of Nodes to use; to be used with torchrun')
    parser.add_argument('--dp-size', type=int, default=None, help='Data Parallel size; to be used with fsdp2')
    parser.add_argument('--tp-size', type=int, default=1, help='Tensor Parallel size; to be used with fsdp2')
    parser.add_argument('--cp-size', type=int, default=1, help='Context Parallel size; to be used with fsdp2')
    parser.add_argument(
        '--sequence-parallel',
        action='store_true',
        help='Use Sequence Parallelism; to be used with fsdp2 and tp_size > 1',
    )
    parser.add_argument('--use-hf-tp-plan', action='store_true', help='Use huggingface TP plan; to be used with TP')
    parser.add_argument('--use-te-optimizer', action='store_true', help='Use TE optimizer')
    parser.add_argument('--grad-clip', type=float, default=None, help='Grad clip value')
    parser.add_argument(
        '--accumulate-grad-batches',
        '--accumulate_grad_batches',
        type=int,
        default=1,
        help='Number of batches to accumulate gradient over.',
    )
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--max-steps', type=int, default=100, help='Maximum number of training steps')
    parser.add_argument('--log-every-n-steps', type=int, default=10, help='Log every n steps')
    parser.add_argument('--max-epochs', type=int, default=2, help='Maximum number of training epochs')
    parser.add_argument('--wandb-project', type=str, default=None, help='Wandb project to use')
    parser.add_argument('--use-torch-jit', action='store_true', help='Enables torch.compile on model')
    parser.add_argument('--enable-cpu-offload', action='store_true', help='Enabled cpu offloading; requires FSDP2')
    parser.add_argument('--auto-resume', action='store_true', help='Enables autoresume from a previous training job')
    parser.add_argument('--liger', action='store_true', help='Enables Liger-Kernels')
    parser.add_argument(
        '--attn-implementation',
        type=str,
        default="sdpa",
        choices=["flash_attention_2", "sdpa", "eager"],
        help='Attention implementation to use. Default: sdpa',
    )
    parser.add_argument('--enable-grad-ckpt', action='store_true', help='Enables gradient checkpoint')
    parser.add_argument(
        '--ckpt-folder', type=str, default=tempfile.TemporaryDirectory().name, help='Directory to save checkpoints'
    )
    parser.add_argument('--global-batch-size', default=128, type=int, help='Global batch size to use for training.')
    parser.add_argument(
        '--batch-size',
        '--micro-batch-size',
        dest='batch_size',
        default=1,
        type=int,
        help='Micro batch size to use for training.',
    )
    parser.add_argument(
        '--limit-val-batches',
        default=0.0,
        type=float,
        help=(
            'How much of validation dataset to check. Useful when debugging or testing '
            'something that happens at the end of an epoch. Default to 0.0 (disabled)'
        ),
    )
    parser.add_argument('--seq-length', default=4096, type=int, help='Sequence length to use for training')
    parser.add_argument(
        '--trust-remote-code',
        action='store_true',
        help='Enables trust_remote_code to load HF models with unverified sources',
    )
    parser.add_argument('--load-in-4bit', action='store_true', help='Use 4-bit quantization for e.g. for qlora')
    parser.add_argument('--fp8', action='store_true', help='Enables fp8 training')
    parser.add_argument('--lr', type=float, default=3e-6, help='Learning rate for training.')
    parser.add_argument(
        '--use-chunked-ce', action='store_true', help='Use chunked cross entropy loss instead of the standard CE loss.'
    )
    parser.add_argument('--no-lce', action='store_false', help='Disables LCE')
    parser.add_argument('--mock-dataset', action='store_true', help='Use HFMockDataModule for training.')
    parser.add_argument(
        '--limit-dataset-samples',
        type=int,
        default=None,
        help='If set will limit num of dataset samples. Default None (disabled)',
    )
    parser.add_argument(
        '--packed-sequence-size',
        type=int,
        default=-1,
        help='If a positive integer, this arg enables training with sequence packing in case of HFDatasetDataModule'
        'class and specifies the pack size. If less than or equal to 0, sequence packing is disabled. Packed sequences'
        'are currently supported only with position_ids and not attention_mask. Hence packed sequences needs to be'
        'run with --attn-implementation=flash_attention_2',
    )
    parser.add_argument('--start-of-turn-token', default=None, help='Chat turn token')
    parser.add_argument('--lora', action='store_true', help='Enables PEFT (LoRA) finetuning; Default: off (SFT).')

    args = parser.parse_args()

    # --- fix DP auto + checks ---
    world = args.devices * args.num_nodes
    denom = args.tp_size * args.cp_size

    RANK = os.environ['SLURM_PROCID']
    
    if denom <= 0:
        raise ValueError("tp_size and cp_size must be >=1")
    
    if world % denom != 0:
        raise ValueError(f"Incompatible config: world={world} not divisible by TP*CP={denom}")
    
    if args.dp_size is None:
        args.dp_size = world // denom
    
    assert args.dp_size * args.tp_size * args.cp_size == world, \
        f"3D mismatch: DP*TP*CP={args.dp_size*args.tp_size*args.cp_size} != world={world}"


    # CPUOffload WA for known issue
    if args.enable_cpu_offload and args.use_te_optimizer:
        args.use_te_optimizer = False

    try:
        args.accumulate_grad_batches = calculate_valid_accumulate_grad_batches(
            global_batch_size=args.global_batch_size,
            micro_batch_size=args.batch_size,
            devices=args.devices,
            num_nodes=args.num_nodes,
            tp_size=args.tp_size,
            cp_size=args.cp_size,
        )
    except ValueError as e:
        print(f"Error calculating gradient accumulation steps: {e}")
        print("Using default value of 1 for accumulate_grad_batches")
        args.accumulate_grad_batches = 1

    if RANK == 0: print(
    f"world size: {world}, GBS: {args.global_batch_size}, BSperDev: {args.batch_size}, grad accumulation:{args.grad_acc}, sequence length: {args.seq_length}, grad_check: {args.enable_grad_ckpt},  chunked CE: {args.chunked_cross_entropy}"
    )

    wandb = None
    if args.wandb_project is not None:
        model = '_'.join(args.model.split('/')[-2:])
        from lightning.pytorch.loggers import WandbLogger

        wandb = WandbLogger(
            project=args.wandb_project,
            name=f"{model}_nodes{args.num_nodes}_dev{args.devices}_strat_{args.strategy}_dp{args.dp_size}_cp{args.cp_size}_tp{args.tp_size}_sp{args.sequence_parallel}_seqlen{args.seq_length}_gb{args.global_batch_size}_mb{args.batch_size}_lr{args.lr}",
        )

    callbacks = []
    if args.use_torch_jit:
        jit_config = JitConfig(use_torch=True, torch_kwargs={'dynamic': False}, use_thunder=False)
        callbacks = [JitTransform(jit_config)]

    callbacks.append(MyChronoCallback(RANK, args.accumulate_grad_batches))

    if args.use_te_optimizer:
        # Use TE optimizer
        # Faster convergence but may lead to memory issues
        optimizer = fdl.build(llm.adam.te_adam_with_flat_lr(lr=args.lr))
    else:
        optimizer = fdl.build(
            llm.adam.pytorch_adam_with_flat_lr(lr=args.lr, foreach=False)
        )  # foreach need to be False for TP

    if args.fp8:
        from nemo.lightning.pytorch.accelerate.transformer_engine import TEConfig

        model_accelerator = TEConfig(fp8_autocast=True)
    else:
        model_accelerator = None

    # ---------- data (parquet + collate) ----------
    train_dataset = datasets.load_dataset(
        "parquet",
        data_files="/lustre/fswork/dataset/tulu-3-sft-mixture/data/*.parquet",
        split="train"
    )
    # tokenizer HF (left padding) — cohérent avec SFT causal
    try:
        # chemin local via $DSDIR/HuggingFace_Models/<model>
        local_path = Path(os.environ.get("DSDIR", "")) / "HuggingFace_Models" / args.model
        tokenizer = AutoTokenizer.from_pretrained(str(local_path), padding_side="left")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    
    def wrap_tuple_to_dict(old_collate_fn):
        def new_collate_fn(batch):
            t = old_collate_fn(batch)
            if isinstance(t, (list, tuple)) and len(t) >= 3:
                return {"input_ids": t[0], "attention_mask": t[1], "labels": t[2]}
            raise TypeError(f"collate_fn returned {type(t)}, expected tuple of 3 elements")
        return new_collate_fn
    
    collate_fn = wrap_tuple_to_dict(make_sft_collate(tokenizer, max_seq_length=args.seq_length))

    hf_dataset = HFDatasetDataModule(
        train_dataset,
        split="train",
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        seq_length=args.seq_length,
        micro_batch_size=args.batch_size,
        global_batch_size=args.global_batch_size,
        pad_seq_len_divisible=16 if args.fp8 else None,
        # packed_sequence=False,      # à activer si tu veux packer
    )

    local_model_path = Path(os.environ.get("DSDIR", "")) / "HuggingFace_Models" / args.model

    model = llm.HFAutoModelForCausalLM(
        model_name=local_model_path,
        tokenizer=tokenizer,
        model_accelerator=model_accelerator,
        attn_implementation=args.attn_implementation,
        loss_fn=chunked_cross_entropy if args.use_chunked_ce else masked_cross_entropy,
        trust_remote_code=args.trust_remote_code,
        use_liger_kernel=args.liger,
        enable_grad_ckpt=args.enable_grad_ckpt,
        use_linear_ce_loss=args.no_lce,
        load_in_4bit=args.load_in_4bit,
    )

    assert (
        args.devices * args.num_nodes == args.dp_size * args.tp_size * args.cp_size
    ), f"Total devices {args.devices * args.num_nodes} must equal Data Parallel size {args.dp_size} * Tensor Parallel size {args.tp_size} * Context Parallel size {args.cp_size}."

    strategy = make_strategy(
        RANK,
        args.strategy,
        model,
        args.devices,
        args.num_nodes,
        False,
        args.enable_cpu_offload,
        dp_size=args.dp_size,
        tp_size=args.tp_size,
        cp_size=args.cp_size,
        sequence_parallel=args.sequence_parallel,
        use_hf_tp_plan=args.use_hf_tp_plan,
    )

    resume = (
        nl.AutoResume(
            resume_if_exists=True,
            resume_ignore_no_checkpoint=True,
        )
        if args.auto_resume
        else None
    )

    
    if args.lora:
        peft = llm.peft.LoRA(
            target_modules=['*_proj'],
            dim=8,
        )
    else:
        peft = None
    llm.api.finetune(
        model=model,
        data=hf_dataset,
        trainer=nl.Trainer(
            devices=args.devices,
            num_nodes=args.num_nodes,
            max_steps=args.max_steps,
            max_epochs=args.max_epochs,
            accelerator='gpu',
            strategy=strategy,
            log_every_n_steps=args.log_every_n_steps,
            num_sanity_val_steps=0,
            limit_val_batches=args.limit_val_batches,
            accumulate_grad_batches=args.accumulate_grad_batches,
            gradient_clip_val=args.grad_clip,
            use_distributed_sampler=True,  # needed to use PL DistributedSampler
            logger=wandb,
            callbacks=callbacks,
            precision="bf16-mixed",
            enable_checkpointing=False,      # désactive les checkpoints automatiques
            enable_model_summary=False,
        ),
        peft=peft,
        optim=optimizer,
        #log=logger(args.ckpt_folder, args.max_steps // 2),
        #resume=resume,
    )


if __name__ == '__main__':
    main()

