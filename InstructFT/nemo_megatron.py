import os
import tempfile
from pathlib import Path

import torch
from nemo import lightning as nl
from nemo.collections import llm
from megatron.core.optimizer import OptimizerConfig

from transformers import AutoTokenizer
import datasets
from utils import (
    make_sft_collate,
    Chronometer,
    MyChronoCallback
)

import lightning.pytorch as pl
from torch.utils.data import DataLoader
from nemo.lightning.data import add_megatron_sampler

RANK = int(os.environ.get('SLURM_PROCID', '0'))

def main():
    import argparse
    import math

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-72B-Instruct', choices=['Qwen/Qwen2.5-14B-Instruct', 'Qwen/Qwen2.5-32B-Instruct', 'Qwen/Qwen2.5-72B-Instruct'],
                        help='HF model id (ex: Qwen/Qwen2.5-32B-Instruct)')

    # cluster / parallelism
    parser.add_argument('--devices', type=int, default=2, help='Number of GPUs per node')
    parser.add_argument('--num-nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--dp-size', type=int, default=None, help='Data Parallel size (auto if None)')
    parser.add_argument('--tp-size', type=int, default=1, help='Tensor Parallel size')
    parser.add_argument('--pp-size', type=int, default=1, help='Pipeline Parallel size (Megatron)')
    parser.add_argument('--cp-size', type=int, default=1, help='Context Parallel size')
    parser.add_argument('--virtual-pp-size', type=int, default=None, help='Interleaved PP size')
    #parser.add_argument('--ep-size', type=int, default=None, help='Expert Parallel size (MoE)')
    parser.add_argument('--sequence-parallel', action='store_true',
                        help='Use sequence parallel (when TP>1)')

    # training & logging
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--accumulate-grad-batches', '--accumulate_grad_batches',
                        dest='accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=100)
    #parser.add_argument('--max-epochs', type=int, default=2)
    parser.add_argument('--log-every-n-steps', type=int, default=10)
    #parser.add_argument('--limit-val-batches', type=float, default=0.0)
    parser.add_argument('--wandb-project', type=str, default=None)
    parser.add_argument('--auto-resume', action='store_true')

    # io / data
    parser.add_argument('--ckpt-folder', type=str, default='ckpt_folder',
                        help='Workdir for checkpoints/logs')
    parser.add_argument('--seq-length', type=int, default=4096)
    parser.add_argument('--batch-size', '--micro-batch-size', dest='batch_size', type=int, default=None)
    parser.add_argument('--global-batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--attn-implementation', type=str, default='sdpa',
                        choices=['flash_attention_2','sdpa','eager'])

    # optim
    parser.add_argument("--warmup-steps", default=2000, type=int, help="linear warmup of learning rate before cosine annealing")
    parser.add_argument('--lr', '--learning-rate', dest='learning_rate', type=float, default=1e-5)
    parser.add_argument('--wd', '--weight-decay', dest='weight_decay', type=float, default=0.1)

    args = parser.parse_args()  

    # ---------- 3D parallelism checks ----------
    world_size = args.devices * args.num_nodes
    tp, pp, cp = args.tp_size, args.pp_size, args.cp_size

    if args.dp_size is None:
        denom = tp * pp * cp
        if world_size % denom != 0:
            raise ValueError(
                f"Incompatible 3D config: devices*num_nodes={world_size} "
                f"not divisible by TP*PP*CP={denom}."
            )
        args.dp_size = world_size // denom
    total_3d = args.dp_size * tp * pp * cp
    assert total_3d == world_size, (
        f"3D mismatch: DP*TP*PP*CP={total_3d} must equal devices*num_nodes={world_size}"
    )
    if RANK==0: print(f"[Megatron 3D] DP={args.dp_size}, TP={tp}, PP={pp}, CP={cp} | world_size={world_size}")

    # Déduire une micro-batch size quand --batch-size n'est pas fournie
    if args.batch_size is None:
        denom = args.dp_size * args.accumulate_grad_batches
        assert args.global_batch_size % denom == 0, (
            f"global_batch_size ({args.global_batch_size}) doit être divisible par "
            f"dp_size*accumulate_grad_batches ({denom})."
        )
        args.batch_size = args.global_batch_size // denom
        if RANK==0: print(f"micro_batch size: {args.batch_size} * Accumulate Grad: {args.accumulate_grad_batches} * DP_size: {args.dp_size} = global BS: {args.global_batch_size}")
        
    ## setup the dataset
    #data = llm.MockDataModule(seq_length=seq_length, global_batch_size=global_batch_size)

    train_dataset = datasets.load_dataset("parquet", data_files=str("/lustre/fswork/dataset/tulu-3-sft-mixture/data") + '/*.parquet', split="train")
    tokenizer = AutoTokenizer.from_pretrained(str(Path(os.environ["DSDIR"]) / "HuggingFace_Models" / args.model), padding_side="left")
    tokenizer.padding_side = "left"
    
    def wrap_tuple_to_dict(old_collate_fn):
        def new_collate_fn(batch):
            t = old_collate_fn(batch)
            if isinstance(t, (list, tuple)) and len(t) >= 3:
                return {"input_ids": t[0], "attention_mask": t[1], "labels": t[2]}
            raise TypeError(f"collate_fn returned {type(t)}, expected tuple of 3 elements")
        return new_collate_fn
    
    collate_fn = wrap_tuple_to_dict(make_sft_collate(tokenizer, max_seq_length=args.seq_length))

    '''dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=None,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )'''

    model_parallel_group_size = args.tp_size * args.pp_size * args.cp_size
    dp_world_size = args.dp_size
    dp_rank = RANK // model_parallel_group_size


    '''dataloader = add_megatron_sampler(
        dataloader=dataloader,
        micro_batch_size=args.batch_size,
        global_batch_size=args.global_batch_size,
        drop_last=False,
        rank=dp_rank,
        world_size=dp_world_size,     
    )'''

    datamodule = llm.HFDatasetDataModule(
        path_or_dataset=train_dataset,
        split="train",
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        seq_length=args.seq_length,
        # IMPORTANT : batchs
        micro_batch_size=args.batch_size,          # micro_batch
        global_batch_size=args.global_batch_size,  # batch global
        # IMPORTANT : activer le sampler Megatron
        use_mcore_sampler=True,
        mcore_dataloader_type="cyclic",            # par défaut, OK
    )


    ## initialize a small GPT model
    if '14B' in args.model:
        config = llm.Qwen25Config14B()
    elif '32B' in args.model:
        config = llm.Qwen25Config32B()
    elif '72B' in args.model:
        config = llm.Qwen25Config72B()
    else:
        raise TypeError(f"this model is not configured ! Please complete the py script")
        
    model = llm.Qwen2Model(config, tokenizer=tokenizer)
    #model = llm.GPTModel(gpt_config)

    ## initialize the strategy
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=args.virtual_pp_size,
        context_parallel_size=args.cp_size,
        use_tp_pp_dp_mapping=False,
        sequence_parallel=args.sequence_parallel,
        )
    
    ## setup the optimizer
    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=6e-4,
        bf16=True,
    )
    opt = nl.MegatronOptimizerModule(config=opt_config)

    ## callback
    callbacks = [MyChronoCallback(RANK, args.accumulate_grad_batches)]


    ## Note : pour la loss CE le ignore_index=-10 y est par défaut
    trainer = nl.Trainer(
        devices=args.devices, ## you can change the number of devices to suit your setup
        num_nodes=args.num_nodes,
        max_steps=100,
        accelerator="gpu",
        strategy=strategy,
        accumulate_grad_batches=args.accumulate_grad_batches,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        gradient_clip_val=args.grad_clip,
        limit_val_batches=0.0,
        callbacks=callbacks,
        use_distributed_sampler=False
    )

    nemo_logger = nl.NeMoLogger(
        log_dir="test_logdir", ## logs and checkpoints will be written here
    )
    
    llm.train(
        model=model,
        data=datamodule,
        trainer=trainer,
        log=nemo_logger,
        tokenizer='model',
        optim=opt,
    )


if __name__ == "__main__":
    main()
