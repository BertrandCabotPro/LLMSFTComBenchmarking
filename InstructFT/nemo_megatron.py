#!/usr/bin/python3

import os
import tempfile
from pathlib import Path

import fiddle as fdl
import functools
#import lightning.pytorch as pl

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.gpt.data.hf_dataset import HFDatasetDataModule
from megatron.core.optimizer import OptimizerConfig

from transformers import AutoTokenizer

import datasets
from utils import (
    make_sft_collate,
    Chronometer,
    MyChronoCallback
)

#import idr_torch
RANK = int(os.environ['SLURM_PROCID'])


def main():
    import argparse
    import math

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-72B-Instruct',
                        help='HF model id (ex: Qwen/Qwen2.5-32B-Instruct)')

    # cluster / parallelism
    parser.add_argument('--devices', type=int, default=2, help='Number of GPUs per node')
    parser.add_argument('--num-nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--dp-size', type=int, default=None, help='Data Parallel size (auto if None)')
    parser.add_argument('--tp-size', type=int, default=1, help='Tensor Parallel size')
    parser.add_argument('--pp-size', type=int, default=1, help='Pipeline Parallel size (Megatron)')
    parser.add_argument('--cp-size', type=int, default=1, help='Context Parallel size')
    parser.add_argument('--virtual-pp-size', type=int, default=None, help='Interleaved PP size')
    parser.add_argument('--ep-size', type=int, default=None, help='Expert Parallel size (MoE)')
    parser.add_argument('--sequence-parallel', action='store_true',
                        help='Use sequence parallel (when TP>1)')

    # training & logging
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--accumulate-grad-batches', '--accumulate_grad_batches',
                        dest='accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--max-epochs', type=int, default=2)
    parser.add_argument('--log-every-n-steps', type=int, default=10)
    parser.add_argument('--limit-val-batches', type=float, default=0.0)
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

    if args.batch_size is None:
        args.batch_size = args.global_batch_size // (args.dp_size * args.accumulate_grad_batches)
    if RANK==0: print(f"GBS = {args.global_batch_size} - DP={args.dp_size}, BS per Replica={args.accumulate_grad_batches*args.batch_size}, PP_microBS={args.batch_size}, PP_nchunks={args.accumulate_grad_batches}")

    DSDIR = Path(os.environ["DSDIR"])
    model_path = DSDIR / "HuggingFace_Models" / args.model

     # 1) Construire la recipe selon la taille
    if args.model == 'Qwen/Qwen2.5-14B-Instruct':
        recipe = llm.recipes.qwen25_14b.finetune_recipe(
            name=args.model,
            dir=args.ckpt_folder,              # dossier de travail/checkpoints
            num_nodes=args.num_nodes,
            num_gpus_per_node=args.devices,
            peft_scheme='none',                # SFT complet (pas LoRA)
            packed_sequence=False,             # tu peux mettre True si tu packs
        )
    elif args.model == 'Qwen/Qwen2.5-32B-Instruct':
        recipe = llm.recipes.qwen25_32b.finetune_recipe(
            name=args.model,
            dir=args.ckpt_folder,              # dossier de travail/checkpoints
            num_nodes=args.num_nodes,
            num_gpus_per_node=args.devices,
            peft_scheme='none',                # SFT complet (pas LoRA)
            packed_sequence=False,             # tu peux mettre True si tu packs
        )
    elif args.model == 'Qwen/Qwen2.5-72B-Instruct':
        recipe = llm.recipes.qwen25_72b.finetune_recipe(
            name=args.model,
            dir=args.ckpt_folder,              # dossier de travail/checkpoints
            num_nodes=args.num_nodes,
            num_gpus_per_node=args.devices,
            peft_scheme='none',                # SFT complet (pas LoRA)
            packed_sequence=False,             # tu peux mettre True si tu packs
        )

    # 3D parallelism depuis tes flags
    recipe.trainer.strategy.tensor_model_parallel_size   = args.tp_size
    recipe.trainer.strategy.pipeline_model_parallel_size = args.pp_size
    recipe.trainer.strategy.virtual_pipeline_model_parallel_size = args.virtual_pp_size
    recipe.trainer.strategy.context_parallel_size = args.cp_size
    recipe.trainer.strategy.sequence_parallel = args.sequence_parallel
    # DP sera déduit par le monde: DP = (devices*num_nodes)/(TP*PP) si tu utilises la recipe "run".

    # 4) Precision / accélérateurs (aligne avec ton script)
    recipe.trainer.accumulate_grad_batches = args.accumulate_grad_batches
    recipe.trainer.log_every_n_steps = args.log_every_n_steps
    recipe.trainer.max_steps = args.max_steps
    

    # 5) Données: réutilise ton parquet + collate
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
    
    collate_fn = wrap_tuple_to_dict(make_sft_collate(tokenizer, max_seq_length=args.seq_length + 1))
    
    dataset = HFDatasetDataModule(
        train_dataset,
        split="train",
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        seq_length=args.seq_length,
        micro_batch_size=args.batch_size,
        pad_seq_len_divisible=None,
    )

    # <- ajoute l’attribut attendu par la recipe
    setattr(dataset, "global_batch_size", int(args.global_batch_size))

    recipe.data = dataset


    # 8) Logger + callbacks (tes moniteurs)
    #recipe.log.ckpt=None
    recipe.log.tensorboard=None
    callbacks = []
    recipe.trainer.callbacks = callbacks

    if RANK==0: print(recipe)
    # 10) fit via llm.api.finetune (comme chez toi), mais avec modèle NeMo Megatron
    
    product = fdl.build(recipe)           # peut être un partial selon la recette
    if RANK==0: print(product)
    if isinstance(product, functools.partial):
        product = product()               # <-- matérialise les objets réels
    if RANK==0: print(product)
    # maintenant 'product' peut être:
    # - un namespace/obj avec des attrs
    # - un tuple
    # - un dict
    if hasattr(product, "model"):
        if RANK==0: print('YES-Attributs')
        model   = product.model
        data    = product.data
        trainer = product.trainer
        optim   = getattr(product, "optim", None)
        resume  = getattr(product, "resume", None)
    elif isinstance(product, tuple):
        if RANK==0: print('YES-Tuple')
        # ordre courant : (model, data, trainer, strategy, optim, resume)
        model, data, trainer, strategy, optim, resume = product + (None,) * max(0, 6 - len(product))
    elif isinstance(product, dict):
        if RANK==0: print('YES-Dico !!')
        model   = product["model"]
        data    = product["data"]
        trainer = product["trainer"]
        optim   = product.get("optim")
        resume  = product.get("resume")
    else:
        raise TypeError(f"Unexpected built type: {type(product)}")

    
    llm.api.train(model=model, data=data, trainer=trainer, optim=optim, resume=resume)
    # ou: llm.api.finetune(...)
    #from nemo.lightning import run as nemo_run
    #nemo_run(recipe)
    
    
    return  # <- on sort : la branche recipe a fait l’entraînement    


if __name__ == '__main__':
    main()
