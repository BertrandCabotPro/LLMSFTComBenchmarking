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
from megatron.core import num_microbatches_calculator as nm
import torch.distributed as dist

RANK = int(os.environ['SLURM_PROCID'])

seq_length = 4096
global_batch_size = 128
batch_size = 8
model = "Qwen/Qwen2.5-14B-Instruct"
accumulate_grad_batches=1


class InitNumMicrobatchesCallback(pl.Callback):
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not dist.is_initialized():
            dist.init_process_group("nccl")

        dp_size = 16
        mb_size = 8
        gbs = 128

        print(nm._GLOBAL_NUM_MICROBATCHES_CALCULATOR)

        nm.init_num_microbatches_calculator(
            rank=dist.get_rank(),
            rampup_batch_size=None,
            global_batch_size=gbs,
            micro_batch_size=mb_size,
            data_parallel_size=dp_size,
        )
        print(
            f"[Rank {dist.get_rank()}] Initialized num_microbatches="
            f"{nm.get_num_microbatches()} (gbs={gbs}, micro={mb_size}, dp={dp_size})"
        )

        print(nm._GLOBAL_NUM_MICROBATCHES_CALCULATOR)


if __name__ == "__main__":
    

    ## setup the dummy dataset
    #data = llm.MockDataModule(seq_length=seq_length, global_batch_size=global_batch_size)

    train_dataset = datasets.load_dataset("parquet", data_files=str("/lustre/fswork/dataset/tulu-3-sft-mixture/data") + '/*.parquet', split="train")
    tokenizer = AutoTokenizer.from_pretrained(str(Path(os.environ["DSDIR"]) / "HuggingFace_Models" / model), padding_side="left")
    tokenizer.padding_side = "left"
    
    def wrap_tuple_to_dict(old_collate_fn):
        def new_collate_fn(batch):
            t = old_collate_fn(batch)
            if isinstance(t, (list, tuple)) and len(t) >= 3:
                return {"input_ids": t[0], "attention_mask": t[1], "labels": t[2]}
            raise TypeError(f"collate_fn returned {type(t)}, expected tuple of 3 elements")
        return new_collate_fn
    
    collate_fn = wrap_tuple_to_dict(make_sft_collate(tokenizer, max_seq_length=seq_length + 1))
    
    data = llm.HFDatasetDataModule(
        train_dataset,
        split="train",
        collate_fn=collate_fn,
        num_workers=4,
        seq_length=seq_length,
        micro_batch_size=batch_size,
        pad_seq_len_divisible=None,
    )

    # <- ajoute l’attribut attendu par la recipe
    setattr(data, "global_batch_size", int(global_batch_size))
    

    ## initialize a small GPT model
    config = llm.Qwen25Config14B()
    
    model = llm.Qwen2Model(config, tokenizer=tokenizer)
    #model = llm.GPTModel(gpt_config)

    ## initialize the strategy
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        use_tp_pp_dp_mapping=False,
        sequence_parallel=False,
    )

    ## setup the optimizer
    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=6e-4,
        bf16=True,
    )
    opt = nl.MegatronOptimizerModule(config=opt_config)

    ## callback
    callbacks = [MyChronoCallback(RANK, accumulate_grad_batches)]


    trainer = nl.Trainer(
        devices=4, ## you can change the number of devices to suit your setup
        num_nodes=8,
        max_steps=100,
        accelerator="gpu",
        strategy=strategy,
        accumulate_grad_batches=accumulate_grad_batches,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        limit_val_batches=0.0,
        callbacks=callbacks,
    )

    nemo_logger = nl.NeMoLogger(
        log_dir="test_logdir", ## logs and checkpoints will be written here
    )
    
    llm.train(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        tokenizer='model',
        optim=opt,
    )