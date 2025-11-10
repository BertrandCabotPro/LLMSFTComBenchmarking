import os
import datasets
import functools

import time
import torch
import torch.distributed as dist
from argparse import ArgumentParser, Namespace, BooleanOptionalAction
from pathlib import Path
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LinearLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.aggregation import RunningMean
from torchmetrics.text import Perplexity
from transformers import AutoModelForCausalLM, AutoTokenizer
#from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from utils import (
    make_sft_collate,
    apply_fsdp_checkpointing,
    Chronometer
)

try:
    import idr_torch
    RANK = idr_torch.rank
    LOCAL_RANK = idr_torch.local_rank
    WORLD_SIZE = idr_torch.world_size
except:
    RANK = int(os.environ['SLURM_PROCID'])
    LOCAL_RANK = int(os.environ['SLURM_LOCALID'])
    WORLD_SIZE = int(os.environ['SLURM_NTASKS'])

    



if RANK == 0:
    print(f">>> Training on {WORLD_SIZE} processes")

def parse_args() -> Namespace:
    parser = ArgumentParser()

    # Memory related arguments
    parser.add_argument('--bsz', "--batch-size", dest="batch_size", default=1, type=int, help='batch size per GPU')
    parser.add_argument('--seq-len', default=4096, type=int, help='sequence length of each sample per GPU')
    parser.add_argument('--grad-acc', default=2, type=int, help='Gradient Accumulation count')
    parser.add_argument('--epochs', default=2, type=int, help='Number of epochs')

    # SIGINT
    parser.add_argument('--test', default=False, action='store_true', help='Test 100 iterations')
    parser.add_argument('--test-nsteps', default=100, type=int, help='the number of steps in test mode')

    # JIT related arguments
    parser.add_argument("--compile", default=False, action=BooleanOptionalAction, help="whether or not to compile model")
    parser.add_argument("--compile-warmup-steps", default=10, type=int, help="number of steps to warm up compilation")

    # DataLoader related arguments
    parser.add_argument('--num-workers', default=4, type=int, help='num workers in dataloader')
    parser.add_argument('--persistent-workers', default=False, action=BooleanOptionalAction, help='activate persistent workers in dataloader')
    parser.add_argument('--pin-memory', default=True, action=BooleanOptionalAction, help='activate pin memory option in dataloader')
    parser.add_argument('--non-blocking', default=True, action=BooleanOptionalAction, help='activate asynchronuous GPU transfer')
    parser.add_argument('--prefetch-factor', default=3, type=int, help='prefectch factor in dataloader')
    parser.add_argument('--drop-last', default=False, action=BooleanOptionalAction, help='activate drop_last option in dataloader')

    # Training related arguments
    parser.add_argument("--lr-warmup-ratio", default=0.1, type=float, help="linear warmup of learning rate before cosine annealing")
    parser.add_argument("--lr", "--learning-rate", dest="learning_rate", type=float, default=1e-5, help="learning rate for adamw")
    parser.add_argument("--wd", "--weight-decay", dest="weight_decay", type=float, default=0.1, help="weight decay for adamw")

    # Other
    parser.add_argument("--nccl-profile", default=False, action=BooleanOptionalAction, help="whether or not to profile nccl communications. Clutters stdout heavily.")
    parser.add_argument("--model", default='Qwen/Qwen2.5-32B-Instruct', type=str, help="HuggingFaceHub Model's Name")
    parser.add_argument("--fsdp-checkpointing", default=None, type=str, help='For a given ac ratio p, we should essentially apply ac on every "1/p" blocks.')
    
    return parser.parse_args()


args = parse_args()
os.environ["TOKENIZER_PARALLELISM"] = "false"
if args.nccl_profile:
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,COLL"

chrono = Chronometer(RANK, args.grad_acc)

torch.set_float32_matmul_precision('high')

dist.init_process_group(
    backend="nccl",
    rank=RANK,
    world_size=WORLD_SIZE,
)

DSDIR = Path(os.environ["DSDIR"])
model_path = DSDIR / "HuggingFace_Models" / args.model
#dataset_path = DSDIR / "HuggingFace" / "allenai" / "tulu-3-sft-mixture" / "data"
dataset_path = "/lustre/fswork/dataset/tulu-3-sft-mixture/data"
torch.cuda.set_device(LOCAL_RANK)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#### Initialize the model and its tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="bfloat16")
num_parameters = sum(param.numel() for param in model.parameters())
tokenizer = AutoTokenizer.from_pretrained(str(model_path), padding_side="left")
####

### Gradient Checkpointing
if args.fsdp_checkpointing:
    model.config.use_cache = False
    BlockCls = type(model.model.layers[0])
    apply_fsdp_checkpointing(model, BlockCls, args.fsdp_checkpointing)

#### Distribute the Model
fsdp_kwargs = {}
fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16
        )

for layer in model.model.layers:
    fully_shard(layer, **fsdp_kwargs)
fully_shard(model, **fsdp_kwargs)

model = model.to(device)

####

#### JIT
if args.compile:
    model = torch.compile(model)
####

if RANK == 0:
    #print(f"model: {model}")
    print(f"number of parameters: {num_parameters}")

print(f'Pre-loop Model MaxMemory for GPU:{RANK} {torch.cuda.max_memory_allocated()} Bytes')


#### Data Loading
train_dataset = datasets.load_dataset("parquet", data_files=str(dataset_path) + '/*.parquet', split="train")  # 
collate_fn = make_sft_collate(tokenizer, max_seq_length=args.seq_len+1)

sampler = DistributedSampler(
    dataset=train_dataset,
    rank=RANK,
    num_replicas=WORLD_SIZE,
    shuffle=True,
)

dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    collate_fn=collate_fn,
    pin_memory=args.pin_memory,
    drop_last=args.drop_last,
    persistent_workers=args.persistent_workers,
    prefetch_factor=args.prefetch_factor,
    sampler=sampler,
)
####


#### Training step
criterion = CrossEntropyLoss(ignore_index=-100)
optimizer = AdamW(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, eps=1e-05)

if RANK == 0:
    print(f'global batch size: {args.batch_size * WORLD_SIZE} - mini batch size: {args.batch_size}')
    print(f"DATALOADER {args.num_workers} {args.persistent_workers} {args.pin_memory} {args.non_blocking} {args.prefetch_factor} {args.drop_last} ")
    print(f"Optimizer: {optimizer}")

lr_warmup_iters = int(len(dataloader) * args.lr_warmup_ratio)  * args.epochs / args.grad_acc
warmup_lr_scheduler = LinearLR(optimizer, start_factor=1e-9, end_factor=1, total_iters=lr_warmup_iters)
annealing_lr_scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloader) * args.epochs / args.grad_acc - lr_warmup_iters, eta_min=0.)
lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_lr_scheduler, annealing_lr_scheduler], milestones=[lr_warmup_iters])

loss_metric = RunningMean(window=5).to(device)
perplexity = Perplexity(ignore_index=-100).to(device)
####

#### Compile warmup
for i, (input_ids, attention_mask, labels) in enumerate(dataloader, start=1):
    if not args.compile or i > args.compile_warmup_steps:
        break
    input_ids = input_ids.to(device, non_blocking=args.non_blocking)
    attention_mask = attention_mask.to(device, non_blocking=args.non_blocking)
    labels = labels.to(device, non_blocking=args.non_blocking)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids, attention_mask=attention_mask).logits
        bsz, seq_len, vocab_size = logits.shape
        loss = criterion(logits.view(bsz * seq_len, vocab_size), labels.view(bsz * seq_len))
    loss.backward()
####



#### Training loop
chrono.start()
chrono.dataload()
if RANK == 0: chrono.tac_time(clear=True)
for i, (input_ids, attention_mask, labels) in enumerate(dataloader, start=1):
    if args.test and i > args.test_nsteps * args.grad_acc: break

    input_ids = input_ids.to(device, non_blocking=args.non_blocking)
    attention_mask = attention_mask.to(device, non_blocking=args.non_blocking)
    labels = labels.to(device, non_blocking=args.non_blocking)

    chrono.dataload()
    chrono.training()
    chrono.forward()

     # passes and weights update
    with torch.set_grad_enabled(True):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits: torch.Tensor = model(input_ids, attention_mask=attention_mask).logits
            bsz, seq_len, vocab_size = logits.shape
            loss: torch.Tensor = criterion(logits.view(bsz * seq_len, vocab_size), labels.view(bsz * seq_len))
            loss /= WORLD_SIZE
    
    
        loss_metric.update(loss)
        perplexity.update(logits, labels)
    

        chrono.forward()
        chrono.backward()
        loss.backward()
        if i % args.grad_acc == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
        optimizer.zero_grad()
        chrono.backward()
        chrono.training()
    
        step = (i // args.grad_acc) + 1
        if step % 10 == 0 and i % args.grad_acc == 0:
            L = loss_metric.compute()
            perp = perplexity.compute()
            last_lr = lr_scheduler.get_last_lr()[0]
            if RANK == 0:
                print(f"Step {step} / {args.test_nsteps if args.test else len(dataloader) // args.grad_acc} | Loss: {L.item():.3f} | Perplexity: {perp.item():.3f} | LR: {last_lr:0.3e} | Wall: {chrono.tac_time()}")

        chrono.dataload()
####

chrono.display()
dist.barrier()
if RANK == 0:
    print(f'Max Memory Allocated {torch.cuda.max_memory_allocated()} Bytes')
else:
    print(f'MaxMemory for GPU:{RANK} {torch.cuda.max_memory_allocated()} Bytes')

dist.barrier()
dist.destroy_process_group()