import os
import datasets
import functools

import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from argparse import ArgumentParser, Namespace, BooleanOptionalAction
from pathlib import Path
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LinearLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.aggregation import RunningMean
from torchmetrics.text import Perplexity
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    make_sft_collate,
    apply_fsdp_checkpointing,
    Chronometer
)

### TP/FSDP2: DTensor & DeviceMesh imports
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._tensor import Shard, Replicate
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    SequenceParallel
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
        print("idr_torch is not installed, but its OK !! :)")
    ## Don't forget to export in slurm file 
    #export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
    #export MASTER_PORT=29500


if RANK == 0:
    print(f">>> Training on {WORLD_SIZE} processes")


def parse_args() -> Namespace:
    parser = ArgumentParser()

    # Memory related arguments
    parser.add_argument('--bsz', "--batch-size", dest="batch_size", default=1, type=int, help='batch size per GPU')
    parser.add_argument('--seq-len', "--seq-length", dest="seq_length", default=4096, type=int, help='sequence length of each sample per GPU')
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
    parser.add_argument("--selective-activation-checkpointing", "--sac",
                        dest="sac",
                        default=None,
                        type=str,
                        help='For a given ac ratio p, we should essentially apply ac on every "1/p" blocks.')

    ### TP/FSDP2: new argument
    parser.add_argument("--tp-size", default=1, type=int, help="Tensor parallel size (must divide WORLD_SIZE)")

    ## Better for slow inter-node connection
    parser.add_argument('--use-ddp', default=False, action='store_true', help='Use DDP instead of FSDP')

    return parser.parse_args()

args = parse_args()

#### To setup (Jean Zay environment)
DSDIR = Path(os.environ["DSDIR"])
model_path = DSDIR / "HuggingFace_Models" / args.model
dataset_path = "/lustre/fswork/dataset/tulu-3-sft-mixture/data"

os.environ["TOKENIZER_PARALLELISM"] = "false"
if args.nccl_profile:
    os.environ["NCCL_DEBUG"] = "INFO"
    #os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,COLL"

chrono = Chronometer(RANK, args.grad_acc)

torch.set_float32_matmul_precision('high')

dist.init_process_group(
    backend="nccl",
    rank=RANK,
    world_size=WORLD_SIZE,
)

torch.cuda.set_device(LOCAL_RANK)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### TP/FSDP2: build 2D mesh (data_parallel, tensor_parallel)
assert WORLD_SIZE % args.tp_size == 0, "WORLD_SIZE must be divisible by tp_size"
TP_SIZE = args.tp_size
DP_SIZE = WORLD_SIZE // TP_SIZE

# mapping rank -> (dp_rank, tp_rank)
tp_rank = RANK % TP_SIZE
dp_rank = RANK // TP_SIZE

world_mesh = init_device_mesh(
    "cuda",
    mesh_shape=(DP_SIZE, TP_SIZE),
    mesh_dim_names=("data", "tensor"),
)

dp_mesh = world_mesh["data"]      # for FSDP2
tp_mesh = world_mesh["tensor"]    # for TP

gbs = args.batch_size * args.grad_acc * DP_SIZE   # GBS en "vraie" data parallel
if RANK == 0:
    print(
        f"world size: {WORLD_SIZE} (DP={DP_SIZE}, TP={TP_SIZE}), "
        f"GBS (data)={gbs}, BSperDev={args.batch_size}, "
        f"sequence length: {args.seq_length}, "
        f"selective AC ratio: {args.sac}, grad accumulation:{args.grad_acc}"
    )


#### Initialize the model and its tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
num_parameters = sum(param.numel() for param in model.parameters())
tokenizer = AutoTokenizer.from_pretrained(str(model_path), padding_side="left")
####

### Gradient Checkpointing (sélectif)
if args.sac:
    model.config.use_cache = False
    BlockCls = type(model.model.layers[0])
    apply_fsdp_checkpointing(model, BlockCls, args.sac)


# parallelize the first embedding and the last linear out projection
# To build according to model
# Here with Qwen2
'''
Qwen2ForCausalLM(
  (model): Qwen2Model(
    (embed_tokens): Embedding(152064, 5120)
    (layers): ModuleList(
      (0-47): 48 x Qwen2DecoderLayer(
        (self_attn): Qwen2Attention(
          (q_proj): Linear(in_features=5120, out_features=5120, bias=True)
          (k_proj): Linear(in_features=5120, out_features=1024, bias=True)
          (v_proj): Linear(in_features=5120, out_features=1024, bias=True)
          (o_proj): Linear(in_features=5120, out_features=5120, bias=False)
        )
        (mlp): Qwen2MLP(
          (gate_proj): Linear(in_features=5120, out_features=13824, bias=False)
          (up_proj): Linear(in_features=5120, out_features=13824, bias=False)
          (down_proj): Linear(in_features=13824, out_features=5120, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen2RMSNorm((5120,), eps=1e-05)
        (post_attention_layernorm): Qwen2RMSNorm((5120,), eps=1e-05)
      )
    )
    (norm): Qwen2RMSNorm((5120,), eps=1e-05)
    (rotary_emb): Qwen2RotaryEmbedding()
  )
  (lm_head): Linear(in_features=5120, out_features=152064, bias=False)
)
'''

if args.tp_size > 1:
    model = parallelize_module(
        model,
        tp_mesh,
        {
            "model.embed_tokens": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "model.norm": SequenceParallel(),
            "lm_head": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Replicate()
            ),
        }
    )
    
    for layer in model.model.layers:
        layer_tp_plan = {
            "input_layernorm": SequenceParallel(),   # ou self_attn_layer_norm si ça s’appelle comme ça
            #"self_attn": PrepareModuleInput(
            #    # à adapter selon la signature de forward de Qwen2Attention
            #    input_layouts=(Shard(1), Replicate(), Replicate(), Replicate()),
            #    desired_input_layouts=(Replicate(), Replicate(), Replicate(), Replicate()),
            #),
            #"self_attn.q_proj": ColwiseParallel(use_local_output=False),
            #"self_attn.k_proj": ColwiseParallel(use_local_output=False),
            #"self_attn.v_proj": ColwiseParallel(use_local_output=False),
            #"self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
            #"post_attention_layernorm": SequenceParallel(),
            "mlp": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "mlp.gate_proj": ColwiseParallel(),
            "mlp.up_proj": ColwiseParallel(),
            "mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
        }
    
        # Custom parallelization plan for the model
        parallelize_module(
            module=layer,
            device_mesh=tp_mesh,
            parallelize_plan=layer_tp_plan
        )
    
    
    """from torch.distributed._tensor import distribute_tensor
    from torch.distributed._tensor.placement_types import Shard
    
    def apply_tensor_parallel_to_linears(module, tp_mesh):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.Linear):
                w = child.weight.data
                w_dt = distribute_tensor(w, tp_mesh, placements=[Shard(0)])
                child.weight = torch.nn.Parameter(w_dt)
                if child.bias is not None:
                    b = child.bias.data
                    b_dt = distribute_tensor(b, tp_mesh, placements=[Shard(0)])
                    child.bias = torch.nn.Parameter(b_dt)
            else:
                apply_tensor_parallel_to_linears(child, tp_mesh)
    
    apply_tensor_parallel_to_linears(model.model, tp_mesh)"""



#### Distribute the Model (FSDP2 sur la dimension data_parallel)
if args.use_ddp:
    model = DistributedDataParallel(model, device_ids=[LOCAL_RANK])
else:
    fsdp_kwargs = {}
    fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16
    )
    
    # IMPORTANT: on shard les "layers" puis le modèle complet,
    # en indiquant le mesh data-parallel.
    for layer in model.model.layers:
        fully_shard(layer, mesh=dp_mesh, **fsdp_kwargs)
    fully_shard(model, mesh=dp_mesh, **fsdp_kwargs)

model = model.to(device)

#### JIT
if args.compile:
    model = torch.compile(model)

if RANK == 0:
    print(f"number of parameters: {num_parameters}")
    print(f'Pre-loop Model MaxMemory for GPU:{RANK} {torch.cuda.max_memory_allocated()/2**30} GBytes')


#### Data Loading
train_dataset = datasets.load_dataset("parquet", data_files=str(dataset_path) + '/*.parquet', split="train")
collate_fn = make_sft_collate(tokenizer, max_seq_length=args.seq_length)

### TP/FSDP2: sampler sur groupes data_parallel, pas sur WORLD_SIZE
sampler = DistributedSampler(
    dataset=train_dataset,
    rank=dp_rank,          # data-parallel rank
    num_replicas=DP_SIZE,  # data-parallel size
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

#### Training step
criterion = CrossEntropyLoss(ignore_index=-100)
optimizer = AdamW(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, eps=1e-05, foreach=True)   ## Here Add foreach=True

if RANK == 0:
    print(f'global batch size (data parallel only): {args.batch_size * DP_SIZE} - mini batch size: {args.batch_size}')
    print(f"DATALOADER {args.num_workers} {args.persistent_workers} {args.pin_memory} {args.non_blocking} {args.prefetch_factor} {args.drop_last} ")
    print(f"Optimizer: {optimizer}")

lr_warmup_iters = int(len(dataloader) * args.lr_warmup_ratio) * args.epochs / args.grad_acc
warmup_lr_scheduler = LinearLR(optimizer, start_factor=1e-9, end_factor=1, total_iters=lr_warmup_iters)
annealing_lr_scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloader) * args.epochs / args.grad_acc - lr_warmup_iters, eta_min=0.)
lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_lr_scheduler, annealing_lr_scheduler], milestones=[lr_warmup_iters])

loss_metric = RunningMean(window=5).to(device)
perplexity = Perplexity(ignore_index=-100).to(device)

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


#### Training loop
chrono.start()
chrono.dataload()
if RANK == 0:
    chrono.tac_time(clear=True)

for i, (input_ids, attention_mask, labels) in enumerate(dataloader, start=1):
    if args.test and i > args.test_nsteps * args.grad_acc:
        break

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
            # NOTE: on ne divise plus par WORLD_SIZE ici :
            # FSDP2 + loss moyenne -> gradients corrects.

        loss_metric.update(loss.detach())
        perplexity.update(logits.detach(), labels)

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
                print(
                    f"Step {step} / {args.test_nsteps if args.test else len(dataloader) // args.grad_acc} | "
                    f"Loss: {L.item():.3f} | Perplexity: {perp.item():.3f} | "
                    f"LR: {last_lr:0.3e} | Wall: {chrono.tac_time()}"
                )

        chrono.dataload()

chrono.display()
dist.barrier()
if RANK == 0:
    print(f'MaxMemory for GPU:{RANK} {torch.cuda.max_memory_allocated()/2**30} GBytes')

dist.barrier()
dist.destroy_process_group()
