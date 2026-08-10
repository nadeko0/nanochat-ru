"""
Finds the largest --device-batch-size that fits in VRAM for a given --depth,
without committing to a full multi-hour training run to find out via OOM crash.

Builds the real GPT model + real Muon/AdamW optimizer (same as scripts/base_train.py)
on a single GPU, torch.compiles it, and runs a few forward/backward/step iterations on
random token batches at decreasing batch sizes via accelerate.utils.find_executable_batch_size
(handles the OOM-retry edge cases -- clearing the CUDA cache and running gc.collect()
between attempts -- so a failed attempt doesn't just fragment memory and produce a false
negative on the next one).

Tests on a single GPU: DDP replicates the model per-GPU, so per-GPU memory usage doesn't
depend on world_size, only on local batch size -- no need to launch under torchrun for this.

Usage:
    python kaggle/vram_probe.py --depth=5 --max-seq-len=2048 --starting-batch-size=32
"""
import argparse
import os
import sys

# Running this as `python kaggle/vram_probe.py` (not `python -m ...`) puts this file's own
# directory (kaggle/) on sys.path[0], not the repo root -- so the nanochat/ package next to
# it wouldn't otherwise be importable. Add the repo root explicitly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nanochat.gpt import GPT, GPTConfig

parser = argparse.ArgumentParser(description="Find the largest device-batch-size that fits in VRAM")
parser.add_argument("--depth", type=int, required=True)
parser.add_argument("--aspect-ratio", type=int, default=64)
parser.add_argument("--head-dim", type=int, default=128)
parser.add_argument("--vocab-size", type=int, default=32768)
parser.add_argument("--max-seq-len", type=int, default=2048)
parser.add_argument("--window-pattern", type=str, default="L")
parser.add_argument("--starting-batch-size", type=int, default=32, help="upper bound to search down from")
args = parser.parse_args()

try:
    from accelerate.utils import find_executable_batch_size
except ImportError:
    raise SystemExit("Run `pip install accelerate` first (diagnostic-only dependency, not in pyproject.toml).")

assert torch.cuda.is_available(), "This probe needs a GPU -- run it on a Kaggle GPU session."
device = torch.device("cuda")

base_dim = args.depth * args.aspect_ratio
model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
num_heads = model_dim // args.head_dim
config = GPTConfig(
    sequence_len=args.max_seq_len, vocab_size=args.vocab_size,
    n_layer=args.depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
    window_pattern=args.window_pattern,
)
print(f"depth={args.depth} model_dim={model_dim} n_head={num_heads}")

model = GPT(config).to(device)
model.init_weights()
optimizer = model.setup_optimizer(unembedding_lr=0.004, embedding_lr=0.3, matrix_lr=0.02, weight_decay=0.0)
compiled_model = torch.compile(model, dynamic=False)

found_batch_size = None
peak_memory_mib = None

@find_executable_batch_size(starting_batch_size=args.starting_batch_size)
def probe(batch_size):
    global found_batch_size, peak_memory_mib
    torch.cuda.reset_peak_memory_stats()
    x = torch.randint(0, args.vocab_size, (batch_size, args.max_seq_len), device=device, dtype=torch.int32)
    y = torch.randint(0, args.vocab_size, (batch_size, args.max_seq_len), device=device, dtype=torch.int64)
    for _ in range(3):  # a few steps -- first one compiles, later ones give a steadier peak
        loss = compiled_model(x, y)
        loss.backward()
        optimizer.step()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    found_batch_size = batch_size
    peak_memory_mib = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"  batch_size={batch_size} OK, peak memory: {peak_memory_mib:.0f} MiB")

probe()

print(f"\nLargest working --device-batch-size for depth={args.depth}, max-seq-len={args.max_seq_len}: {found_batch_size}")
print(f"Peak memory at that batch size: {peak_memory_mib:.0f} MiB / {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024:.0f} MiB")
