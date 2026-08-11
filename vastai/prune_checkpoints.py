"""
Background watcher that deletes old local checkpoint steps, keeping only the most recent N.

nanochat's save_checkpoint() (nanochat/checkpoint_manager.py) has no retention policy -- every
--save-every interval is kept on local disk forever (model_STEP.pt + meta_STEP.json +
optim_STEP_rank0.pt, ~500-650MB per step depending on model size). A long run on a small rented
disk fills it and corrupts an in-progress checkpoint write (see docs/RESEARCH_LOG.md 2026-08-11
"A10 pretrain complete on Vast.ai" -- this happened once already). All steps are already backed
up to Drive by kaggle/sync_checkpoints.py, so only the last few need to stay local for a resume.

Deliberately conservative: only prunes a step once at least `--min-age` seconds have passed
since it was written, so it isn't deleted before sync_checkpoints.py's own poll interval has had
a chance to upload it. Never touches the single most recent step.

Usage (started in the background before/alongside training, alongside sync_checkpoints.py):
    python vastai/prune_checkpoints.py --model-tag a9 --checkpoint-type base --keep 3 --interval 60 &
    PRUNE_PID=$!
    ... run training ...
    kill $PRUNE_PID
"""
import argparse
import os
import re
import sys
import time

STEP_RE = re.compile(r"model_(\d+)\.pt$")


def find_steps(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        return []
    steps = []
    for f in os.listdir(checkpoint_dir):
        m = STEP_RE.search(f)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def prune_once(checkpoint_dir, keep, min_age, log):
    steps = find_steps(checkpoint_dir)
    if len(steps) <= keep:
        return
    # Never prune the most recent `keep` steps, and never prune a step younger than min_age
    # (gives sync_checkpoints.py time to have uploaded it first).
    now = time.time()
    candidates = steps[:-keep] if keep > 0 else steps
    for step in candidates:
        step_str = f"{step:06d}"
        model_path = os.path.join(checkpoint_dir, f"model_{step_str}.pt")
        try:
            age = now - os.path.getmtime(model_path)
        except FileNotFoundError:
            continue
        if age < min_age:
            continue
        removed = []
        for pattern in (f"model_{step_str}.pt", f"meta_{step_str}.json"):
            p = os.path.join(checkpoint_dir, pattern)
            if os.path.exists(p):
                os.remove(p)
                removed.append(pattern)
        # Optimizer files are per-rank -- remove all ranks present.
        for f in os.listdir(checkpoint_dir):
            if f.startswith(f"optim_{step_str}_rank"):
                os.remove(os.path.join(checkpoint_dir, f))
                removed.append(f)
        if removed:
            log(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] pruned step {step} ({len(removed)} files, age {age:.0f}s)")


def main():
    parser = argparse.ArgumentParser(description="Keep only the last N local checkpoint steps (older ones stay on Drive)")
    parser.add_argument("--model-tag", type=str, required=True)
    parser.add_argument("--checkpoint-type", type=str, default="base", choices=["base", "sft", "rl"],
                         help="base_checkpoints | chatsft_checkpoints | chatrl_checkpoints")
    parser.add_argument("--keep", type=int, default=3, help="number of most recent steps to keep locally")
    parser.add_argument("--min-age", type=int, default=90, help="seconds a step must exist before it's eligible for pruning (give sync a head start)")
    parser.add_argument("--interval", type=int, default=60, help="seconds between prune polls")
    parser.add_argument("--once", action="store_true", help="run a single prune pass and exit")
    parser.add_argument("--log-file", type=str, default=None)
    args = parser.parse_args()

    base_dir = os.environ.get("NANOCHAT_BASE_DIR")
    if not base_dir:
        print("NANOCHAT_BASE_DIR is not set", file=sys.stderr)
        sys.exit(1)

    subdir = {"base": "base_checkpoints", "sft": "chatsft_checkpoints", "rl": "chatrl_checkpoints"}[args.checkpoint_type]
    checkpoint_dir = os.path.join(base_dir, subdir, args.model_tag)

    log_fh = open(args.log_file, "a") if args.log_file else None

    def log(msg):
        print(msg, flush=True)
        if log_fh:
            log_fh.write(msg + "\n")
            log_fh.flush()

    log(f"prune_checkpoints: dir={checkpoint_dir} keep={args.keep} min_age={args.min_age}s interval={args.interval}s once={args.once}")

    if args.once:
        prune_once(checkpoint_dir, args.keep, args.min_age, log)
        return

    while True:
        prune_once(checkpoint_dir, args.keep, args.min_age, log)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
