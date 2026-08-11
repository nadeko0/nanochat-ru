# Running on a rented GPU (Vast.ai) instead of Kaggle

Kaggle's free T4x2 is pre-Ampere (SM75): no bf16, no Flash Attention, and
`nanochat` falls back to fp32 + PyTorch SDPA there (see the runtime warnings
in every Kaggle run log). A rented Ampere/Ada-class GPU (RTX 4070 Ti/5070/
4090, etc.) supports both, and is a single GPU (no DDP/cross-GPU sync
overhead like the Kaggle T4x2 runs had) -- plausibly several times faster
for the same work, though this hasn't actually been measured yet.

Rented boxes give direct SSH access, which is simpler than Kaggle in one
way: no `kaggle_secrets` API, no notebook cells -- just a shell.

## 1. Rent an instance

1. [cloud.vast.ai](https://cloud.vast.ai/) -> Search / Create -> filter by
   GPU type. For this project's model sizes (tens of millions of params),
   a 12-24GB single consumer Ampere/Ada card is plenty -- no need for the
   48GB/96GB/179GB listings, those are wasted money here. Cheap, decent-
   reliability (>99%) options: RTX 4070 Ti, RTX 5070, RTX 3090, ~$0.09-0.16/hr.
2. Pick a **PyTorch template** when renting (has CUDA + PyTorch preinstalled;
   `run_a10.sh` still reinstalls the exact pinned versions via `uv`, so the
   base image's exact PyTorch version doesn't matter much).
3. Give it enough disk (dataset shards + checkpoints + deps are a few GB;
   20-30GB is comfortable) -- **disk size can't be changed after creation**.
4. Click Rent. A fresh (uncached) image pull can take 10-60 minutes; a
   cached one starts in seconds.

## 2. SSH access

1. Generate a key locally if you don't have one:
   `ssh-keygen -t ed25519 -C "your_email@example.com"`
2. Upload the **public** key (`~/.ssh/id_ed25519.pub`) at
   [cloud.vast.ai/manage-keys](https://cloud.vast.ai/manage-keys/) --
   this only applies to instances created *after* the key is added.
3. Once the instance is running, click its SSH icon for the exact command,
   e.g. `ssh -p 20544 root@<host>`. You land in a `tmux` session by default
   -- if the SSH connection drops, `ssh` back in and run `tmux attach` to
   pick the running job back up rather than losing it.

## 3. Get the checkpoint credentials onto the box

Same 4 values as the Kaggle Secrets (`GDRIVE_CLIENT_ID`,
`GDRIVE_CLIENT_SECRET`, `GDRIVE_OAUTH_TOKEN`, `GDRIVE_FOLDER_ID` -- see
[docs/RCLONE_GDRIVE_SETUP.md](RCLONE_GDRIVE_SETUP.md) for what these are),
but as plain environment variables instead of Kaggle Secrets, since there's
no equivalent secrets API here:

```bash
export GDRIVE_CLIENT_ID="..."
export GDRIVE_CLIENT_SECRET="..."
export GDRIVE_OAUTH_TOKEN='...'   # single-quote: the token is a JSON blob with double quotes inside
export GDRIVE_FOLDER_ID="..."
```

Type these directly in the SSH session (or paste into a local `.env` file
you `source`, kept out of git same as always) -- never put them in a
committed file or a shell history that leaves the box.

## 4. Run

```bash
curl -sL https://raw.githubusercontent.com/nadeko0/nanochat-ru/master/vastai/run_a10.sh -o run_a10.sh
chmod +x run_a10.sh
./run_a10.sh
```

Rented a 2-GPU box instead of one? `NUM_GPUS=2 ./run_a10.sh` (uses `torchrun`
instead of plain `python -m`). Recommended for the *first* run on a new GPU
class regardless: stick with the default single-GPU (`NUM_GPUS=1`) so
there's one fewer moving part (no DDP sync) while confirming this class of
hardware is actually faster than Kaggle's T4x2 at all -- no speedup has
been measured yet, only estimated from hardware specs. Two GPUs generally
costs close to 2x for a real-world DDP speedup closer to 1.6-1.9x on a
model this small (gradient-sync overhead is proportionally larger for tiny
models), so it's about matching wall-clock urgency, not saving money.

(Or `git clone` the repo and run `vastai/run_a10.sh` directly -- the script
clones its own working copy separately either way, so either entry point is
fine.) See [vastai/run_a10.sh](../vastai/run_a10.sh) for what it does:
clone+deps, rclone from the env vars above, a VRAM probe to pick a safe
`--device-batch-size` for *this* GPU (no torchrun/DDP needed -- single GPU,
so this differs from the Kaggle notebooks' `--nproc_per_node=2` launches),
pretrain, SFT, then a quick chat + repetition-metric check. Checkpoints
sync to Drive continuously, same as the Kaggle notebooks.

## 5. When done

- **Stop** the instance to pause GPU billing (storage still accrues small
  charges).
- **Delete** it to stop all charges once you've confirmed the results synced
  to Drive (check `kaggle/kaggle_eval.ipynb`-style verification, or just
  `rclone lsf gdrive:chatsft_checkpoints/a10` from anywhere).
- Balance hitting $0 auto-stops running instances (data isn't deleted
  immediately, but stops accruing new charges beyond storage).
