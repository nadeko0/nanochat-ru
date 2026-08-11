# Running on a rented GPU (Vast.ai) instead of Kaggle

Kaggle's free T4x2 is pre-Ampere (SM75): no bf16, no Flash Attention, and
`nanochat` falls back to fp32 + PyTorch SDPA there (see the runtime warnings
in every Kaggle run log). A rented Ampere/Ada/Blackwell-class GPU (RTX
4070 Ti/5070/4090/5070 Ti, etc.) supports bf16, and is a single GPU (no
DDP/cross-GPU sync overhead like the Kaggle T4x2 runs had) -- **measured,
not just plausible**: the A10 pretrain (RTX 5070 Ti) ran ~9.6x faster than
`d6`'s Kaggle T4x2 pretrain (28.74 min vs 255.7 min for a comparable
token count) -- see docs/RESEARCH_LOG.md. Flash Attention 3 was still
unavailable even on the 5070 Ti (PyTorch SDPA fallback), so this speedup
comes from bf16 tensor cores and no cross-GPU sync, not FA3.

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
3. Give it enough disk -- **disk size can't be changed after creation**, and
   16GB was already not enough: `save_checkpoint()` keeps every
   `--save-every` interval's checkpoint forever (no pruning, ~627MB each
   for the `a10` config), which filled a 16GB box and corrupted an
   in-progress write mid-run (see RESEARCH_LOG.md 2026-08-11). **30GB+**
   minimum recommended; `run_a10.sh` doesn't prune checkpoints itself, so
   either budget disk for `save_every x checkpoint_size` across the whole
   run, or pass a larger `--save-every` to reduce how many accumulate.
4. Click Rent. A fresh (uncached) image pull can take 10-60 minutes; a
   cached one starts in seconds.

## 2. Getting a shell on the box

1. Generate a key locally if you don't have one:
   `ssh-keygen -t ed25519 -C "your_email@example.com"`
2. Upload the **public** key (`~/.ssh/id_ed25519.pub`) at
   [cloud.vast.ai/manage-keys](https://cloud.vast.ai/manage-keys/) --
   this only applies to instances created *after* the key is added.
3. Once the instance is running, click its SSH icon for the exact command,
   e.g. `ssh -p 20544 root@<host>`. You land in a `tmux` session by default
   -- if the SSH connection drops, `ssh` back in and run `tmux attach` to
   pick the running job back up rather than losing it.

**In practice, the SSH connect string can be hard to find in the Vast.ai UI** (it's easy to
land on the Instance Portal -- a separate web dashboard the instance itself runs, exposing
Jupyter/Syncthing/Tensorboard via Cloudflare quick tunnels -- instead of the actual `ssh -p
PORT root@host` line, which lives elsewhere on the Instances card). If it's not obvious, the
**Jupyter Terminal** app (visible from the Instance Portal, or via the instance's Jupyter link)
is a working alternative that needs no SSH string at all -- a real shell in the browser, same
box, same `tmux`-less persistence caveat (closing the tab doesn't kill the process, but there's
no `tmux attach` equivalent to reconnect to a specific pane). This project's two Vast.ai
notebooks (`vastai/vastai_eval_a10.ipynb`, `vastai/vastai_train_a9.ipynb`) are written to be
run cell-by-cell in the instance's own Jupyter app for exactly this reason -- no SSH needed.

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
instead of plain `python -m`). Still recommended to start with single-GPU
(`NUM_GPUS=1`) on any *new* GPU class: one fewer moving part (no DDP sync)
while confirming this class of hardware is actually faster than Kaggle's
T4x2 -- now confirmed true for the RTX 5070 Ti (~9.6x, see above), not yet
tested for other cards. Two GPUs generally costs close to 2x for a
real-world DDP speedup closer to 1.6-1.9x on a model this small
(gradient-sync overhead is proportionally larger for tiny models), so it's
about matching wall-clock urgency, not saving money.

(Or `git clone` the repo and run `vastai/run_a10.sh` directly -- the script
clones its own working copy separately either way, so either entry point is
fine.) See [vastai/run_a10.sh](../vastai/run_a10.sh) for what it does:
clone+deps, rclone from the env vars above, a VRAM probe to pick a safe
`--device-batch-size` for *this* GPU (no torchrun/DDP needed -- single GPU,
so this differs from the Kaggle notebooks' `--nproc_per_node=2` launches),
pretrain, SFT, then a quick chat + repetition-metric check. Checkpoints
sync to Drive continuously, same as the Kaggle notebooks. In practice,
`run_a10.sh`'s SFT step ended up incomplete in the one real run so far --
finished it (plus the full eval suite) afterward via
[vastai/vastai_eval_a10.ipynb](../vastai/vastai_eval_a10.ipynb) instead,
cell by cell in the Jupyter Terminal's Jupyter app. For a from-scratch
architecture variant (different `--vocab-size`/`--depth`/`--aspect-ratio`),
[vastai/vastai_train_a9.ipynb](../vastai/vastai_train_a9.ipynb) is a fuller
template: tokenizer retrain -> pretrain -> SFT -> eval in one notebook.

**Two gotchas worth knowing before running any of these, both hit for real
(see docs/RESEARCH_LOG.md 2026-08-11):**

- **Disk-full checkpoint accumulation.** `save_checkpoint()` keeps every
  `--save-every` step's checkpoint on local disk forever -- fine on Kaggle's
  ephemeral sessions, not fine on a rented box's fixed disk. Either budget
  disk generously (see step 1.3 above) or run
  [vastai/prune_checkpoints.py](../vastai/prune_checkpoints.py) in the
  background (used in `vastai_train_a9.ipynb`) to keep only the last few
  local steps once each has synced to Drive.
- **Tokenizer collisions across architecture variants.** `nanochat/tokenizer.py`
  always writes to `{NANOCHAT_BASE_DIR}/tokenizer` -- there's no
  vocab-size-specific path, and `kaggle/sync_checkpoints.py`'s background
  sync always targets `gdrive:tokenizer`. Training a *different*
  `--vocab-size` tokenizer with the same `NANOCHAT_BASE_DIR`/sync setup as
  an existing model would silently overwrite that model's tokenizer on
  Drive. If you ever need two tokenizers side by side, use a separate
  `NANOCHAT_BASE_DIR`, a separate Drive path for the new tokenizer, and
  `sync_checkpoints.py --skip-subdirs tokenizer` for the background sync --
  see `vastai_train_a9.ipynb` Cells 2-3 for the working pattern.

## 5. When done

- **Stop** the instance to pause GPU billing (storage still accrues small
  charges).
- **Delete** it to stop all charges once you've confirmed the results synced
  to Drive (`vastai/vastai_eval_a10.ipynb`/`vastai_train_a9.ipynb`-style
  verification, or just `rclone lsf gdrive:chatsft_checkpoints/<tag>` from
  anywhere).
- Balance hitting $0 auto-stops running instances (data isn't deleted
  immediately, but stops accruing new charges beyond storage).
- A full pretrain+SFT+eval cycle for a `d6`-scale model has cost well under
  $1 on a ~$0.14/hr card so far (a10 and a9 combined) -- renting is cheap at
  this project's model scale, the time savings (~9.6x measured) matter more
  than the money.
