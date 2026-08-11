#!/bin/bash
# A10 experiment on a rented GPU box (Vast.ai or similar) -- see docs/VASTAI_SETUP.md.
# Run over SSH (inside tmux, which Vast.ai puts you in by default -- if the SSH session
# drops, `tmux attach` picks the run back up).
#
# Requires these env vars set before running (same 4 values as the Kaggle Secrets, just
# plain env vars here since there's no kaggle_secrets API on a rented box -- see
# docs/RCLONE_GDRIVE_SETUP.md for what these are):
#   GDRIVE_CLIENT_ID, GDRIVE_CLIENT_SECRET, GDRIVE_OAUTH_TOKEN, GDRIVE_FOLDER_ID
#
# Optional: NUM_GPUS (default 1). Set NUM_GPUS=2 if you rented a 2-GPU box -- uses
# torchrun instead of plain `python -m` in that case. Single GPU is the recommended
# default for a first run: no real speedup has been measured yet on this class of
# hardware, and single-GPU has fewer moving parts (no DDP sync) while we find out.
set -e

REPO_URL="https://github.com/nadeko0/nanochat-ru.git"
REPO_DIR="$HOME/repo"
MODEL_TAG="a10"
NUM_GPUS="${NUM_GPUS:-1}"

for v in GDRIVE_CLIENT_ID GDRIVE_CLIENT_SECRET GDRIVE_OAUTH_TOKEN GDRIVE_FOLDER_ID; do
    if [ -z "${!v}" ]; then
        echo "Missing required env var: $v (export it before running this script)"
        exit 1
    fi
done

# Build the launch prefix and arg separator once, used for both base_train and chat_sft below.
# torchrun needs "--" to split its own args from the wrapped script's; plain `python -m`
# does NOT use that convention and argparse chokes on a literal "--" token (hit this exact
# bug with chat_rl.py earlier, see docs/RESEARCH_LOG.md) -- so this must differ by branch,
# not just the launch command.
if [ "$NUM_GPUS" -gt 1 ]; then
    LAUNCH="torchrun --standalone --nproc_per_node=$NUM_GPUS -m"
    SEP="--"
else
    LAUNCH="python3 -m"
    SEP=""
fi
echo "NUM_GPUS=$NUM_GPUS -> launch prefix: '$LAUNCH' (sep: '$SEP')"

# -----------------------------------------------------------------------------
echo "=== Step 1: clone + deps ==="
if [ -d "$REPO_DIR/.git" ]; then
    git -C "$REPO_DIR" pull
else
    git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

command -v cargo >/dev/null 2>&1 || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
export PATH="$HOME/.cargo/bin:$PATH"

command -v rclone >/dev/null 2>&1 || (curl https://rclone.org/install.sh | sudo bash)

uv pip install --system --python "$(which python3)" --extra gpu -r pyproject.toml
uv pip install --system --python "$(which python3)" accelerate  # for vram_probe.py

# -----------------------------------------------------------------------------
echo "=== Step 2: rclone from env vars, pull tokenizer + dataset ==="
mkdir -p ~/.config/rclone
cat > ~/.config/rclone/rclone.conf <<EOF
[gdrive]
type = drive
scope = drive
client_id = ${GDRIVE_CLIENT_ID}
client_secret = ${GDRIVE_CLIENT_SECRET}
token = ${GDRIVE_OAUTH_TOKEN}
root_folder_id = ${GDRIVE_FOLDER_ID}
team_drive =
EOF
rclone lsd gdrive:

export NANOCHAT_BASE_DIR="$HOME/nanochat_cache"
mkdir -p "$NANOCHAT_BASE_DIR"
rclone copy gdrive:tokenizer "$NANOCHAT_BASE_DIR/tokenizer" --checksum -v
rclone copy gdrive:base_data_climbmix "$NANOCHAT_BASE_DIR/base_data_climbmix" --checksum -v

# Resume detection, in case this is a rerun after an interruption.
RESUME_STEP=-1
if rclone lsf "gdrive:base_checkpoints/${MODEL_TAG}" >/dev/null 2>&1 && [ -n "$(rclone lsf gdrive:base_checkpoints/${MODEL_TAG})" ]; then
    echo "Found existing ${MODEL_TAG} checkpoint on Drive, downloading..."
    rclone copy "gdrive:base_checkpoints/${MODEL_TAG}" "$NANOCHAT_BASE_DIR/base_checkpoints/${MODEL_TAG}" --checksum -v
    RESUME_STEP=$(python3 -c "
import sys; sys.path.insert(0, '.')
from nanochat.checkpoint_manager import find_last_step
try:
    print(find_last_step('$NANOCHAT_BASE_DIR/base_checkpoints/${MODEL_TAG}'))
except FileNotFoundError:
    print(-1)
")
    echo "Will resume from step $RESUME_STEP"
else
    echo "No prior ${MODEL_TAG} checkpoint, starting fresh."
fi

# -----------------------------------------------------------------------------
echo "=== Step 3: VRAM probe -- find the largest safe --device-batch-size on THIS GPU ==="
# total_batch_size=262144 (auto-computed) must be evenly divisible by
# device_batch_size * max_seq_len(2048) * NUM_GPUS -- i.e. device_batch_size must divide
# 262144 / (2048 * NUM_GPUS). That's 128 for NUM_GPUS=1, 64 for NUM_GPUS=2 (the same 64
# the Kaggle T4x2 notebooks use, since those are also world_size=2).
DIVISOR_TARGET=$((262144 / (2048 * NUM_GPUS)))
PROBE_OUT=$(python3 kaggle/vram_probe.py --depth=7 --aspect-ratio=48 --max-seq-len=2048 --starting-batch-size=$DIVISOR_TARGET 2>&1 | tee /dev/stderr)
PROBE_BATCH=$(echo "$PROBE_OUT" | grep -oP 'Largest working --device-batch-size.*: \K[0-9]+')
echo "Probe found: $PROBE_BATCH (searching divisors of $DIVISOR_TARGET, per-GPU batch since VRAM is per-GPU regardless of NUM_GPUS)"
DEVICE_BATCH_SIZE=8
for candidate in 128 64 32 16 8; do
    if [ "$candidate" -le "$DIVISOR_TARGET" ] && [ "$candidate" -le "$PROBE_BATCH" ]; then
        DEVICE_BATCH_SIZE=$candidate
        break
    fi
done
echo "Using --device-batch-size=$DEVICE_BATCH_SIZE (probe ceiling was $PROBE_BATCH, rounded down to a clean divisor of $DIVISOR_TARGET)"

# -----------------------------------------------------------------------------
echo "=== Step 4: background Drive sync + pretrain ==="
python3 kaggle/sync_checkpoints.py --remote gdrive: --interval 60 --log-file /tmp/sync.log &
SYNC_PID=$!

RESUME_ARGS=""
if [ "$RESUME_STEP" -ge 0 ]; then
    RESUME_ARGS="--resume-from-step=$RESUME_STEP"
fi

$LAUNCH scripts.base_train $SEP \
    --depth=7 --aspect-ratio=48 --window-pattern=L \
    --device-batch-size="$DEVICE_BATCH_SIZE" --target-param-data-ratio=20 \
    --save-every=300 $RESUME_ARGS --run=dummy --model-tag="$MODEL_TAG"

kill $SYNC_PID 2>/dev/null || true
python3 kaggle/sync_checkpoints.py --remote gdrive: --once --log-file /tmp/sync.log

# -----------------------------------------------------------------------------
echo "=== Step 5: SFT ==="
python3 kaggle/sync_checkpoints.py --remote gdrive: --interval 60 --log-file /tmp/sync.log &
SYNC_PID=$!

$LAUNCH scripts.chat_sft $SEP \
    --model-tag="$MODEL_TAG" --mmlu-epochs=0 --gsm8k-epochs=0 \
    --num-iterations=500 --chatcore-every=-1 --eval-every=100 --run=dummy

kill $SYNC_PID 2>/dev/null || true
python3 kaggle/sync_checkpoints.py --remote gdrive: --once --log-file /tmp/sync.log

# -----------------------------------------------------------------------------
echo "=== Step 6: quick chat test + repetition metric ==="
python3 -m scripts.chat_cli -i sft -g "$MODEL_TAG" -p "hi"
python3 -m scripts.chat_cli -i sft -g "$MODEL_TAG" -p "What is your name?"
python3 -m scripts.eval_repetition -i sft -g "$MODEL_TAG" --repetition-penalty 1.2 --no-repeat-ngram-size 3

echo "=== A10 run complete (NUM_GPUS=$NUM_GPUS). ==="
