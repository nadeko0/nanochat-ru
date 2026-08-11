"""
Background watcher that mirrors NANOCHAT_BASE_DIR checkpoint/tokenizer
subfolders to a Google Drive rclone remote while training runs.

nanochat's save_checkpoint() (nanochat/checkpoint_manager.py) is called
in-process by base_train.py / chat_sft.py and writes straight to local disk
via torch.save. We deliberately don't patch that vendored code to call rclone
directly -- instead this script polls the local checkpoint dirs and uploads
any file that is new or changed since the last poll. Kaggle sessions can die
at any point, so the poll interval bounds how much progress is at risk, same
as --save-every bounds it locally.

Usage (started in the background before/alongside training):
    python kaggle/sync_checkpoints.py --remote gdrive: --interval 120 &
    SYNC_PID=$!
    ... run training ...
    kill $SYNC_PID
    python kaggle/sync_checkpoints.py --remote gdrive: --once  # final sync
"""
import argparse
import os
import subprocess
import sys
import time

TRACKED_SUBDIRS = [
    "tokenizer",
    "base_checkpoints",
    "chatsft_checkpoints",
    "chatrl_checkpoints",
]


def sync_once(base_dir, remote, log, skip_subdirs=()):
    any_failed = False
    for subdir in TRACKED_SUBDIRS:
        if subdir in skip_subdirs:
            continue
        local_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(local_path):
            continue
        remote_path = f"{remote.rstrip('/')}/{subdir}"
        cmd = ["rclone", "copy", local_path, remote_path, "--checksum", "-v"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        if result.returncode == 0:
            log(f"[{ts}] OK   {subdir} -> {remote_path}")
        else:
            any_failed = True
            log(f"[{ts}] FAIL {subdir} -> {remote_path}")
            log(result.stderr.strip()[-2000:])
    return not any_failed


def main():
    parser = argparse.ArgumentParser(description="Sync nanochat checkpoints to Google Drive via rclone")
    parser.add_argument("--remote", type=str, default="gdrive:", help="rclone remote root (default: gdrive:)")
    parser.add_argument("--interval", type=int, default=120, help="seconds between sync polls")
    parser.add_argument("--once", action="store_true", help="run a single sync pass and exit (for a final sync after training)")
    parser.add_argument("--log-file", type=str, default=None, help="also append log lines to this file")
    parser.add_argument("--skip-subdirs", type=str, default="",
                         help="comma-separated subdirs to skip (e.g. 'tokenizer' when NANOCHAT_BASE_DIR "
                              "holds a tokenizer trained for a different vocab_size than the shared "
                              "gdrive:tokenizer remote -- syncing it there would silently overwrite it)")
    args = parser.parse_args()
    skip_subdirs = {s.strip() for s in args.skip_subdirs.split(",") if s.strip()}

    base_dir = os.environ.get("NANOCHAT_BASE_DIR")
    if not base_dir:
        print("NANOCHAT_BASE_DIR is not set", file=sys.stderr)
        sys.exit(1)

    log_fh = open(args.log_file, "a") if args.log_file else None

    def log(msg):
        print(msg, flush=True)
        if log_fh:
            log_fh.write(msg + "\n")
            log_fh.flush()

    log(f"sync_checkpoints: base_dir={base_dir} remote={args.remote} interval={args.interval}s once={args.once} skip_subdirs={skip_subdirs or '{}'}")

    if args.once:
        ok = sync_once(base_dir, args.remote, log, skip_subdirs)
        sys.exit(0 if ok else 1)

    while True:
        sync_once(base_dir, args.remote, log, skip_subdirs)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
