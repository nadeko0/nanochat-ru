"""
Downloads pretraining shards for the Russian corpus (FineWeb-2, config rus_Cyrl) -- the
Russian-language counterpart to nanochat/dataset.py's ClimbMix downloader.

Not folded into nanochat/dataset.py itself (that file's BASE_URL/MAX_SHARD/index_to_filename
are ClimbMix-specific, and its list_parquet_files()/parquets_iter_batched() are already
generic enough to reuse as-is once files land in the right directory -- see the
NANOCHAT_CORPUS_NAME env var it now supports).

Real, checked-not-guessed sizing (2026-08-11, see docs/RESEARCH_LOG.md): FineWeb-2's
auto-converted parquet export for rus_Cyrl is 440 shards, but each one is ~4.84GB (verified via
HTTP HEAD on the actual file) -- nothing like ClimbMix's ~40MB shards. This project's token
budget per run (~500-600M tokens) needs nowhere near that much text, so downloading just 2
shards (one train, one held out as val -- mirroring dataset.py's own "last shard = val"
convention) is already generous headroom, not a corner cut. Budget disk accordingly: 2 shards
is already ~9.7GB, before tokenizer/checkpoints/deps -- a rented box for this needs more disk
than the ~16-30GB that sufficed for the English (ClimbMix) runs.

Usage:
    NANOCHAT_CORPUS_NAME=fineweb2_ru python -m scripts.download_ru_corpus -n 2
    (the env var must also be set for tok_train.py/base_train.py later, so they read from the
    same NANOCHAT_BASE_DIR/base_data_fineweb2_ru directory this script writes to)
"""
import argparse
import os
import time

import requests

from nanochat.common import get_base_dir

REPO_ID = "HuggingFaceFW/fineweb-2"
CONFIG = "rus_Cyrl"
SPLIT = "train"
# The auto-converted parquet export (not the raw dataset files, which are organized
# differently) -- same kind of URL tasks/common.py's load_hub_dataset already relies on
# elsewhere in this project, just resolved by hand here since we want to stream large files
# to disk rather than load a whole HubDataset into memory.
PARQUET_URL = f"https://huggingface.co/api/datasets/{REPO_ID}/parquet/{CONFIG}/{SPLIT}/{{index}}.parquet"


def download_single_file(index, data_dir):
    filename = f"shard_{index:05d}.parquet"
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    url = PARQUET_URL.format(index=index)
    print(f"Downloading shard {index} -> {filename} (large file, ~4.8GB, this takes a while)...")

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):  # 8MB chunks, these files are big
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True
        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FineWeb-2 rus_Cyrl pretraining shards")
    parser.add_argument("-n", "--num-shards", type=int, default=2,
                         help="total shards to download, INCLUDING the val shard (default 2: 1 train + 1 val). "
                              "Each shard is ~4.8GB -- this is plenty for this project's token budget, don't "
                              "download more than you need.")
    parser.add_argument("-w", "--num-workers", type=int, default=1,
                         help="parallel download workers (default 1 -- these files are huge, "
                              "parallelizing risks saturating disk I/O/bandwidth for no real speedup)")
    args = parser.parse_args()

    corpus_name = os.environ.get("NANOCHAT_CORPUS_NAME")
    if corpus_name != "fineweb2_ru":
        raise SystemExit(
            "Set NANOCHAT_CORPUS_NAME=fineweb2_ru before running this (and keep it set for "
            "tok_train.py/base_train.py afterward, so they read the same directory)."
        )

    base_dir = get_base_dir()
    data_dir = os.path.join(base_dir, f"base_data_{corpus_name}")
    os.makedirs(data_dir, exist_ok=True)

    print(f"Downloading {args.num_shards} shard(s) (~{args.num_shards * 4.84:.1f}GB total) to {data_dir}")
    for i in range(args.num_shards):
        download_single_file(i, data_dir)

    print(f"Done. {args.num_shards - 1} train shard(s) + 1 val shard in {data_dir}")
