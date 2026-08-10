# nanochat-ru (pet project)

A small (20-50M parameter) chat LLM, trained from scratch, first in English then
adapted to Russian. **Based on [karpathy/nanochat](https://github.com/karpathy/nanochat)**
(MIT licensed) — the tokenizer, model, training loop, and evaluation code in
`nanochat/`, `scripts/`, and `tasks/` are the vendored upstream source,
largely unmodified. The upstream README is kept at
[docs/UPSTREAM_NANOCHAT_README.md](docs/UPSTREAM_NANOCHAT_README.md) for
reference.

This is a personal learning project, not a research contribution: the goal is
to run the full nanochat pipeline (tokenize -> pretrain -> SFT -> chat) end to
end on free-tier hardware, at a scale small enough to fit that hardware, and
then repeat it for Russian.

## Status

Phase 1 (English pretrain + SFT) is scaffolded but has not been run to
completion yet — training happens manually on Kaggle, in bursts bounded by
Kaggle's 12h session limit and ~30 GPU-hours/week free quota. This README
will be updated with real numbers once a full pretrain + SFT + narrow-eval
pass completes. Until then, treat any specific loss/bpb figures elsewhere in
this repo as provisional.

## What's different from upstream nanochat

| | upstream nanochat | this fork |
|---|---|---|
| Model size | `--depth=20`..`26` (GPT-2 grade, ~560M-1.9B params) | `--depth=4` (~36.7M params) |
| Hardware | 8x H100 node | Kaggle T4 x2 (free tier) |
| Session length | single long-running job | resumable across many <=12h sessions |
| Checkpoint storage | local disk | synced continuously to Google Drive via rclone |
| Language | English (ClimbMix) | English first, then a Russian corpus (phase 2) |

Everything else — model architecture, tokenizer (rustbpe), training loop,
optimizer, evaluation harness — is unmodified nanochat code.

### Why `--depth=4`

nanochat computes `model_dim = depth * 64` (rounded up to a multiple of
`head_dim=128`) and uses untied token embedding + LM head, each
`vocab_size x model_dim`. With the default `vocab_size=32768`, embeddings
dominate the parameter count at small depths, so param count doesn't scale
linearly with depth the way it does for larger models. Exact counts,
computed via `GPT(config).num_scaling_params()` on a meta-device model
(see `nanochat/gpt.py`):

| depth | model_dim | n_head | total params |
|---|---|---|---|
| 3 | 256 | 2 | 35.91M |
| **4** | **256** | **2** | **36.70M** |
| 5 | 384 | 3 | 71.76M |
| 6 | 384 | 3 | 73.53M |

`depth=4` was chosen as the largest depth that stays under 50M params before
the next width tier (`model_dim=384`) roughly doubles the count.

## Training workflow

Development happens locally (this git repo); training happens on Kaggle,
copied in by hand from [kaggle/kaggle_train.ipynb](kaggle/kaggle_train.ipynb).

1. **Local**: edit code, config, or the notebook here; commit to git; push to
   GitHub.
2. **Kaggle**: open a new T4 x2 notebook, copy each code cell from
   `kaggle/kaggle_train.ipynb` in order (see the notebook's own markdown
   cells for what each one does — clone + deps, Drive auth + resume
   detection, optional dataset caching, training with a background Drive
   sync watcher, final sync).
3. **Google Drive** (5TB, via `rclone` + a service account scoped to one
   shared folder — see [docs/RCLONE_GDRIVE_SETUP.md](docs/RCLONE_GDRIVE_SETUP.md))
   holds the source of truth for checkpoints, so a session dying at any
   point (not just the 12h limit) loses at most one `--save-every` interval
   of progress, and the next session picks up where it left off.
4. Logs/metrics pulled back from Kaggle/Drive are committed to git by hand
   for a record of progress over time.

### Reproducing locally (setup only — training needs a GPU)

```bash
uv sync --extra gpu   # or --extra cpu for CPU-only smoke tests
```

Building the tokenizer (`nanochat/tokenizer.py`, via the `rustbpe` package)
requires a Rust toolchain; see [rustup.rs](https://rustup.rs/). The Kaggle
notebook installs this automatically.

## Phase 2: Russian

Planned after phase 1 (English pretrain + SFT + `chat_cli.py` smoke test)
completes successfully:

1. Swap the corpus loaded by `nanochat/dataset.py` (currently
   `karpathy/climbmix-400b-shuffle` parquet shards) for a Russian source —
   candidates to evaluate: `HuggingFaceFW/fineweb-2` (Russian subset),
   `uonlp/CulturaX` (`ru`), or a filtered Common Crawl Russian dump.
2. Size the corpus with the Chinchilla ratio (~20 tokens/param): for the
   ~36.7M-param `d4` model, target ~734M training tokens, plus a held-out
   validation slice.
3. Retrain the tokenizer (`scripts/tok_train.py`) from scratch on the
   Russian corpus — Cyrillic text needs its own BPE vocab, not a reused
   English one.
4. Repeat the same Kaggle + Drive training workflow.

## Evaluation

Upstream nanochat's benchmarks (CORE, MMLU, GSM8K, HumanEval, ARC) assume a
GPT-2-to-GPT-3-scale model and aren't informative at 36.7M params. Instead,
this project reports:

- `val_bpb` (bits-per-byte on a held-out validation split of the training
  corpus) — vocab-size-invariant, computed by `scripts/base_eval.py`.
- Perplexity on the same held-out split.

Both computed separately for the English and Russian phases once each
finishes training. Numbers will be added here (not in a separate leaderboard
doc, given the small scope of this project) once available.

## License

MIT, inherited from upstream nanochat (Copyright (c) Andrej Karpathy) — see
[LICENSE](LICENSE).
