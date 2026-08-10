# nanochat-ru (pet project)

A small (20-50M parameter) chat LLM, trained from scratch, first in English then
adapted to Russian. **Based on [karpathy/nanochat](https://github.com/karpathy/nanochat)**
(MIT licensed) — the tokenizer, model, training loop, and evaluation code in
`nanochat/`, `scripts/`, and `tasks/` are the vendored upstream source, with a
small number of deliberate, logged modifications (see
[CHANGELOG.md](CHANGELOG.md)). The upstream README is kept at
[docs/UPSTREAM_NANOCHAT_README.md](docs/UPSTREAM_NANOCHAT_README.md) for
reference.

This is a personal learning/research project, not a production system: the
goal is to run the full nanochat pipeline (tokenize -> pretrain -> SFT ->
chat) end to end on free-tier hardware, at a scale small enough to fit that
hardware, understand every layer of it, and repeat it for Russian. See
[docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md) for the full, honest experiment
log — including dead ends and things that didn't work — and
[CHANGELOG.md](CHANGELOG.md) for what changed in the repo and when.

## Status (see CHANGELOG.md for exact dates/details)

- **`d4` (36.7M params, ratio=20/Chinchilla-optimal): pretrain + SFT done.**
  880 pretrain steps / 230.7M tokens, min val_bpb 1.0994. SFT: 1 SmolTalk
  epoch, min val_bpb 0.6616. Chat quality is real but inconsistent — coherent
  on many prompts, degenerates into repetition loops on others. Checkpoint +
  full-output run notebooks: [kaggle/runs/](kaggle/runs/).
- **Repetition-loop fix**: added `repetition_penalty` +
  `no_repeat_ngram_size` to `Engine.generate()` (standard techniques, ported
  not invented — see CHANGELOG). Verified locally: stopped the observed
  looping across every test seed tried so far.
- **`d4v2` (same architecture, ratio=100, ~1.15B tokens): in progress.**
  "Overtraining" past the compute-optimal point for better quality, the way
  real small deployed models (Qwen, Llama) do — see RESEARCH_LOG.md for why.
- **VRAM probe tool ready** (`kaggle/kaggle_vram_probe.ipynb`) to check
  whether `d5`/`d6` (71-73M params, next size tier) fit on a T4 before
  committing hours to a run.
- **Phase 2 (Russian): deferred** until the English side is judged "done
  enough" within the free-tier compute budget.

Training happens manually on Kaggle, in bursts bounded by Kaggle's 12h
session limit and ~30 GPU-hours/week free quota, checkpoints synced
continuously to Google Drive so a killed session never loses much progress.

## What's different from upstream nanochat

| | upstream nanochat | this fork |
|---|---|---|
| Model size | `--depth=20`..`26` (GPT-2 grade, ~560M-1.9B params) | `--depth=4` (~36.7M params), experimenting with `d4` at higher token ratios and possibly `d5`/`d6` |
| Hardware | 8x H100 node | Kaggle T4 x2 (free tier) |
| Session length | single long-running job | resumable across many <=12h sessions |
| Checkpoint storage | local disk | synced continuously to Google Drive via rclone |
| Language | English (ClimbMix) | English first, then a Russian corpus (phase 2) |
| `Engine.generate()` | temperature + top_k only | + `repetition_penalty` + `no_repeat_ngram_size` (see CHANGELOG.md) |

Everything else — model architecture, tokenizer (rustbpe), training loop,
optimizer, evaluation harness — is unmodified nanochat code. Every deviation
from vendored upstream code is logged in [CHANGELOG.md](CHANGELOG.md), not
made silently.

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

Development happens locally (this git repo); training happens on Kaggle, one
notebook per phase, each uploaded directly (File -> Upload Notebook) rather
than copied cell by cell:

- [kaggle/kaggle_train.ipynb](kaggle/kaggle_train.ipynb) — pretraining. Done for English (see Status).
- [kaggle/kaggle_sft.ipynb](kaggle/kaggle_sft.ipynb) — SFT, run after pretraining. Not run yet.

1. **Local**: edit code, config, or a notebook here; commit to git; push to
   GitHub.
2. **Kaggle**: open a new T4 x2 notebook, upload the phase's `.ipynb` (see
   its own markdown cells for what each code cell does — clone + deps, Drive
   auth + resume detection, training with a background Drive sync watcher,
   final sync).
3. **Google Drive** (5TB, via `rclone` with a personal OAuth remote — see
   [docs/RCLONE_GDRIVE_SETUP.md](docs/RCLONE_GDRIVE_SETUP.md) for why a
   service account doesn't work here: on a personal, non-Workspace Google
   account it has no storage quota of its own and can't write into a shared
   folder) holds the source of truth for checkpoints, so a session dying at any
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
2. Size the corpus the same way the English run did:
   `--target-param-data-ratio=20` with `--depth=4`, which `base_train.py`
   resolved to 230.7M training tokens (it targets nanochat's own
   "scaling params" subset — transformer matrices only, excluding
   embeddings — not the full 36.7M param count naively times 20). Download
   enough shards to cover that with margin, plus a held-out validation
   slice.
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

### English results (d4, 2026-08-10)

| stage | metric | value |
|---|---|---|
| pretrain | depth / params | 4 / 36.70M |
| pretrain | training tokens | 230,686,720 (880 steps, `--target-param-data-ratio=20`) |
| pretrain | hardware | Kaggle T4 x2, ~64.1 min wall-clock |
| pretrain | peak memory | 11.16GiB / 15GiB |
| pretrain | min validation bpb | 1.0994 (from 3.85 at random init), on ClimbMix val shard |
| SFT | data | 1 epoch SmolTalk (460,341 rows), 125 steps (dataset exhausted before the 500-step cap) |
| SFT | hardware | Kaggle T4 x2, 8.3 min wall-clock |
| SFT | min validation bpb | 0.6616, on held-out SmolTalk/MMLU/GSM8K mixture |

Chat quality is genuinely mixed, logged honestly in
[docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md) rather than cherry-picked: some
prompts get a coherent, on-topic reply, others degenerate into repetition
loops (fixed since, see CHANGELOG.md for the `repetition_penalty` /
`no_repeat_ngram_size` addition — not yet re-measured with a formal metric,
only spot-checked). `d4v2` (more pretraining data) is in progress to see if
it improves consistency further.

## License

MIT, inherited from upstream nanochat (Copyright (c) Andrej Karpathy) — see
[LICENSE](LICENSE).
