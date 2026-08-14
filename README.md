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
goal was to run the full nanochat pipeline (tokenize -> pretrain -> SFT ->
chat) end to end on free-tier/cheap-rented hardware, at a scale small enough
to fit that hardware, understand every layer of it, and repeat it for
Russian. **Both languages are done** — six models trained total (four
English, two Russian), each pretrained, SFT'd, and evaluated on a
same-scale eval suite. See [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) for
the closed-out checklist.

- [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md) — the full, honest experiment
  log, including dead ends and things that didn't work.
- [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) — the tracked completion
  checklist this project is judged against.
- [CHANGELOG.md](CHANGELOG.md) — what changed in the repo and when.
- [kaggle/runs/](kaggle/runs/) and [vastai/runs/](vastai/runs/) — every real
  run, archived with full output (or a raw console log when a notebook's own
  save failed — see [vastai/runs/README.md](vastai/runs/README.md)).
- [comparisons/](comparisons/) — honest side-by-side vs. real open small models
  (SmolLM2-135M, Qwen2.5-0.5B), run after Phase C closed.

## Status (see CHANGELOG.md for exact dates/details)

- **`d4` (36.7M params, ratio=20/Chinchilla-optimal): pretrain + SFT done.**
  880 pretrain steps / 230.7M tokens, min val_bpb 1.0994. SFT: 1 SmolTalk
  epoch, min val_bpb 0.6616. Chat quality is real but inconsistent — coherent
  on many prompts, degenerates into repetition loops on others. Checkpoint +
  full-output run notebooks: [kaggle/runs/](kaggle/runs/).
- **Repetition-loop fix, objectively measured**: added `repetition_penalty` +
  `no_repeat_ngram_size` to `Engine.generate()` (standard techniques, ported
  not invented — see CHANGELOG). [`scripts/eval_repetition.py`](scripts/eval_repetition.py)
  (distinct-1/2 + max-4gram-repeat over 10 prompts x 3 seeds, [Li et al.
  2016](https://arxiv.org/abs/1510.03055)) confirms it: **18/30 (`d4`) and
  8/30 (`d6`) generations looped without it, 0/30 for both with it.** With
  the fix on, `d4` and `d6` are nearly indistinguishable on this metric —
  loop-avoidance turned out to be a decoding fix, not something model size
  fixes on its own. Full numbers: [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md).
- **`d6` (73.53M params, ratio=20/Chinchilla-optimal): pretrain + SFT done.**
  VRAM-probed first ([kaggle/kaggle_vram_probe.ipynb](kaggle/kaggle_vram_probe.ipynb))
  rather than guessing, chosen over overtraining `d4` on more data (the
  originally-planned `d4v2`) — see
  [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md) for why. Pretrain: 1770 steps / 464.0M tokens, 255.7
  min, **min val_bpb 0.9945** (vs `d4`'s 1.0994). SFT: 1 SmolTalk epoch, 7.0
  min, **min val_bpb 0.6169** (vs `d4`'s 0.6616). Chat test (with the
  repetition fix active): no loops in either test prompt, but responses
  still ramble/contradict themselves sometimes — bpb improved measurably,
  perceived quality only modestly, logged honestly in
  [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md) rather than oversold.
- **Full eval suite run, both models, full test sets (not sampled).**
  `chat_eval.py` (ARC/MMLU/GSM8K/HumanEval, upstream's own benchmarks):
  both models at/below the random-guessing baseline on every task
  (ChatCORE ≈ 0 for both) — confirms the expectation that knowledge/reasoning
  is out of reach at this scale, and that `d6`'s extra capacity doesn't
  change that. BLiMP (grammar, 67 categories × 1000 pairs, [Warstadt et al.
  2020](https://arxiv.org/abs/1912.00582)): **`d4` 66.46%, `d6` 70.31%** —
  both well above the 50% chance level, and `d6` measurably ahead here,
  unlike on chat_eval. Full numbers: [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md).
- **RL ([`scripts/chat_rl.py`](scripts/chat_rl.py)) on `d6`, bounded run
  (480 GSM8K examples): confirms the low expectation.** 508/510 logged
  reward values were exactly 0.0 — essentially no reward signal to learn
  from at ~0% baseline GSM8K accuracy. RL sharpens existing capability
  rather than creating it; there wasn't any here to sharpen. Full numbers:
  [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md).
- **A9/A10 architecture experiments — both done.** Both `d4` and `d6` put
  85-91% of their params into embeddings (`vocab_size=32768` was tuned
  upstream for much bigger models) — A9/A10 test reallocating that budget,
  sized against `d6`, both trained on a rented Vast.ai RTX 5070 Ti instead
  of Kaggle (measured **~9.6x faster** pretrain than Kaggle T4x2).
  **A9** (`--vocab-size=16384 --depth=7`, smaller vocab instead of more
  depth, 72.35M params): **a clean sweep** — best pretrain bpb (0.956752),
  best SFT bpb (0.612585), best BLiMP (73.48%), best repetition metric of
  all four English models. **A10** (`--aspect-ratio=48 --depth=7`, more
  depth at `d6`'s width, 87.88M params): best pretrain bpb/BLiMP among the
  first three, but SFT bpb landed *between* `d4` and `d6` — a more mixed
  result. Smaller vocab reallocation (A9) beat more depth (A10) as an
  architecture lever at this scale — see
  [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md) for the full honest
  breakdown.
- **Phase 2 (Russian): done.** Vocab_size sweep (16384 vs 32768, both at A9's `depth=7`
  architecture shape) run for real on a rented RTX 5070 Ti — **`vocab=32768` won on every
  metric** (val_bpb, CORE, RuBLiMP), the *opposite* direction from A9's English finding,
  confirming the plan's prediction that Cyrillic needs more vocab capacity to compress well.
  Winner SFT'd (`saiga_ru`) and fully evaluated: SFT min val_bpb **0.4785**, `eval_repetition
  --lang ru` **0/30 loops**, full RuBLiMP (SFT) **91.10%**. Full breakdown below and in
  [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md).

Training happens manually, on either of two backends depending on the run: Kaggle's free T4x2
(`d4`, `d6`, RL, full eval — bounded by a 12h session limit and ~30 GPU-hours/week free quota)
or a rented Vast.ai GPU (`a9`, `a10` — bf16-capable, single GPU, no session limit, real money
but cheap at this model scale — measured ~9.6x faster than Kaggle T4x2 for pretrain). Both sync
checkpoints continuously to Google Drive so a killed session never loses much progress.

## What's different from upstream nanochat

| | upstream nanochat | this fork |
|---|---|---|
| Model size | `--depth=20`..`26` (GPT-2 grade, ~560M-1.9B params) | `d4` (36.70M), `d6` (73.53M), `a10` (87.88M), `a9` (72.35M), `ru_v16384` (72.35M), `ru_v32768` (122.68M) all trained; see results below |
| Hardware | 8x H100 node | Kaggle T4 x2 (free) for `d4`/`d6`; rented single Vast.ai GPU (bf16-capable) for `a9`/`a10`/Russian |
| Session length | single long-running job | resumable across many <=12h Kaggle sessions, or an open-ended rented instance |
| Checkpoint storage | local disk | synced continuously to Google Drive via rclone |
| Language | English (ClimbMix) | English (`d4`/`d6`/`a10`/`a9`), then Russian (FineWeb-2 `rus_Cyrl`, `ru_v16384`/`ru_v32768`) |
| `Engine.generate()` | temperature + top_k only | + `repetition_penalty` + `no_repeat_ngram_size` (see CHANGELOG.md) |
| `scripts/chat_rl.py` | full GSM8K train set only | + `--max-train-examples` to bound RL to a subset |
| Eval tooling | `chat_eval.py` (ARC/MMLU/GSM8K/HumanEval) only | + [`scripts/eval_blimp.py`](scripts/eval_blimp.py), [`scripts/eval_repetition.py`](scripts/eval_repetition.py) |
| VRAM sizing | none (find out via OOM) | [`kaggle/vram_probe.py`](kaggle/vram_probe.py), tests before committing hours |
| Checkpoint sync | none (single long run, no interruption risk) | [`kaggle/sync_checkpoints.py`](kaggle/sync_checkpoints.py), background poller to Drive, `--skip-subdirs` for isolated tokenizers |
| Checkpoint retention | none (`save_checkpoint()` keeps every step forever) | [`vastai/prune_checkpoints.py`](vastai/prune_checkpoints.py), keeps only the last N local steps once synced |
| GPU peak-flops table (`nanochat/common.py`) | no RTX 50-series entries | + RTX 5070 Ti/5070/4070 Ti (verified specs, see CHANGELOG.md) |

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
the next width tier (`model_dim=384`) roughly doubles the count. `d6` (still
`model_dim=384`, one more layer) followed the same table. The later `a9`/`a10`
architecture experiments moved to `depth=7` (`model_dim=512` by default) and
instead varied `--vocab-size`/`--aspect-ratio` to change the embedding/
transformer split at a similar total param budget — see
[docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md) for that sizing.

## Training workflow

Development happens locally (this git repo); training happens on Kaggle or a rented Vast.ai
GPU depending on the run, each notebook uploaded directly (File -> Upload Notebook) or run cell
by cell in the box's own Jupyter app, rather than copied cell by cell into a local editor.

**Kaggle** (`d4`, `d6`, full eval, RL — all actually run there):

- [kaggle/kaggle_train.ipynb](kaggle/kaggle_train.ipynb) — pretraining. Run for `d4` and `d6`.
- [kaggle/kaggle_sft.ipynb](kaggle/kaggle_sft.ipynb) — SFT, run after pretraining. Run for `d4` and `d6`.
- [kaggle/kaggle_vram_probe.ipynb](kaggle/kaggle_vram_probe.ipynb) — finds the largest
  `--device-batch-size` that fits before committing hours to a bigger-depth pretrain run. Run
  for depths 5-8.
- [kaggle/kaggle_eval.ipynb](kaggle/kaggle_eval.ipynb) — full `chat_eval.py` (ARC/MMLU/GSM8K/
  HumanEval) + full BLiMP grammar eval, `d4` and `d6`. Run.
- [kaggle/kaggle_rl.ipynb](kaggle/kaggle_rl.ipynb) — bounded `chat_rl.py` (RL on GSM8K). Run for `d6`.
- [kaggle/kaggle_train_a10.ipynb](kaggle/kaggle_train_a10.ipynb) — the original planned A10
  pipeline for Kaggle. **Superseded, never actually run**: A10 ended up trained on a rented
  Vast.ai GPU instead (see below) once that turned out much faster/cheaper than expected. Kept
  around as a Kaggle-only fallback if renting isn't an option.

**Vast.ai** (`a9`, `a10`, and both Russian models — rented single GPU, bf16-capable, measured
~9.6x faster pretrain than Kaggle T4x2; see [docs/VASTAI_SETUP.md](docs/VASTAI_SETUP.md) for
renting/setup):

- [vastai/run_a10.sh](vastai/run_a10.sh) — A10's full pretrain+SFT+quick-eval pipeline as a
  single shell script (used for the actual `a10` pretrain).
- [vastai/vastai_eval_a10.ipynb](vastai/vastai_eval_a10.ipynb) — SFT + full `chat_eval.py`/
  `eval_blimp.py`/`eval_repetition.py` for `a10`, run cell-by-cell in the browser Jupyter app
  already running on any Vast.ai instance.
- [vastai/vastai_train_a9.ipynb](vastai/vastai_train_a9.ipynb) — A9's full pipeline in one
  notebook: isolated tokenizer retrain (own `NANOCHAT_BASE_DIR` + Drive path, so it can't
  collide with the shared `d4`/`d6`/`a10` tokenizer) -> VRAM probe -> pretrain (with
  [vastai/prune_checkpoints.py](vastai/prune_checkpoints.py) keeping local disk bounded) ->
  SFT -> full eval suite.
- [vastai/vastai_train_ru.ipynb](vastai/vastai_train_ru.ipynb) — Phase B's (Russian) full
  pipeline: corpus download (shared once, symlinked into each vocab-size arm) -> vocab_size
  sweep (16384 vs 32768, both tokenizer + pretrain, isolated base dirs/Drive paths) -> RuBLiMP
  on both base checkpoints -> manual winner decision -> SFT + full eval on the winner only.
- [vastai/runs/](vastai/runs/) — archived output for every Vast.ai run (real notebooks where
  Jupyter's own save succeeded, plain console logs where it didn't — see that folder's README
  for why).

1. **Local**: edit code, config, or a notebook here; commit to git; push to GitHub.
2. **Kaggle path**: open a new T4 x2 notebook, upload the phase's `.ipynb` (see its own markdown
   cells for what each code cell does — clone + deps, Drive auth + resume detection, training
   with a background Drive sync watcher, final sync).
   **Vast.ai path**: rent an instance, SSH or Jupyter Terminal in, either run the shell script
   directly or open the notebook in the instance's own Jupyter app and run cell by cell.
3. **Google Drive** (5TB, via `rclone` with a personal OAuth remote — see
   [docs/RCLONE_GDRIVE_SETUP.md](docs/RCLONE_GDRIVE_SETUP.md) for why a
   service account doesn't work here: on a personal, non-Workspace Google
   account it has no storage quota of its own and can't write into a shared
   folder) holds the source of truth for checkpoints, so a session dying at any
   point (not just Kaggle's 12h limit) loses at most one `--save-every` interval
   of progress, and the next session picks up where it left off.
4. Logs/metrics pulled back from either backend are committed to git by hand
   for a record of progress over time.

### Reproducing locally (setup only — training needs a GPU)

```bash
uv sync --extra gpu   # or --extra cpu for CPU-only smoke tests
```

Building the tokenizer (`nanochat/tokenizer.py`, via the `rustbpe` package)
requires a Rust toolchain; see [rustup.rs](https://rustup.rs/). The Kaggle
notebook installs this automatically.

## Phase 2: Russian (done)

Ran after Phase 1 (English) was judged done enough. What actually happened, vs. the original
plan below (kept for reference, since most of it held up):

1. Corpus: **`HuggingFaceFW/fineweb-2` config `rus_Cyrl`** — the one candidate actually used, no
   need to fall back to `uonlp/CulturaX` or a raw Common Crawl dump. `nanochat/dataset.py` now
   reads a `NANOCHAT_CORPUS_NAME` env var (default `"climbmix"`) instead of a hardcoded path —
   the 4th deliberate deviation from vendored code, zero behavior change when unset.
2. Corpus sizing: same `--target-param-data-ratio=20` mechanism, but sized against **A9's
   architecture shape** (`depth=7`) rather than `d4`, since A9 had already become the leading
   English architecture by the time Phase 2 started — 608.2M/775.9M tokens for the two
   vocab-sweep arms (see results above), not the originally-planned 230.7M.
3. Tokenizer: retrained from scratch, **twice** (16384 and 32768, a sweep rather than reusing
   A9's vocab_size on faith) — Cyrillic needed its own vocab_size decision, not just its own
   vocab. Each in its own `NANOCHAT_BASE_DIR`/Drive path (`gdrive:tokenizer_ru_v16384`/`_v32768`),
   confirmed isolated from the shared English `gdrive:tokenizer` via direct Drive inspection.
4. Ran on a rented Vast.ai RTX 5070 Ti (same backend as A9/A10) — full pipeline, one notebook,
   see [vastai/vastai_train_ru.ipynb](vastai/vastai_train_ru.ipynb) and the archived run above.

Full results: see the "Russian results" table above and
[docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md) 2026-08-14.

## Evaluation

Upstream nanochat's own benchmarks (CORE, MMLU, GSM8K, HumanEval, ARC)
assume a GPT-2-to-GPT-3-scale model — running them at 36-73M params was
expected to (and did) mostly return the random-guessing baseline. Rather
than skip them, this project runs the full suite anyway for completeness,
and adds two eval methods actually informative at this scale:

- **`val_bpb`** (bits-per-byte on held-out data) — computed inline during
  pretrain/SFT by the vendored training scripts, vocab-size-invariant.
- **`scripts/chat_eval.py`** (vendored, unmodified) — ARC-Easy/Challenge,
  MMLU, GSM8K, HumanEval, run on the *full* test sets, not sampled.
- **`scripts/eval_blimp.py`** (added this project) — [BLiMP](https://arxiv.org/abs/1912.00582),
  67 grammar categories × 1000 minimal pairs each, scored by which sentence
  of a grammatical/ungrammatical pair the model assigns higher probability
  to. A much better fit for this model's scale than MMLU/GSM8K.
- **`scripts/eval_repetition.py`** (added this project) — distinct-1/
  distinct-2 ([Li et al. 2016](https://arxiv.org/abs/1510.03055)) +
  max-4gram-repeat over a fixed prompt set, an objective replacement for
  eyeballing chat transcripts for repetition loops.
- **`scripts/chat_rl.py`** (vendored, `--max-train-examples` flag added
  this project) — GRPO-lite RL on GSM8K.

### English results

| | `d4` (08-10) | `d6` (08-10/11) | `a10` (08-11) | `a9` (08-11) |
|---|---|---|---|---|
| lever | baseline | width (depth=6, wider) | +depth at `d6`'s width | smaller vocab, more transformer |
| depth / vocab / params | 4 / 32768 / 36.70M | 6 / 32768 / 73.53M | 7 / 32768 / 87.88M | 7 / 16384 / 72.35M |
| hardware | Kaggle T4 x2 | Kaggle T4 x2 | rented RTX 5070 Ti | rented RTX 5070 Ti |
| pretrain tokens | 230.7M (880 steps) | 464.0M (1770 steps) | 499.4M (1905 steps) | 608.2M (2320 steps) |
| pretrain wall-clock | 64.1 min | 255.7 min | 28.74 min (~9.6x vs `d6`) | 37.87 min |
| **pretrain min val_bpb** | 1.0994 | 0.9945 | 0.977060 | **0.956752** |
| base-model CORE metric | not measured | not measured | 0.0747 | **0.0949** |
| SFT | 8.3 min, 125 steps | 7.0 min, full epoch | 0.31 min, 32 steps | ~1 min, 32 steps |
| **SFT min val_bpb** | 0.6616 | 0.6169 | 0.6285 | **0.612585** |
| distinct-1 / distinct-2 (with fix) | 0.7917 / 0.9848 | 0.8039 / 0.9864 | not run | **0.8167 / 0.9865** |
| repetition loops (of 30, with fix) | 0/30 | 0/30 | not run | 0/30 |
| ARC-Easy / ARC-Challenge / MMLU | 25.34% / 22.61% / 22.90% | 24.87% / 22.44% / 22.94% | 25.04% / 22.78% / 22.94% | 24.00% / 31.00%\* / 23.50% |
| GSM8K / HumanEval | 0.08% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| ChatCORE | -0.0109 | -0.0127 | -0.0113 | +0.0093\*\* |
| chat_eval methodology | full test sets | full test sets | full test sets | **sampled, `-x 200`** |
| **BLiMP (67×1000 pairs)** | 66.46% | 70.31% | 72.13% | **73.48%** |
| RL on GSM8K (480 examples) | not run | reward ≈ 0 on 508/510 | not run | not run |

\* ARC-Challenge's 6pp-above-baseline result is at n=200 (binomial stdev ~3pp) — plausibly
noise, not a real signal. \*\* `a9`'s `chat_eval` ran sampled (`-x 200`, GSM8K/HumanEval are
generative and dominated `a10`'s full run), so its ChatCORE isn't a clean apples-to-apples
comparison to the other three's full-test-set runs — all four values sit close enough to zero
that "no real knowledge/reasoning capability" is the honest read regardless of sign.

**`a9` is a clean sweep** — best pretrain bpb, best SFT bpb, best CORE, best BLiMP, best
repetition metric, all four models compared. `a10` only won on pretrain bpb/BLiMP among the
first three and landed *worse* than `d6` on SFT bpb — a more mixed result. Tentative read
(three architecture variants, not enough for full confidence): reallocating the
embedding-dominated budget toward a **smaller vocabulary** was a more effective lever at this
scale than **more depth at fixed width**. What hasn't moved at all, across every model and
lever tried: chat_eval/GSM8K/HumanEval sit at the random-guessing floor regardless of
architecture — that gap is a scale problem, not an architecture one. Full reasoning and
qualitative chat samples: [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md).

### Russian results (Phase 2)

Same architecture shape as `a9` (`depth=7`, default aspect_ratio → `model_dim=512`), corpus
swapped for FineWeb-2 `rus_Cyrl`, SFT data swapped for `saiga_ru`, eval swapped for RuBLiMP (the
Russian structural equivalent of BLiMP) — `chat_eval`-equivalent (Russian MMLU/GSM8K) tasks
deliberately skipped, since the English results above already show that gap is a scale problem,
not a language one.

| | `ru_v16384` | `ru_v32768` (winner) |
|---|---|---|
| total params | 72.35M (same as `a9`) | 122.68M |
| pretrain tokens | 608.2M (2320 steps) | 775.9M (2960 steps) |
| pretrain wall-clock | 36.55 min | 55.71 min |
| **pretrain min val_bpb** | 0.652795 | **0.616911** |
| base-model CORE metric | 0.0531 | **0.0630** |
| RuBLiMP, base (sampled 200/category) | 92.36% | **93.19%** |
| SFT | — (lost the vocab_size sweep, not SFT'd) | 32/32 steps, 0.42 min |
| **SFT min val_bpb** | — | **0.4785** |
| distinct-1 / distinct-2 (`--lang ru`) | — | **0.8567 / 0.9717** |
| repetition loops (of 30) | — | **0/30** |
| **RuBLiMP, SFT (full 45×1000 pairs)** | — | **91.10%** |

`vocab=32768` beat `vocab=16384` on *every* metric measured — the opposite direction from A9's
English finding (`vocab=16384` won there). This isn't a contradiction, it's the predicted
result: Cyrillic/morphologically-rich text is a documented source of poor tokenizer fertility,
so the embedding-vs-transformer-capacity tradeoff that favored a smaller vocab for English tips
the other way for Russian. The lesson generalizes: **architecture/tokenizer levers found on one
language aren't assumed to transfer to another without checking** — the vocab_size sweep existed
specifically to test that assumption rather than skip straight to reusing A9's config.

SFT dropped RuBLiMP from 93.19% (base, sampled) to 91.10% (SFT, full) — a real, measured
regression, consistent with SFT trading some pure grammatical competence for chat-format
fluency. Chat quality itself sits at the same ceiling as every English model: grammatically
fluent (Russian-shaped sentences, no repetition loops) but semantically empty or garbled (code/
HTML fragments mixed into responses that don't answer the actual question) — a scale problem
that transferred across languages exactly as expected, not something the language swap fixed or
worsened on its own. Confirmed a second way: pulled the winning checkpoint locally via `rclone`
and ran 5 more prompts on CPU (mirroring `d6`'s English spot-check methodology exactly, same
prompt categories, temperature/seed) — same fluent-but-empty pattern on all 5, no loops. Full
numbers and methodology: [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md) 2026-08-14; full run
archive: [vastai/runs/2026-08-14_ru_vocab_sweep_sft_eval.ipynb](vastai/runs/2026-08-14_ru_vocab_sweep_sft_eval.ipynb).

Chance baselines: ARC/MMLU 25%, GSM8K/HumanEval 0%, ChatCORE 0, BLiMP/RuBLiMP 50%.
Full per-run numbers, methodology, and honest interpretation are in
[docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md); raw output for every run is in
[kaggle/runs/](kaggle/runs/) (`d4`/`d6`) and [vastai/runs/](vastai/runs/)
(`a9`/`a10`/Russian).

**The short version**: all six models (four English, two Russian) are fluent at the
sentence/paragraph level (BLiMP/RuBLiMP well above chance, grammatically well-formed output in
ad hoc chat testing across both languages) but have essentially no knowledge or reasoning
capability (MMLU/ARC/GSM8K at or below chance) — a real, measured split, not a guess, that held
across every architecture variant *and* both languages tried. The repetition-penalty fix (a
decoding change, not a training one) eliminated loops entirely regardless of model size or
language — see CHANGELOG.md. This is the project's central, unavoidable finding: at this
compute budget, architecture and tokenizer choices measurably move grammatical fluency and bpb,
but not knowledge or reasoning capability off the random-guessing floor.

### How does this compare to a real small open model?

The project's very first research entry (2026-08-10) mentions trying Qwen2.5-0.5B and finding
it noticeably more coherent — but that was a vague impression, never actually measured. After
Phase C closed, ran the same 28-prompt set (RU and EN) against **`HuggingFaceTB/SmolLM2-135M-Instruct`**
(134.5M params — almost exactly `ru_v32768`'s size) and **`Qwen/Qwen2.5-0.5B-Instruct`** (494M —
the actual model referenced back in the first entry), plus this project's own `a9` (EN) and
`ru_v32768` (RU) on the identical prompts, for a genuine side-by-side. Full write-up, quoted
examples, scripts, and raw output: [comparisons/](comparisons/),
[docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md) 2026-08-14 "External comparison".

The honest result, in one line: **a same-size real model (SmolLM2) beats our best English model
(`a9`) on English because it has ~18,000x more training tokens — but that same real model loses
to our from-scratch Russian model on Russian, because it barely saw any Russian at all.**
Training-data scale and mixture dominate the architecture choices this project spent most of
its effort on. `a9`'s response to "What is 17 times 3?" collapses into literal mojibake
(`12 * 36 ��� 72 [...] 600 / 5 = still 2200 ��� 2280 ���`); SmolLM2's stays readable English even
when the arithmetic is wrong. Qwen2.5-0.5B (4x bigger, RLHF-tuned) gets simple facts and
arithmetic right and shows real safety-refusal behavior none of this project's raw SFT models
have — but still falls apart on the exact same class of multi-step problem every model here
failed at. Not a disappointing coda — it's the actual answer to the question this project set
out, from day one, to measure rather than assume.

## License

MIT, inherited from upstream nanochat (Copyright (c) Andrej Karpathy) — see
[LICENSE](LICENSE).
