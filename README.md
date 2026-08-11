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
hardware, understand every layer of it, and repeat it for Russian.

- [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md) — the full, honest experiment
  log, including dead ends and things that didn't work.
- [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) — the tracked completion
  checklist this project is judged against.
- [CHANGELOG.md](CHANGELOG.md) — what changed in the repo and when.
- [kaggle/runs/](kaggle/runs/) — every real run, archived with full output.

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
- **Phase 2 (Russian): deferred** until the English side is judged "done
  enough" within the free-tier compute budget.

Training happens manually on Kaggle, in bursts bounded by Kaggle's 12h
session limit and ~30 GPU-hours/week free quota, checkpoints synced
continuously to Google Drive so a killed session never loses much progress.

## What's different from upstream nanochat

| | upstream nanochat | this fork |
|---|---|---|
| Model size | `--depth=20`..`26` (GPT-2 grade, ~560M-1.9B params) | `--depth=4` (36.70M) and `--depth=6` (73.53M) both trained; see results below |
| Hardware | 8x H100 node | Kaggle T4 x2 (free tier) |
| Session length | single long-running job | resumable across many <=12h sessions |
| Checkpoint storage | local disk | synced continuously to Google Drive via rclone |
| Language | English (ClimbMix) | English first, then a Russian corpus (phase 2) |
| `Engine.generate()` | temperature + top_k only | + `repetition_penalty` + `no_repeat_ngram_size` (see CHANGELOG.md) |
| `scripts/chat_rl.py` | full GSM8K train set only | + `--max-train-examples` to bound RL to a subset |
| Eval tooling | `chat_eval.py` (ARC/MMLU/GSM8K/HumanEval) only | + [`scripts/eval_blimp.py`](scripts/eval_blimp.py), [`scripts/eval_repetition.py`](scripts/eval_repetition.py) |
| VRAM sizing | none (find out via OOM) | [`kaggle/vram_probe.py`](kaggle/vram_probe.py), tests before committing hours |
| Checkpoint sync | none (single long run, no interruption risk) | [`kaggle/sync_checkpoints.py`](kaggle/sync_checkpoints.py), background poller to Drive |

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

- [kaggle/kaggle_train.ipynb](kaggle/kaggle_train.ipynb) — pretraining. Run for `d4` and `d6` (see Status).
- [kaggle/kaggle_sft.ipynb](kaggle/kaggle_sft.ipynb) — SFT, run after pretraining. Run for `d4` and `d6`.
- [kaggle/kaggle_vram_probe.ipynb](kaggle/kaggle_vram_probe.ipynb) — finds the largest
  `--device-batch-size` that fits before committing hours to a bigger-depth pretrain run. Run
  for depths 5-8.
- [kaggle/kaggle_eval.ipynb](kaggle/kaggle_eval.ipynb) — full `chat_eval.py` (ARC/MMLU/GSM8K/
  HumanEval) + full BLiMP grammar eval, both models. Run.
- [kaggle/kaggle_rl.ipynb](kaggle/kaggle_rl.ipynb) — bounded `chat_rl.py` (RL on GSM8K). Run for `d6`.

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

| | `d4` (2026-08-10) | `d6` (2026-08-10/11) |
|---|---|---|
| depth / params | 4 / 36.70M | 6 / 73.53M |
| pretrain tokens | 230,686,720 (880 steps, ratio=20) | 463,994,880 (1770 steps, ratio=20) |
| pretrain hardware | Kaggle T4 x2, 64.1 min | Kaggle T4 x2, 255.7 min |
| **pretrain min val_bpb** | 1.0994 (from 3.85 at init) | **0.9945** (from 3.17 at init) |
| SFT | 1 epoch SmolTalk, 8.3 min | 1 epoch SmolTalk, 7.0 min |
| **SFT min val_bpb** | 0.6616 | **0.6169** |
| ARC-Easy / ARC-Challenge / MMLU | 25.34% / 22.61% / 22.90% | 24.87% / 22.44% / 22.94% |
| GSM8K / HumanEval | 0.08% / 0.00% | 0.00% / 0.00% |
| ChatCORE | -0.0109 | -0.0127 |
| **BLiMP (grammar, 67×1000 pairs)** | 66.46% | **70.31%** |
| repetition loops, 30 generations (no fix / with fix) | 18/30 / 0/30 | 8/30 / 0/30 |
| RL on GSM8K (480 examples) | not run | reward ≈ 0 on 508/510 steps |

Chance baselines: ARC/MMLU 25%, GSM8K/HumanEval 0%, ChatCORE 0, BLiMP 50%.
Full per-run numbers, methodology, and honest interpretation (including
where `d6` did and didn't actually beat `d4`) are in
[docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md); raw notebook output for every
run is in [kaggle/runs/](kaggle/runs/).

**The short version**: both models are fluent at the sentence level
(BLiMP well above chance, grammatically well-formed output in ad hoc chat
testing) but have essentially no knowledge or reasoning capability (MMLU/
ARC/GSM8K at or below chance) — a real, measured split, not a guess. `d6`'s
extra capacity measurably helped bpb and BLiMP, not chat_eval or RL. The
repetition-penalty fix (a decoding change, not a training one) eliminated
loops entirely regardless of model size — see CHANGELOG.md.

## License

MIT, inherited from upstream nanochat (Copyright (c) Andrej Karpathy) — see
[LICENSE](LICENSE).
