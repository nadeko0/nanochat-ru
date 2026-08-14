# Changelog

Notable changes to this repo, newest first. See [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md)
for the reasoning/dead-ends behind these changes, [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md)
for the tracked checklist, and [kaggle/runs/](kaggle/runs/) / [vastai/runs/](vastai/runs/) for
the full-output notebooks/console logs behind each result.

## 2026-08-14 (cont'd)

- **Phase C (project close-out) complete.** Pulled the winning Russian checkpoint
  (`chatsft_checkpoints/ru_v32768`) and tokenizer locally via `rclone` (already configured on
  this machine) into `dev-ignore/` (git-ignored, checkpoints don't belong in git) and ran 5 more
  chat prompts on CPU, deliberately mirroring `d6`'s English spot-check methodology (same prompt
  categories, temperature=0.6, seed=42) for a directly comparable qualitative result — same
  fluent-but-empty pattern, no loops, confirming the GPU session's 2-prompt chat test wasn't a
  fluke. Logged in docs/RESEARCH_LOG.md.
- Added a closing "Phase C -- closing conclusions" section to docs/RESEARCH_LOG.md: what worked
  (verify-before-trust discipline, computed-not-guessed sizing, testing the vocab_size
  assumption instead of reusing it), what didn't (the universal knowledge/reasoning ceiling
  across every model and both languages, Jupyter's unexplained notebook-save failures, A-opt-1
  left unanswered), what would be done differently starting over.
- Checked off C1-C5 in docs/PROJECT_PLAN.md — the entire tracked project checklist (Phases A,
  B, C) is now complete. Updated README.md's intro/status/evaluation sections accordingly (six
  models total, both languages, consolidated results tables) so nothing reads as in-progress.
- No further code changes this entry — Phase C was documentation/verification, not new features.

## 2026-08-14

- **Phase B (Russian) complete — vocab_size sweep decided the opposite way from English.** Ran
  `vastai_train_ru.ipynb` end to end on a rented RTX 5070 Ti; whole notebook checked cell by
  cell for hidden failures, found none besides the intentional `WINNER_VOCAB` manual-decision
  guard. `vocab=16384` (72.35M params, same shape as `a9`): min val_bpb 0.652795, CORE 0.0531,
  RuBLiMP (sampled) 92.36%. `vocab=32768` (122.68M params): min val_bpb **0.616911**, CORE
  **0.0630**, RuBLiMP (sampled) **93.19%** — **wins on every metric**, opposite to A9's English
  finding, confirming the Cyrillic-needs-more-vocab prediction from the Phase B planning entry.
  Winner SFT'd (`saiga_ru`): 32/32 steps, min val_bpb **0.4785**. `eval_repetition --lang ru`:
  0/30 loops. Full `eval_rublimp.py` (SFT, 45×1000 pairs): **91.10%** — a real regression from
  the base checkpoint's 93.19%, consistent with SFT trading grammar for chat fluency. Chat
  quality sits at the same fluent-but-empty ceiling every English model showed. Verified the
  final SFT checkpoint actually landed on Google Drive directly via the Drive API (not just
  trusted the sync tool's log). Full breakdown: docs/RESEARCH_LOG.md.
- Archived the run as a real downloaded notebook (Jupyter's save succeeded this time, unlike
  A9/A10's "database is locked" failures) at
  `vastai/runs/2026-08-14_ru_vocab_sweep_sft_eval.ipynb`.
- Updated README/CHANGELOG/PROJECT_PLAN/RESEARCH_LOG with the real Phase B results (checked off
  B0-B8), added a "Russian results" table to README.md alongside the existing English one.
- Added `dev-ignore/IDEAS.md` (git-ignored, not committed) — brainstormed follow-up ideas from
  this session (a cheap "just wants to talk" overtraining experiment, real ~1B-model cost
  estimates, a personal multilingual domain-narrowed model) parked for after Phase C.

## 2026-08-11 (cont'd, 7)

- **Phase B built and locally validated against real data** (not just researched -- actual
  code, tested before ever renting a GPU for it):
  - **4th deviation from vendored code**: `nanochat/dataset.py`'s `DATA_DIR` now reads a
    `NANOCHAT_CORPUS_NAME` env var (default `"climbmix"`, zero behavior change for `d4`/`d6`/
    `a10`/`a9`). `tok_train.py`/`base_train.py`/`dataloader.py` needed no further changes.
  - Added `scripts/download_ru_corpus.py`: downloads FineWeb-2 `rus_Cyrl` shards. Found via a
    real HTTP HEAD check that these auto-export shards are ~4.84GB *each* (not ClimbMix-sized
    ~40MB) -- only 2 needed for this project's token budget, but that's still ~9.7GB, so a
    Russian run needs 50GB+ rented disk, not the ~16-30GB that sufficed before.
  - Added `tasks/saiga.py` (`IlyaGusev/saiga_scored` filtered to Russian, quality-scored) and a
    `--sft-dataset` flag on `scripts/chat_sft.py` (default `smoltalk`, unaffected). Tested
    exhaustively against real data: 28,237 candidate rows -> found the dataset uses role
    `"bot"` not `"assistant"` (normalized) and ~0.14% of rows are malformed (dropped at init)
    -> **28,198 clean rows**, every one verified to round-trip through validation.
  - Added `scripts/eval_rublimp.py` (RuBLiMP, the real Russian equivalent of BLiMP) --
    transcribing its 45 category names from the GitHub README's prose list turned out to have
    2 real errors, only caught by querying every config against the dataset's own metadata API
    (`adp_government_case` -> `adposition_government`, `nominalization_cas` ->
    `nominalization_case`). Also fixed an import-safety bug found the same way: the script had
    no `__main__` guard, so importing anything from it ran the entire eval as a side effect
    (accidentally triggered a real RuBLiMP pass against a local `d6` checkpoint on CPU while
    testing). All 45/45 corrected category names now verified against real data.
  - Added `--lang ru` to `scripts/eval_repetition.py` (was English-prompts-only; feeding
    English prompts to a Russian-only model would just measure OOD garbage).
  - Added `vastai/vastai_train_ru.ipynb`: downloads the corpus once, sweeps `vocab_size` in
    {16384, 32768} at A9's architecture shape (own base dir + Drive path per vocab size, corpus
    symlinked rather than re-downloaded), requires a manual winner decision from real
    val_bpb/RuBLiMP (not auto-picked, matching the A9-vs-A10 policy), then SFTs and fully
    evaluates the winner only. Not run yet -- needs a rented GPU.

## 2026-08-11 (cont'd, 6)

- **A9 complete — clean sweep, best English model yet.** `--vocab-size=16384 --depth=7`
  (72.35M params) on the same rented RTX 5070 Ti: pretrain 2320 steps/608.2M tokens/37.87 min,
  **min val_bpb 0.956752** (beats `d4`/`d6`/`a10`), CORE 0.0949. SFT 32/32 steps, **min val_bpb
  0.612585** (also beats all three — unlike `a10`, this advantage held through SFT).
  `eval_repetition`: distinct-1 0.8167, 0/30 loops, best of all. `chat_eval` (sampled `-x 200`
  — GSM8K/HumanEval's generative eval dominated `a10`'s full run for a task already floored at
  0%): baseline on ARC/MMLU/GSM8K/HumanEval, ChatCORE +0.0093. **BLiMP 73.48%, best of all
  four.** Tentative read: smaller-vocab reallocation (A9) beat more-depth (A10) as an
  architecture lever at this scale. Full breakdown: docs/RESEARCH_LOG.md.
- Caught and fixed 3 real bugs before/during the A9 run, each logged: `base_train.py` has no
  `--vocab-size` flag (reads it from the tokenizer already in `NANOCHAT_BASE_DIR` instead);
  `sync_checkpoints.py`'s "tokenizer" subdir always targets the shared `gdrive:tokenizer`
  remote regardless of `NANOCHAT_BASE_DIR` — would have silently overwritten the `d4`/`d6`/
  `a10` tokenizer with A9's during the background sync (added a `--skip-subdirs` flag); and
  Jupyter lost the SFT cell's displayed output a second time ("database is locked") — piped
  SFT through `tee` in both Vast.ai notebooks going forward, and recovered A9's real val_bpb
  from the saved checkpoint's `meta_000032.json` instead of leaving it blank or guessing.
- Refreshed README/CHANGELOG/PROJECT_PLAN/RESEARCH_LOG for the four-way `d4`/`d6`/`a10`/`a9`
  comparison, fixed several stale references (the "Training workflow" section still described
  `kaggle_train_a10.ipynb` as the live A10 plan after A10 had already moved to Vast.ai; the old
  eval summary paragraph only mentioned two models). Removed a stray, output-less duplicate of
  `vastai_eval_a10.ipynb` that had ended up at the repo root instead of `vastai/`.

## 2026-08-11 (cont'd, 5)

- **A10 SFT + full eval complete**: min val_bpb 0.6285 (between `d4` 0.6616 and `d6` 0.6169),
  chat_eval at/below baseline like `d4`/`d6` (ChatCORE -0.0113), **BLiMP 72.13% -- the best of
  all three English models**. Pretrain-bpb advantage didn't carry over to SFT bpb -- a real,
  non-monotonic result, logged honestly rather than smoothed over. Hit and fixed a `!python3`
  vs `sys.executable` interpreter mismatch (bare `python3` in a shell cell missed the deps
  Cell 1 installed) and a repeat of the whole-folder-`rclone copy` disk-full bug, this time on
  the checkpoint *pull* side. Jupyter's own notebook-save kept failing ("database is locked"),
  so archived the pasted console output as a plain log
  (`vastai/runs/2026-08-11_a10_sft_eval_console.log`) instead of a fabricated baked-output
  notebook.
- Added `eval_repetition.py` to `vastai_eval_a10.ipynb` (was the one d4/d6 metric missing for
  `a10`, not run yet as of this entry).
- **Added `vastai/prune_checkpoints.py`**: background watcher (same pattern as
  `sync_checkpoints.py`) that keeps only the last N local checkpoint steps once each is old
  enough to have synced to Drive -- direct fix for the disk-full pattern hit twice on A10.
- **Added `vastai/vastai_train_a9.ipynb`**: full tokenizer-retrain -> pretrain -> SFT -> eval
  pipeline for A9 in one notebook, using a separate `NANOCHAT_BASE_DIR`
  (`~/nanochat_cache_a9`) with the dataset symlinked in from the shared cache, so A9's
  `vocab_size=16384` tokenizer can never collide with the shared `vocab_size=32768` one.
- Fixed a real bug found on review before A9 ever ran: `sync_checkpoints.py`'s "tokenizer"
  subdir always maps to the shared `gdrive:tokenizer` remote, regardless of which
  `NANOCHAT_BASE_DIR` it's pointed at -- running it in the background during A9's pretrain/SFT
  would have silently overwritten the shared `d4`/`d6`/`a10` tokenizer on Drive with A9's. Added
  a `--skip-subdirs` flag to `sync_checkpoints.py` and used it in the A9 notebook.

## 2026-08-11 (cont'd, 4)

- **A10 pretrain complete on Vast.ai (real numbers)**: 1905 steps, 499.4M
  tokens (ratio=20), 28.74 min on a rented RTX 5070 Ti, min val_bpb
  **0.977060** -- beats both `d4` (1.0994) and `d6` (0.9945). Measured
  ~9.6x speedup vs `d6`'s Kaggle T4x2 pretrain, confirming the earlier
  per-step estimate on the full run. SFT not yet confirmed complete (the
  recovered console log ends right after pretrain). Archived the raw
  console log to `vastai/runs/2026-08-11_a10_pretrain_console.log` and
  added `vastai/runs/README.md` (mirrors `kaggle/runs/README.md`).
- Added `vastai/vastai_eval_a10.ipynb`: real downloadable Jupyter notebook
  (clean source template) for running SFT/`chat_eval.py`/`eval_blimp.py`
  on `a10` via the browser Jupyter app already running on any Vast.ai
  instance -- going forward, Vast.ai runs archive as real notebooks like
  `kaggle/runs/`, not lossy terminal-buffer pastes. Starts with a disk/
  cache cleanup cell (uv/pip/cargo caches, drops local `base_checkpoints`
  since it's not needed for eval and is already on Drive) given the
  disk-full crash already hit once on this project. Credentials are never
  stored in the file -- prompted interactively via `getpass` only if the
  checkpoint isn't already cached locally.

## 2026-08-11 (cont'd, 3)

- **A10 running on Vast.ai (RTX 5070 Ti)**: real measured speedup vs Kaggle
  T4x2, ~9-10x faster per training step (dt~900ms vs `d6`'s measured
  8.67s/step), well above the earlier 3-6x hardware-spec guess. Hit a disk-
  full crash at step 1000/1905 (`save_checkpoint` keeps every `--save-every`
  interval forever, no pruning -- filled the rented box's 16GB disk),
  corrupting one in-progress optimizer-checkpoint write. Recovered by
  deleting the corrupted step and old local checkpoints (already synced to
  Drive), resuming from the last good step (900) with a larger
  `--save-every` to avoid refilling disk. See RESEARCH_LOG.md for the full
  diagnosis.
- **3rd deviation from vendored code**: `nanochat/common.py`'s
  `get_peak_flops`/`get_peak_bandwidth` tables didn't recognize the RTX 5070
  Ti (or 4070 Ti/5070) -- added verified (not guessed) dense BF16 TFLOPS +
  memory bandwidth entries for the GPUs `docs/VASTAI_SETUP.md` recommends
  renting. Informational only (`bf16_mfu` display), doesn't affect training
  correctness.

## 2026-08-11 (cont'd, 2)

- Added `vastai/run_a10.sh` + `docs/VASTAI_SETUP.md`: single-command A10 pipeline
  (clone+deps, rclone from plain env vars instead of Kaggle Secrets, VRAM probe, pretrain,
  SFT, quick eval) for a rented GPU (Vast.ai) instead of Kaggle T4x2 — testing whether
  Ampere/Ada hardware (bf16 + Flash Attention, both unavailable on Kaggle's T4) gives a
  real speedup. Not measured yet, only estimated (~3-6x guess from hardware specs).
  Supports `NUM_GPUS=2` (switches to `torchrun`) for a dual-GPU box, defaulting to 1 for
  the first run to keep the untested hardware class simple to debug.

## 2026-08-11 (cont'd)

- Added `kaggle/kaggle_train_a10.ipynb`: full pretrain + SFT + quick eval for the A10
  architecture experiment (`--aspect-ratio=48 --depth=7` -> `model_dim=384`, same width as `d6`
  but 7 layers instead of 6, 87.88M params, model tag `a10`). Same `--target-param-data-ratio=20`
  as every other run for a clean comparison. Reuses the existing tokenizer/dataset — no retrain
  needed (default `vocab_size=32768`).
- Sized A9/A10 against `d6`'s ~73.53M param budget instead of `d4`'s (`d6` is the current best
  model, and just as embedding-dominated as `d4` — 85.6% vs 91.4% of params). Recomputed A9 as
  `--vocab-size=16384 --depth=7` (72.35M params). Estimated pretrain time for both via
  `model.estimate_flops()` calibrated against `d4`/`d6`'s actual measured wall-clock (not
  guessed): A9 ~7.9h, A10 ~5.2h. Combined exceeds the 12h Kaggle session cap, so running as two
  separate notebooks/sessions (A10 first) instead of forcing both into one unattended run.
- Refreshed all docs (README, CHANGELOG, RESEARCH_LOG, PROJECT_PLAN, kaggle/runs/README) to
  current state and converted several plain-text "see X.md" mentions into real clickable
  markdown links.

## 2026-08-11

- More qualitative `d6` chat samples (5 new English prompts, local CPU): fluent,
  genre-appropriate phrasing throughout, but weak/absent factual grounding and no real
  arithmetic ("What is 2 plus 2?" produced incoherent pseudo-math) — matches the BLiMP-high/
  MMLU-low split measured formally below, now visible in actual transcripts.
- **A7 done**: added `--max-train-examples` to `scripts/chat_rl.py` (exposes
  `tasks.common.Task`'s existing `stop` kwarg via CLI) and ran a bounded RL pass on `d6`
  (480 GSM8K examples, 8 samples each, 128 max tokens) instead of the full 7473-example set,
  which at default settings would have meant many hours for a model already at ~0% GSM8K
  accuracy. Result: 508/510 logged reward values were exactly 0.0 — confirms RL had nothing
  to sharpen. Fixed a launch bug: the `--` separator (a `torchrun` convention, copied by
  mistake) breaks plain `python -m` invocations.
- **A6/A6.5 done**: full (unsampled) `chat_eval.py` and `scripts/eval_blimp.py` runs for both
  `d4` and `d6` on Kaggle GPU (`kaggle/kaggle_eval.ipynb`). `chat_eval` (ARC/MMLU/GSM8K/
  HumanEval): both models at/below the random-guessing baseline on every task, ChatCORE ≈ 0.
  BLiMP (67 grammar categories × 1000 minimal pairs, [Warstadt et al. 2020](https://arxiv.org/abs/1912.00582)):
  `d4` 66.46%, `d6` 70.31%, both well above the 50% chance level. Added `scripts/eval_blimp.py`
  (batched scoring, verified bit-identical to an unbatched reference first).
- **A5 done**: added `scripts/eval_repetition.py` (distinct-1/distinct-2 + max-4gram-repeat,
  [Li et al. 2016](https://arxiv.org/abs/1510.03055)) to replace eyeballing chat transcripts.
  Confirmed the repetition-penalty fix objectively: 18/30 (`d4`) and 8/30 (`d6`) generations
  looped without it, 0/30 for both with it.
- **`d6` SFT done**: 1 full SmolTalk epoch, 7.03 min, min validation bpb 0.6169 (vs `d4`'s
  0.6616).
- **`d6` pretrain done**: 1770/1770 steps, 464.0M tokens (`--target-param-data-ratio=20`),
  255.72 min on Kaggle T4x2, min validation bpb 0.9945 (vs `d4`'s 1.0994), peak memory 7.96GiB.
- Fixed a `d6` launch crash: the auto-computed `total_batch_size=262144` isn't evenly divisible
  by `device_batch_size=12/13` (what the VRAM probe found) at `max_seq_len=2048, world_size=2`
  — dropped to `--device-batch-size=8`, which divides cleanly, instead of overriding
  `--total-batch-size` away from nanochat's own tuned value.
- Added `docs/PROJECT_PLAN.md` (tracked completion checklist, checked off as items actually
  finish with a logged result).
- Chose `d6` (73.53M params, ratio=20/Chinchilla-optimal) over the planned `d4v2` (overtraining
  `d4` on 5x more data): more principled (scale params and data together for a fixed compute
  budget, rather than overtrain a fixed small model), and cheaper in wall-clock.
- Ran `kaggle/kaggle_vram_probe.ipynb`: `d5`=batch 8, `d6`=batch 13, `d7`=batch 6, `d8`=batch 6,
  all fit in 15GB — `d5`'s number is likely an artifact of OOM-retry memory fragmentation within
  the probe's search rather than a true ceiling (see RESEARCH_LOG.md).

## 2026-08-10

- **`nanochat/engine.py` modified** (first deviation from unmodified-vendored code): added
  `repetition_penalty` and `no_repeat_ngram_size` to `Engine.generate()` — standard,
  well-established decoding techniques (CTRL-style penalty, HF-transformers-style n-gram
  blocking), not invented here. Default in `scripts/chat_cli.py`:
  `--repetition-penalty=1.2 --no-repeat-ngram-size=3`. Verified locally against the `d4` SFT
  checkpoint: 4/4 test seeds on the "hi" prompt produced coherent (if sometimes off-topic)
  text with no repetition loops, vs. the same checkpoint looping into "friend's friend's
  friend's..." for 256 tokens on at least one seed pre-fix.
- Added `kaggle/vram_probe.py` + `kaggle/kaggle_vram_probe.ipynb`: finds the largest
  `--device-batch-size` that fits in VRAM for a given `--depth`, using
  `accelerate.utils.find_executable_batch_size` rather than a hand-rolled retry loop.
- Added `kaggle/runs/` (Kaggle notebooks downloaded with full output, kept as evidence/portfolio,
  not discarded) and `docs/RESEARCH_LOG.md` (dated log of what was tried, including failures).
- Fixed `kaggle/vram_probe.py` `ModuleNotFoundError: nanochat` — running as `python script.py`
  puts the script's own directory on `sys.path[0]`, not the repo root.
- Started `d4v2`: same `d4` architecture (36.7M params), `--target-param-data-ratio=100`
  instead of 20 (~1.15B tokens vs 230.7M) — overtraining past the Chinchilla compute-optimal
  point, the way real deployed small models (Qwen, Llama) do.
- Added `kaggle/kaggle_sft.ipynb` (SFT phase, separate notebook from pretraining).
- **Ran `d4` SFT**: 1 full SmolTalk epoch (125/500 capped steps — dataset exhausted first),
  min validation bpb 0.6616. Chat quality inconsistent pre-repetition-fix (see above).
- **Ran `d4` pretraining to completion**: 880 steps, 230.7M tokens (`--target-param-data-ratio=20`),
  64.1 min on Kaggle T4x2, min validation bpb 1.0994.
- Fixed Kaggle Secrets not preserving newlines (rclone OAuth config had to be split into 4
  single-line secrets instead of one multi-line blob).
- Switched Google Drive auth from a GCP service account to a personal OAuth rclone remote —
  service accounts have no storage quota on a personal (non-Workspace) Google account and can't
  write into a shared folder (`storageQuotaExceeded`).
- Fixed missing tokenizer training step and a `total_batch_size` assertion in the smoke test.
- Forked [karpathy/nanochat](https://github.com/karpathy/nanochat), set up the Kaggle + Google
  Drive training workflow (`docs/RCLONE_GDRIVE_SETUP.md`), picked `--depth=4` (36.70M params)
  for the target 20-50M param range.
