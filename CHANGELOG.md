# Changelog

Notable changes to this repo, newest first. See [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md)
for the reasoning/dead-ends behind these changes, [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md)
for the tracked checklist, and [kaggle/runs/](kaggle/runs/) for the full-output notebooks
behind each result.

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
