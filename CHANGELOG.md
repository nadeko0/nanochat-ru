# Changelog

Notable changes to this repo, newest first. See [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md)
for the reasoning/dead-ends behind these changes, and [kaggle/runs/](kaggle/runs/) for the
full-output notebooks behind each result.

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
