# Run archive

Kaggle notebooks downloaded after each run, **with full output baked in** (unlike
`kaggle/*.ipynb`, which are clean source templates with no output). This is the
raw evidence trail -- successes and failures both, not just the ones that worked.

To archive a run: in Kaggle, File -> Download notebook (or the "Save Version" ->
download), then here locally:

```bash
mv "ai-lab*.ipynb" "kaggle/runs/YYYY-MM-DD_<tag>_<what>.ipynb"
```

## Runs

| File | What | Result |
|---|---|---|
| [2026-08-10_d4_pretrain_ratio20.ipynb](2026-08-10_d4_pretrain_ratio20.ipynb) | `d4` pretrain, `--target-param-data-ratio=20`, 880 steps | min val_bpb 1.0994, 64.1 min |
| [2026-08-10_d4_sft_epoch1.ipynb](2026-08-10_d4_sft_epoch1.ipynb) | `d4` SFT, 1 full SmolTalk epoch (stopped at step 125/500 cap -- dataset exhausted first) | min val_bpb 0.6616, 8.3 min. Chat test: coherent on some prompts ("My name is Emily..."), degenerates into repetition loops on others ("friend's friend's...") |
| [2026-08-10_vram_probe_d5-d8.ipynb](2026-08-10_vram_probe_d5-d8.ipynb) | `kaggle_vram_probe.ipynb`: largest working `--device-batch-size` for depth 5/6/7/8 | d5=8, d6=13, d7=6, d8=6, all fit in 15GB (see RESEARCH_LOG.md for the d5-vs-d6 anomaly writeup) |
| [2026-08-10_d6_pretrain_ratio20.ipynb](2026-08-10_d6_pretrain_ratio20.ipynb) | `d6` pretrain, `--target-param-data-ratio=20`, 1770 steps | min val_bpb 0.9945, 255.7 min |
| [2026-08-11_d6_sft.ipynb](2026-08-11_d6_sft.ipynb) | `d6` SFT, 1 full SmolTalk epoch | min val_bpb 0.6169, 7.0 min. Chat test (with the repetition-penalty fix active): coherent, no loops, but still rambles/contradicts itself sometimes |
| [2026-08-11_chat_eval_blimp_d4_d6.ipynb](2026-08-11_chat_eval_blimp_d4_d6.ipynb) | `kaggle_eval.ipynb`: full `chat_eval.py` (ARC/MMLU/GSM8K/HumanEval) + full BLiMP (67 x 1000 pairs), both `d4` and `d6` | chat_eval: both models at/below random-guessing baseline on every task. BLiMP: `d4` 66.46%, `d6` 70.31% -- both well above the 50% chance level |
| [2026-08-11_d6_chat_rl.ipynb](2026-08-11_d6_chat_rl.ipynb) | `kaggle_rl.ipynb`: bounded `chat_rl.py` (GRPO-lite) on `d6`, 480 GSM8K examples, 30 steps | 508/510 logged rewards exactly 0.0 -- confirms the low expectation, nothing to sharpen at ~0% baseline GSM8K accuracy |

See [docs/RESEARCH_LOG.md](../../docs/RESEARCH_LOG.md) for the reasoning behind
each run's configuration and what was learned from the result, including dead
ends and things that didn't work. See
[docs/PROJECT_PLAN.md](../../docs/PROJECT_PLAN.md) for the checklist these
runs map onto. Runs on a rented Vast.ai GPU (A10 onward) are archived separately
in [vastai/runs/](../../vastai/runs/), same convention.
