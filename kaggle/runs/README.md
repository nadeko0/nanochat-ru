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

See [docs/RESEARCH_LOG.md](../../docs/RESEARCH_LOG.md) for the reasoning behind
each run's configuration and what was learned from the result, including dead
ends and things that didn't work.
