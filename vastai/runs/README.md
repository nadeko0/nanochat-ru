# Run archive (Vast.ai)

Same idea as [kaggle/runs/](../../kaggle/runs/): real evidence of what actually ran, not a
cleaned-up retelling. Two kinds of files land here:

- **Downloaded Jupyter notebooks with full output** (`vastai_eval_a10.ipynb` etc., run in the
  browser Jupyter app on the rented instance) -- same convention as `kaggle/runs/`: File ->
  Download when done, `mv "*.ipynb" "vastai/runs/YYYY-MM-DD_<tag>_<what>.ipynb"`.
- **Raw terminal transcripts** (`.log`/`.txt`) for anything run directly over SSH/the Jupyter
  Terminal via `vastai/run_a10.sh` rather than notebook cells -- these don't have the clean
  cell-by-cell structure a notebook gives, and may be partial (a pasted console buffer, not a
  full session capture). Kept as-is, not reconstructed into a fake notebook after the fact --
  see the honest-logging policy in `docs/RESEARCH_LOG.md`'s intro.

Going forward, prefer the notebook path for anything worth archiving (Jupyter is already running
on every Vast.ai instance via the Instance Portal) -- it gives a real downloadable artifact
instead of a lossy copy-pasted terminal buffer.

## Runs

| File | What | Result |
|---|---|---|
| [2026-08-11_a10_pretrain_console.log](2026-08-11_a10_pretrain_console.log) | `vastai/run_a10.sh` on a rented RTX 5070 Ti: `a10` pretrain (1905 steps, resumed once from step 900 after a disk-full crash at step 1000/1905) | Covers pretrain only, ends right after the final checkpoint save -- SFT wasn't captured (either didn't run yet in this session or its output was lost). Min val_bpb **0.977060** (beats `d4` 1.0994 and `d6` 0.9945), 28.74 min total, **~9.6x faster** than `d6`'s Kaggle T4x2 pretrain. Base-model CORE metric: 0.0747 (see RESEARCH_LOG.md for the full breakdown) |

See [docs/RESEARCH_LOG.md](../../docs/RESEARCH_LOG.md) for the reasoning and
[docs/PROJECT_PLAN.md](../../docs/PROJECT_PLAN.md) for the checklist these runs map onto.
