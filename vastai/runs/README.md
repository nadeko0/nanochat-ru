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
| [2026-08-11_a10_pretrain_console.log](2026-08-11_a10_pretrain_console.log) | `vastai/run_a10.sh` on a rented RTX 5070 Ti: `a10` pretrain (1905 steps, resumed once from step 900 after a disk-full crash at step 1000/1905) | Covers pretrain only, ends right after the final checkpoint save. Min val_bpb **0.977060** (beats `d4` 1.0994 and `d6` 0.9945), 28.74 min total, **~9.6x faster** than `d6`'s Kaggle T4x2 pretrain. Base-model CORE metric: 0.0747 (see RESEARCH_LOG.md for the full breakdown) |
| [2026-08-11_a10_sft_eval_console.log](2026-08-11_a10_sft_eval_console.log) | `vastai_eval_a10.ipynb` cells 3-5, run via terminal-pasted output (Jupyter's own save kept hitting "database is locked", so the notebook's baked-in output couldn't be captured -- archived as a plain console log instead, same policy as the pretrain log above: don't fabricate a notebook artifact that didn't actually save) | SFT: 32/32 steps, min val_bpb **0.6285** (between `d4` 0.6616 and `d6` 0.6169). `chat_eval.py`: ARC-Easy 25.04%, ARC-Challenge 22.78%, MMLU 22.94%, GSM8K/HumanEval 0.00%, ChatCORE -0.0113 -- at/below baseline like `d4`/`d6`. BLiMP: **72.13%**, the best of all three models (`d4` 66.46%, `d6` 70.31%). `eval_repetition.py` wasn't run in this session (added to the notebook afterward) -- still pending |

See [docs/RESEARCH_LOG.md](../../docs/RESEARCH_LOG.md) for the reasoning and
[docs/PROJECT_PLAN.md](../../docs/PROJECT_PLAN.md) for the checklist these runs map onto.
