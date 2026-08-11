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
instead of a lossy copy-pasted terminal buffer. In practice, every run archived here so far is
a `.log` -- Jupyter's own notebook-save kept failing ("database is locked", likely tied to the
disk-full incidents also documented below) every time it was tried, so the console-log path
ended up being the only one that actually worked for `a10`/`a9`. Not treating that as solved;
if a future run's notebook save succeeds, archive the real `.ipynb` instead.

## Runs

| File | What | Result |
|---|---|---|
| [2026-08-11_a10_pretrain_console.log](2026-08-11_a10_pretrain_console.log) | `vastai/run_a10.sh` on a rented RTX 5070 Ti: `a10` pretrain (1905 steps, resumed once from step 900 after a disk-full crash at step 1000/1905) | Covers pretrain only, ends right after the final checkpoint save. Min val_bpb **0.977060** (beats `d4` 1.0994 and `d6` 0.9945), 28.74 min total, **~9.6x faster** than `d6`'s Kaggle T4x2 pretrain. Base-model CORE metric: 0.0747 (see RESEARCH_LOG.md for the full breakdown) |
| [2026-08-11_a10_sft_eval_console.log](2026-08-11_a10_sft_eval_console.log) | `vastai_eval_a10.ipynb` cells 3-5, run via terminal-pasted output (Jupyter's own save kept hitting "database is locked", so the notebook's baked-in output couldn't be captured -- archived as a plain console log instead, same policy as the pretrain log above: don't fabricate a notebook artifact that didn't actually save) | SFT: 32/32 steps, min val_bpb **0.6285** (between `d4` 0.6616 and `d6` 0.6169). `chat_eval.py`: ARC-Easy 25.04%, ARC-Challenge 22.78%, MMLU 22.94%, GSM8K/HumanEval 0.00%, ChatCORE -0.0113 -- at/below baseline like `d4`/`d6`. BLiMP: 72.13%, best of three at the time. `eval_repetition.py` wasn't run in this session -- still pending |
| [2026-08-11_a9_pretrain_sft_eval_console.log](2026-08-11_a9_pretrain_sft_eval_console.log) | `vastai_train_a9.ipynb`, full pipeline: tokenizer retrain (vocab_size=16384) -> pretrain -> SFT -> chat test -> `eval_repetition` -> `chat_eval` (`-x 200`, sampled not full) -> BLiMP. Jupyter lost the SFT cell's displayed output again ("database is locked") -- recovered the real val_bpb straight from the saved checkpoint's `meta_000032.json` instead of guessing | Pretrain: 2320/2320 steps, 608.17M tokens, 37.87 min, min val_bpb **0.956752** (base-model CORE: 0.0949) -- **best of all four models**. SFT: 32/32 steps, min val_bpb **0.612585** -- **also best of all four**. `eval_repetition`: distinct-1 0.8167, distinct-2 0.9865, 0/30 loops -- best of all. `chat_eval` (sampled, `-x 200`): ARC-Easy 24.00%, ARC-Challenge 31.00% (likely noise at n=200), MMLU 23.50%, GSM8K/HumanEval 0.00%, ChatCORE +0.0093 (only positive one, though not apples-to-apples with `d4`/`d6`/`a10`'s full runs). BLiMP: **73.48%, best of all four** |

See [docs/RESEARCH_LOG.md](../../docs/RESEARCH_LOG.md) for the reasoning and
[docs/PROJECT_PLAN.md](../../docs/PROJECT_PLAN.md) for the checklist these runs map onto.
