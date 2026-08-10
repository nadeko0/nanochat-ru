# Project plan / completion checklist

Purely technical roadmap to a "done" state for this pet project. Deliberately
excludes the write-up/guide/diagrams the user wants to add later, by hand,
once they understand the whole pipeline — those aren't tracked here.

Check items off as they're actually finished (with a real result logged in
docs/RESEARCH_LOG.md / CHANGELOG.md), not when merely started. "Done" for the
whole project = everything below checked, or explicitly marked skipped with a
reason.

## Phase A — English model, finalize

- [x] A1. VRAM probe (d5-d8) — see docs/RESEARCH_LOG.md 2026-08-10 entry.
      Result: d5=batch8, d6=batch13, d7=batch6, d8=batch6, all fit in 15GB.
- [x] A2. Decided: `d6` (73.53M, ratio=20, `--device-batch-size=13`,
      ~4.3h) over `d4v2` — see docs/RESEARCH_LOG.md 2026-08-10 entry for why.
- [ ] A3. Run the chosen main pretrain to completion.
- [ ] A4. SFT on the resulting checkpoint.
- [ ] A5. Automated repetition-loop metric (not eyeballing transcripts) —
      compare old `d4` vs the new run objectively.
- [ ] A6. `chat_eval.py` — formal ARC/MMLU/GSM8K/HumanEval numbers, for
      completeness (expected to be near-baseline at this scale, run anyway).
- [ ] A7. `chat_rl.py` — RL pass, low expectation, log the actual result
      either way.
- [ ] A8. Consolidate: side-by-side comparison of all English variants tried
      (`d4`, and whichever of `d4v2`/`d5`/`d6`/`d7`/`d8` got run) in README +
      RESEARCH_LOG.md, pick a final English model.

### A-optional (stretch, only if there's budget left)

- [ ] A-opt-1. Smaller `--vocab-size` experiment: at `d4`, embeddings are
      ~46% of total params (32768 vocab). Retrain tokenizer at e.g. 8192-16384
      and compare — does reallocating that budget to transformer capacity
      help at this scale?
- [ ] A-opt-2. Different `--aspect-ratio` (model_dim = depth * aspect_ratio,
      default 64, tuned by upstream for much bigger models): try narrower/
      deeper or wider/shallower at a fixed param budget, compare.
- [ ] A-opt-3. Tied `wte`/`lm_head` experiment (see RESEARCH_LOG.md open
      items) — upstream deliberately keeps them untied; unclear if that's
      right at our scale.

## Phase B — Russian

- [ ] B1. Pick a Russian pretraining corpus (candidates already scouted:
      `HuggingFaceFW/fineweb-2` ru subset, `uonlp/CulturaX` ru, filtered CC).
- [ ] B2. Size the corpus (Chinchilla-style, against whatever architecture
      A8 lands on).
- [ ] B3. Retrain the tokenizer from scratch on the Russian corpus.
- [ ] B4. Pretrain on Russian.
- [ ] B5. Find a Russian SFT/conversational dataset (needs research — no
      candidate picked yet; e.g. Saiga/rulm-family datasets, ru instruction
      sets, OASST ru subset are directions to check, not commitments).
- [ ] B6. SFT on Russian.
- [ ] B7. Repeat A5-A7 (repetition metric, chat_eval, chat_rl) for Russian.
- [ ] B8. Local chat test in Russian (also: fix the Windows console
      UTF-8/cp1252 crash seen earlier — that was a terminal encoding issue,
      not a model bug).

## Phase C — Close out

- [ ] C1. Final consolidated results table (all English + Russian variants)
      in README.md.
- [ ] C2. Every real run archived with full output in `kaggle/runs/`.
- [ ] C3. RESEARCH_LOG.md gets a closing "conclusions" section — what worked,
      what didn't, what I'd do differently.
- [ ] C4. CHANGELOG.md up to date with the final commit.
- [ ] C5. Everything committed and pushed, nothing only-local.

---

Progress snapshot: as of 2026-08-10, Phase A is at A1 done / A2 pending a
decision; Phases B and C not started (deliberately deferred).
