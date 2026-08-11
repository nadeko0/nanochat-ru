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
- [x] A2. Decided: `d6` (73.53M, ratio=20, `--device-batch-size=8` --
      corrected from an initial 12/13, see the "Bug (again)" entry in
      RESEARCH_LOG.md -- ~4.3h) over `d4v2`.
- [x] A2.5. Repetition-loop fix: added `repetition_penalty` +
      `no_repeat_ngram_size` to `Engine.generate()` (CTRL-style penalty +
      HF-style n-gram blocking, ported not invented -- see CHANGELOG.md and
      the RESEARCH_LOG.md "Fixing the repetition loops" entry). Verified
      locally: stopped the observed looping across every seed tried so far
      (small spot-check, not yet the formal metric in A5).
- [x] A3. `d6` pretrain complete: 1770/1770 steps, 255.72 min, min val_bpb
      0.9945 (vs `d4`'s 1.0994), peak memory 7.96GiB. See
      kaggle/runs/2026-08-10_d6_pretrain_ratio20.ipynb.
- [x] A4. `d6` SFT complete: 1 SmolTalk epoch, 7.03 min, min val_bpb 0.6169
      (vs `d4`'s 0.6616). Chat test (2 prompts, with the repetition-penalty
      fix active): both coherent, no loops. See
      kaggle/runs/2026-08-11_d6_sft.ipynb.
- [x] A5. `scripts/eval_repetition.py` (distinct-1/distinct-2 + max-4gram-repeat,
      10 prompts x 3 seeds): confirmed A2.5's fix objectively -- loops
      18/30 (`d4`) and 8/30 (`d6`) without it, 0/30 for both with it. `d6`
      is somewhat more loop-resistant than `d4` even unfixed, but with the
      fix on the two are nearly indistinguishable on this metric. See
      RESEARCH_LOG.md 2026-08-11 entry.
- [x] A6. `chat_eval.py`, full unsampled run, both models. `d4`: ARC-Easy
      25.34%, ARC-Challenge 22.61%, MMLU 22.90%, GSM8K 0.08%, HumanEval
      0.00%, ChatCORE -0.0109. `d6`: ARC-Easy 24.87%, ARC-Challenge 22.44%,
      MMLU 22.94%, GSM8K 0.00%, HumanEval 0.00%, ChatCORE -0.0127. Confirms
      the expectation exactly: both models at/below the random-guessing
      baseline on every task. See
      kaggle/runs/2026-08-11_chat_eval_blimp_d4_d6.ipynb.
- [x] A6.5. BLiMP-style grammar eval ([Warstadt et al. 2020](https://arxiv.org/abs/1912.00582)):
      67 categories of minimal-pair sentences (one grammatical, one with a
      single grammatical violation -- agreement, tense, negation, etc.),
      score by which sentence the model assigns higher probability to. No
      fine-tuning needed (reuses the existing bpb-style forward pass, low
      effort) -- and unlike MMLU/GSM8K, actually tests something this model's
      scale can plausibly do well on (grammar), not just knowledge/reasoning
      it can't. Full 67-category x 1000-pair run, both models: **`d4`
      66.46%**, **`d6` 70.31%** -- both well above the 50% chance baseline
      (and `d4`/`d6`'s chat_eval scores above), and `d6` measurably ahead of
      `d4` here, unlike on the knowledge/reasoning tasks. See
      kaggle/runs/2026-08-11_chat_eval_blimp_d4_d6.ipynb.
- [x] A7. `chat_rl.py` on `d6` (bounded run: 480 GSM8K examples via a new
      `--max-train-examples` flag, 30/30 steps complete). Confirmed the low
      expectation: average reward was exactly 0.0 on 508/510 logged steps
      (2 nonzero: 0.0078, 0.125), Pass@1 0%, Pass@8 2% at the single eval
      pass. No real signal to learn from at ~0% baseline GSM8K accuracy --
      RL sharpens existing capability, there wasn't any to sharpen. See
      kaggle/runs/2026-08-11_d6_chat_rl.ipynb.
- [x] A8 (interim). `d4` vs `d6` side-by-side comparison table done in
      README.md's Evaluation section + RESEARCH_LOG.md's dated entries.
      Will be re-opened to add A9/A10 (and A-opt-1, if run) rows once those
      finish, and to state a final pick then.
- [ ] A9. Smaller `--vocab-size` experiment, sized against `d6`'s budget
      (our current best model, not `d4`): `--vocab-size=16384 --depth=7`
      (default aspect-ratio) -> `model_dim=512`, **72.35M params** (close
      to `d6`'s 73.53M), non-embedding fraction 30.4% vs `d6`'s 14.4%.
      Requires retraining the tokenizer (different vocab_size). Estimated
      ~7.9h pretrain (calibrated from `d4`/`d6`'s measured FLOPs-to-wallclock
      rate). Queued for after A10 finishes (shares Kaggle GPU-hours budget).
- [x] A10 (pretrain). `--aspect-ratio=48 --depth=7` (model_dim=384, same
      width as `d6`, 7 layers instead of 6, 87.88M params) pretrained on a
      rented Vast.ai RTX 5070 Ti instead of Kaggle (`vastai/run_a10.sh`).
      1905/1905 steps, 499.4M tokens (ratio=20), **28.74 min** (measured
      **~9.6x** faster than `d6`'s Kaggle T4x2 run), peak memory 5,386MiB.
      **Min val_bpb: 0.977060** -- beats both `d4` (1.0994) and `d6`
      (0.9945). Survived a disk-full crash at step 1000/1905 (resumed
      cleanly from step 900, see RESEARCH_LOG.md). Full log:
      `vastai/runs/2026-08-11_a10_pretrain_console.log`.
- [ ] A10 (SFT + eval). Not yet confirmed complete -- the recovered console
      log covers pretrain only. Next: run SFT + full `chat_eval.py`/
      `eval_blimp.py` via `vastai/vastai_eval_a10.ipynb` (real downloadable
      notebook, not another terminal paste) on the same instance.

### A-optional (stretch, only if there's budget left)

- [ ] A-opt-1. Tied `wte`/`lm_head` experiment (see RESEARCH_LOG.md open
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

Progress snapshot: as of 2026-08-11, Phase A: A1-A8(interim) done. A10
pretrain done on a rented Vast.ai RTX 5070 Ti (~9.6x measured speedup vs
Kaggle T4x2, new best min val_bpb 0.977060) -- SFT/eval for `a10` still
pending, next up via `vastai/vastai_eval_a10.ipynb`.
`kaggle/kaggle_train_a10.ipynb` stays as the Kaggle fallback. A9 queued
after (~7.9h estimated on Kaggle T4x2, likely well under an hour on the
same rented GPU given A10's measured speedup -- needs a tokenizer retrain,
sized against `d6`'s budget). Phases B and C not started (deliberately
deferred).
