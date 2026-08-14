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
- [x] A8 (interim). `d4`/`d6`/`a10`/`a9` four-way comparison table done in
      README.md's Evaluation section + RESEARCH_LOG.md's dated entries.
      A9 is the clean-sweep winner (best pretrain bpb, SFT bpb, CORE,
      BLiMP, repetition metric) -- current pick for "best English
      architecture" pending A-opt-1 (tied embeddings), still optional.
- [x] A9. Smaller `--vocab-size` experiment: `--vocab-size=16384 --depth=7`
      -> `model_dim=512`, 72,351,976 params. Pretrain: 2320 steps, 608.17M
      tokens, 37.87 min on a rented RTX 5070 Ti, **min val_bpb 0.956752**
      (best of all four English models), CORE metric 0.0949. SFT: 32/32
      steps, **min val_bpb 0.612585** (also best of all four -- unlike
      A10, this advantage held through SFT, not just pretrain).
      `eval_repetition`: distinct-1 0.8167, 0/30 loops (best). `chat_eval`
      (sampled `-x 200`): baseline on ARC/MMLU/GSM8K/HumanEval like every
      other model, ChatCORE +0.0093 (only positive one, not fully
      comparable given the sampling). **BLiMP: 73.48%, best of all four.**
      A clean sweep, unlike A10's mixed result -- see RESEARCH_LOG.md.
- [x] A10 (pretrain). `--aspect-ratio=48 --depth=7` (model_dim=384, same
      width as `d6`, 7 layers instead of 6, 87.88M params) pretrained on a
      rented Vast.ai RTX 5070 Ti instead of Kaggle (`vastai/run_a10.sh`).
      1905/1905 steps, 499.4M tokens (ratio=20), **28.74 min** (measured
      **~9.6x** faster than `d6`'s Kaggle T4x2 run), peak memory 5,386MiB.
      **Min val_bpb: 0.977060** -- beats both `d4` (1.0994) and `d6`
      (0.9945). Survived a disk-full crash at step 1000/1905 (resumed
      cleanly from step 900, see RESEARCH_LOG.md). Full log:
      `vastai/runs/2026-08-11_a10_pretrain_console.log`.
- [x] A10 (SFT + eval). SFT: 32/32 steps, min val_bpb 0.6285 (between `d4`
      0.6616 and `d6` 0.6169 -- pretrain bpb advantage didn't carry over to
      SFT). `chat_eval.py`: ARC-Easy 25.04%, ARC-Challenge 22.78%, MMLU
      22.94%, GSM8K/HumanEval 0.00%, ChatCORE -0.0113 -- baseline, same as
      `d4`/`d6`. **BLiMP: 72.13%, best of all three models** (`d4` 66.46%,
      `d6` 70.31%). `eval_repetition.py` not run yet (added to the notebook
      after this session). See RESEARCH_LOG.md and
      `vastai/runs/2026-08-11_a10_sft_eval_console.log`.

### A-optional (stretch, only if there's budget left)

- [x] A-opt-1. **Skipped, deliberately.** Tied `wte`/`lm_head` experiment
      was the one stretch item on the list; decided not worth the extra
      cycle after A9/A10 already answered the two questions that mattered
      for this project (does architecture reallocation help at this scale,
      and by how much) — this would only refine an already-diminishing-returns
      corner of the design space. Not a time/cost blocker (would've been
      cheap on the same rented GPU), just not judged worth doing.

- [x] B0. Vocab_size sweep for Russian, both at A9's `depth=7` shape, run for real on a
      rented RTX 5070 Ti: `vocab=16384` (72.35M params) min val_bpb 0.652795, CORE 0.0531,
      RuBLiMP (sampled) 92.36%; `vocab=32768` (122.68M params) min val_bpb **0.616911**, CORE
      **0.0630**, RuBLiMP (sampled) **93.19%**. **`vocab=32768` wins on every metric** -- the
      *opposite* direction from A9's English result, confirming the fertility-driven
      prediction from the planning entry (Cyrillic needs more vocab capacity, doesn't transfer
      from English). See RESEARCH_LOG.md 2026-08-14.
- [x] B1. Russian pretraining corpus: **`HuggingFaceFW/fineweb-2` config `rus_Cyrl`**, 2 shards
      (1 train + 1 val, ~9.7GB) downloaded once and shared across the vocab sweep via symlink.
      See RESEARCH_LOG.md.
- [x] B2. Corpus sized automatically via `--target-param-data-ratio=20` for both vocab sweep
      arms (608.2M tokens @ 16384, 775.9M tokens @ 32768).
- [x] B3. Tokenizers retrained from scratch on the Russian corpus for both vocab sizes (92.76s
      for the 32768 one, measured), synced to isolated Drive paths
      (`gdrive:tokenizer_ru_v16384`/`_v32768`), verified there directly via Google Drive.
- [x] B4. Pretrain on Russian, winner (`vocab=32768`): 2960/2960 steps, 775,946,240 tokens,
      55.71 min, peak memory 6,294.17MiB, **min val_bpb 0.616911**, CORE 0.0630. See
      RESEARCH_LOG.md 2026-08-14.
- [x] B5. Russian SFT dataset: `IlyaGusev/saiga_scored` filtered to `language=="Russian"` +
      `opus_score>=8` -> **28,198 clean rows** (tested exhaustively locally -- see
      RESEARCH_LOG.md for 2 real bugs caught: the dataset uses role `"bot"` not `"assistant"`,
      and ~0.14% of rows are malformed). `tasks/saiga.py` + `--sft-dataset` flag on
      `scripts/chat_sft.py`.
- [x] B6. SFT on Russian (`vocab=32768` winner): 32/32 steps, 0.42 min, **min val_bpb 0.4785**.
      See RESEARCH_LOG.md 2026-08-14.
- [x] B7. Russian eval stack run in full: `eval_repetition.py --lang ru` (avg distinct-1
      0.8567, avg distinct-2 0.9717, **0/30 loops**), full `eval_rublimp.py` on the SFT
      checkpoint (all 45 categories x 1000 pairs, **91.10% overall** -- a real regression from
      the base checkpoint's 93.19%, consistent with SFT trading some grammar for chat fluency).
      chat_eval.py-equivalent tasks deliberately skipped as planned (scale floor, not a
      language question). See RESEARCH_LOG.md 2026-08-14.
- [x] B8. Local chat test in Russian: 2 prompts, grammatically Russian-shaped but semantically
      incoherent (code/HTML fragments mixed in) -- same fluent-but-empty ceiling as every
      English model, confirms it's a scale problem that transfers across languages as
      expected. See RESEARCH_LOG.md 2026-08-14.

## Phase C — Close out

- [ ] C1. Final consolidated results table (all English + Russian variants)
      in README.md.
- [ ] C2. Every real run archived with full output (or a console log where
      that failed) in `kaggle/runs/` / `vastai/runs/`.
- [ ] C3. RESEARCH_LOG.md gets a closing "conclusions" section — what worked,
      what didn't, what I'd do differently.
- [ ] C4. CHANGELOG.md up to date with the final commit.
- [ ] C5. Everything committed and pushed, nothing only-local.

---

Progress snapshot: as of 2026-08-14, **Phase A and Phase B are both complete**.
Phase A (A1-A10, A9 done, A-opt-1 explicitly skipped with a reason): **A9 is
the standout**, clean sweep across pretrain bpb (0.956752), SFT bpb
(0.612585), CORE (0.0949), BLiMP (73.48%), and repetition metric -- best of
all four English models on every axis measured. Phase B (B0-B8, Russian):
built and locally validated before ever touching a GPU, then run for real on
a rented RTX 5070 Ti with zero code bugs surfacing on the GPU itself -- the
only new finding was the vocab_size decision (**32768 beat 16384** for
Russian, the opposite of A9's English result, confirming the
Cyrillic-needs-more-vocab prediction made during planning). Winner SFT'd and
fully evaluated: min val_bpb 0.4785, RuBLiMP 91.10%, 0/30 repetition loops.
Phase C not started (deliberately deferred) -- next real step is C1-C5
(consolidated results table, run archiving, closing conclusions).
