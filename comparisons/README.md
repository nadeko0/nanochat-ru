# External model comparison (post-Phase-C, honest closing coda)

Not part of the tracked `docs/PROJECT_PLAN.md` checklist -- Phase A/B/C were already closed out
when this was run. This exists because the user's very first reference point at the start of
this whole project (see `docs/RESEARCH_LOG.md`'s 2026-08-10 entry) was trying Qwen2.5-0.5B and
finding it noticeably more coherent than anything trainable here -- so once both languages were
done, it was worth actually measuring that gap side by side with real prompts instead of
leaving it as a vague memory.

Two real open small models, downloaded and run locally on CPU (`transformers`, no fine-tuning),
against the exact same 28-prompt sets used for this project's own models:

- **`HuggingFaceTB/SmolLM2-135M-Instruct`** (134.5M params) -- picked specifically because it's
  almost exactly `ru_v32768`'s size (122.7M) and close to `a9`'s (72.35M), for an honest
  same-scale comparison, not an apples-to-oranges one.
- **`Qwen/Qwen2.5-0.5B-Instruct`** (494M params) -- the actual model the user originally
  compared against before this project started.

## Files

- `run_small_models_ru.py` -- runs both SmolLM2-135M and Qwen2.5-0.5B against 28 Russian
  prompts (chat template, `repetition_penalty=1.2`, `no_repeat_ngram_size=3`, temperature 0.7).
- `run_smollm2_english.py` -- same 28 prompts translated to English, SmolLM2-135M only (to test
  whether its Russian weakness is a real language-coverage gap or just "it's a small model").
- `results/smollm2_135m_russian.json`, `results/qwen2.5_0.5b_russian.json` -- full prompt/
  response/timing data, Russian.
- `results/smollm2_135m_english.json` -- full prompt/response/timing data, English.
- `results/a9_english_28prompts.txt` -- this project's own best English model (`a9`), the exact
  same 28 English prompts, run via `scripts/chat_cli.py` (unmodified vendored code) for a direct
  same-methodology comparison.
- `results/ru_v32768_local_spotcheck_5prompts.txt` -- 5 more Russian prompts against this
  project's Russian winner (`ru_v32768`), mirroring the English `d6` spot-check methodology
  (see `docs/RESEARCH_LOG.md` 2026-08-14).

Model checkpoints themselves are not committed (standard `huggingface_hub` cache, downloaded
fresh by re-running the scripts) -- only the scripts and the generated text, same policy as
everywhere else in this project (checkpoints don't belong in git).

Full write-up, quoted examples, and honest conclusions: `docs/RESEARCH_LOG.md`, the
"External comparison" entry dated 2026-08-14.
