# Research log

Running record of what was tried, what worked, what didn't, and why -- including
dead ends. This is a personal learning project first; the log is kept honest on
purpose, failures included, not just a highlight reel.

## 2026-08-10 -- Infra: fork setup, sizing, Kaggle+Drive pipeline

- Forked [karpathy/nanochat](https://github.com/karpathy/nanochat) (vendored,
  unmodified) into this repo.
- Picked `--depth=4` for the target 20-50M param range: computed exact param
  counts via `GPT(config).num_scaling_params()` on a meta-device model rather
  than guessing. `d4` = 36.70M params (model_dim=256). `d3` = 35.91M (same
  width, one less layer); `d5` jumps to 71.76M (next width tier) -- see
  README.md for the full table. Small-depth param count is embedding-dominated
  (`vocab_size x model_dim`, untied wte/lm_head), not depth-dominated, which is
  why `d3`/`d4` land at nearly the same size.

### Dead end: service account + personal Google Drive

Planned to use a GCP service account scoped to one shared Drive folder (so a
leaked Kaggle Secret couldn't expose the whole Drive). Set it up fully
(service account, shared folder, Kaggle Secrets), first real `rclone copy`
during the smoke test failed:

```
googleapi: Error 403: Service Accounts do not have storage quota.
Leverage shared drives ... storageQuotaExceeded
```

Service accounts have no storage quota of their own on a personal
(non-Workspace) Google account; writing into a folder shared with them (even
as Editor) still tries to charge the write against the service account's own
(zero) quota. This only works with Shared Drives, which require Google
Workspace -- not available on a plain gmail.com account. Switched to a
personal OAuth rclone remote instead (writes as the actual account, consumes
its own 5TB quota). Trade-off: broader scope (whole Drive, not one folder) --
documented in `docs/RCLONE_GDRIVE_SETUP.md`.

### Bug: Kaggle Secrets don't preserve newlines

Tried storing the whole `[gdrive]` rclone.conf block (`client_id` +
`client_secret` + `token`, 3 lines) as one multi-line Kaggle Secret. Result:
`empty token found - please run "rclone config reconnect gdrive:"`. Diagnosed
by dumping the assembled config's key names and value *lengths* (not values,
to avoid re-leaking the secret) -- only one `client_id` key existed, at 613
chars, way too long for an actual client_id: all 3 lines had been silently
joined onto one line, no real newline in the secret's stored value. Kaggle's
Secrets box is a single-line field. Fixed by splitting into 4 separate
single-line secrets (`GDRIVE_CLIENT_ID`, `GDRIVE_CLIENT_SECRET`,
`GDRIVE_OAUTH_TOKEN`, `GDRIVE_FOLDER_ID`) that the notebook assembles itself.

### Bug: forgot to train the tokenizer before base_train

First real `SMOKE_TEST` run crashed both DDP ranks with
`FileNotFoundError: tokenizer.pkl`. `kaggle_train.ipynb`'s dataset cell
downloaded shards but never called `scripts.tok_train`. Fixed by training the
tokenizer as part of the same cell (skipped if one was already pulled down
from Drive).

### Bug: total_batch_size not a multiple of the per-step token count

`SMOKE_TEST` config used `--device-batch-size=2 --max-seq-len=512
--total-batch-size=1024` on 2 GPUs: `world_tokens_per_fwdbwd = 2*512*2 =
2048`, but 1024 isn't a multiple of 2048 -> `base_train.py`'s own assert
caught it immediately. Fixed: `--total-batch-size=2048`.

### T4-specific: no Flash Attention 3

T4 (SM75, pre-Ampere) has no FA3, falls back to PyTorch SDPA. nanochat's own
runtime warning: SDPA has no sliding-window support, so the default
alternating `SSSL` window pattern tanks GPU utilization. Fixed with
`--window-pattern=L` (full context attention) on all T4 runs.

## 2026-08-10 -- d4 pretrain + SFT (ratio=20)

First real (non-smoke) run. `--target-param-data-ratio=20`
(Chinchilla-compute-optimal), 880 steps, 230.7M tokens, 64.1 min on Kaggle
T4x2, peak memory 11.16GiB/15GiB. **Min validation bpb: 1.0994** (from 3.85 at
random init). Full log: `kaggle/runs/2026-08-10_d4_pretrain_ratio20.ipynb`.

Pulled the checkpoint down locally (CPU) as a sanity check --
`scripts.chat_cli -i base` produced grammatical, if repetitive, English:
*"...in a beautiful city filled with beauty, love, and love... Step 1: Gather
Materials..."* -- expected for a base (non-SFT) model at this scale: it
completes text, doesn't answer questions.

SFT: `chat_sft.py` was capped at `--num-iterations=500` as a safety margin
(unlike `base_train.py`, it has **no `--save-every`** -- only saves at the
very end of the run, so an interrupted session loses 100% of that session's
SFT progress, not just one save interval). Turned out the cap never mattered:
with `--mmlu-epochs=0 --gsm8k-epochs=0`, the training mixture is just SmolTalk
(460,341 rows); bestfit-packing multiple short conversations per 2048-token
row meant a full epoch finished in **125 steps**, 8.3 minutes. Min validation
bpb: **0.6616**. Full log: `kaggle/runs/2026-08-10_d4_sft_epoch1.ipynb`.

Chat test (both in-Kaggle and pulled down locally on CPU) is a mixed bag,
logged honestly:
- "hi" -> *"I'm glad you asked... my favorite hobby..."* (coherent, on-topic)
- "What is your name?" -> *"My name is Emily, and I'm Emily..."* (on-topic,
  answers the actual question, if repetitive)
- "hi" (different sampling draw, same checkpoint) -> *"I'm glad you're
  interested in understanding the intricacies of my friend's friend's
  friend's friend's..."* (degenerate repetition loop, ~256 tokens of "friend's")

Same prompt, same checkpoint, different outcome -- `temperature=0.6` sampling
is stochastic and nanochat's `Engine.generate` has no repetition penalty, so
a 36.7M/230M-token model falls into loops fairly often. Not a bug, a real
capability ceiling.

Also: testing a Russian prompt locally crashed with `UnicodeEncodeError`.
Two unrelated causes, neither a training bug: (1) the tokenizer/dataset are
100% English (ClimbMix) -- the model has never seen Cyrillic, so Russian
input is out-of-distribution garbage in, garbage out; (2) separately, the
Windows console codepage (cp1252) can't print whatever came out. Russian is
phase 2, not attempted yet.

## 2026-08-10 -- Why not just add more data or a bigger model blindly

Before committing more GPU hours, checked two things rather than guessing:

**Dataset choice**: considered swapping ClimbMix for FineWeb-Edu (the
"quality-filtered educational text" pretraining darling). Web search found
nanochat's own `dev/LOG.md` already migrated *from* FineWeb-Edu *to* ClimbMix
(a March-2026 clustering-based data-mixture dataset) specifically because it
measured better val_bpb/CORE on this exact architecture. ClimbMix is already
the better-validated choice for this pipeline -- no dataset change made.
Sources: [FineWeb-Edu paper](https://arxiv.org/abs/2406.17557),
[SmolLM2 data-centric training](https://arxiv.org/pdf/2502.02737).

**Reality check against a real small model**: user had tried Qwen2.5-0.5B and
found it noticeably more coherent. Looked up why: Qwen2.5 series was trained
on **12-18 trillion tokens** (vs our 230.7M -- a ~50,000-80,000x gap), plus
full instruction-tuning/RLHF, plus curated/deduplicated data. Real deployed
small models are trained far past the Chinchilla compute-optimal point
(20 tokens/param) because they optimize for quality-per-parameter at
inference time, not training-compute efficiency. 12T tokens on our T4x2 would
take >6 years -- not a realistic target, but "some overtraining past ratio=20"
is a real, cheap lever within our budget. Source:
[Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115).

Decision: **d4v2** run started -- same `d4` architecture (36.7M params),
`--target-param-data-ratio=100` instead of 20 (~1.15B tokens, ~5x),
estimated ~5.3h on T4x2 (linear scaling from the measured ratio=20 run).
New model tag (`d4v2`) so it can't collide with / accidentally resume into
the finished `d4` checkpoint, which stays on Drive as a before/after
comparison point.

Also considered `d5`/`d6` (71.76M/73.53M params, next width tier) at
ratio=20 instead -- similar wall-clock budget (~3.7-4.3h), and arguably more
principled per Chinchilla (scale N and D together for a fixed compute budget,
rather than overtraining a fixed small N). Estimated risk: `model_dim` jumps
256->384 (1.5x), pushing peak memory from the measured 11.16GiB up to a rough
estimated 22-26GiB -- would not fit in a T4's 15GiB at
`--device-batch-size=16`. Built `kaggle/vram_probe.py` +
`kaggle/kaggle_vram_probe.ipynb` (using
[accelerate's `find_executable_batch_size`](https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/memory.py),
not a hand-rolled retry loop, to correctly handle OOM-retry edge cases like
CUDA cache fragmentation between attempts) to check this cheaply (a couple of
minutes per depth) before ever committing hours to a `d5`/`d6` run. Results:
not yet run -- to be logged here once available.

## 2026-08-10 -- Fixing the repetition loops (a decoding problem, not (only) a training one)

The "friend's friend's friend's..." loop above is a known, well-studied failure
mode of small/undertrained LMs under naive sampling, not something specific to
nanochat. Rather than guess at a fix, looked for the standard, battle-tested
approach: HuggingFace `transformers`' `repetition_penalty` (CTRL-style,
[Keskar et al. 2019](https://arxiv.org/abs/1909.05858)) and
`no_repeat_ngram_size` (classic seq2seq/beam-search n-gram blocking). nanochat's
`Engine.generate()` had neither -- only temperature + top_k.

Ported both algorithms into `nanochat/engine.py` (see CHANGELOG.md) -- the
first deviation from "unmodified vendored code" in this project, done
deliberately and logged, not silently. Defaulted `chat_cli.py` to
`repetition_penalty=1.2, no_repeat_ngram_size=3` (values referenced as a
common real-world combination in HF's own docs).

**Result** (local CPU, `d4` SFT checkpoint, prompt "hi", 4 different seeds):
all 4 produced coherent, loop-free (if sometimes off-topic/rambling) text.
The same checkpoint had produced a 256-token "friend's" loop on at least one
seed before this change. Caveat: this is a small spot-check (4 seeds, 1
prompt), not a systematic before/after comparison -- see the open item below
about building an actual repetition-rate metric instead of eyeballing
transcripts.

## Open questions / next up

- VRAM probe results for d5-d8 (not run yet).
- d4v2 pretrain + SFT results (running).
- Build an automated repetition-loop metric (e.g. max repeated n-gram length
  over a fixed prompt set) to replace eyeballing transcripts -- would let
  `d4` vs `d4v2` vs (if pursued) `d5`/`d6` be compared objectively instead of
  anecdotally.
- Consider a tied-embeddings experiment (`wte`==`lm_head`): at `d4`,
  embeddings are ~46% of total params (untied by upstream design, see
  `nanochat/gpt.py` docstring "untied weights for token embedding and lm_head").
  Tying them would free ~8.4M params' worth of budget at this depth -- unclear
  if that's better spent as more transformer capacity or just makes the
  embedding table itself weaker. Not started.
- Whether `chat_rl.py` (RL stage) is worth trying at this scale -- expectation
  going in is low (RL tends to sharpen existing capability more than create
  it), will log the actual result either way.
- Phase 2 (Russian) intentionally deferred until the English side is judged
  "as done as it's going to get" within the free-tier budget.
