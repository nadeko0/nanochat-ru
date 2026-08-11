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
minutes per depth) before ever committing hours to a `d5`/`d6` run.

## 2026-08-10 -- VRAM probe results (d5-d8) and why they're not perfectly clean

### How it was tested

`kaggle/vram_probe.py` builds the *real* `GPT` model and the *real*
Muon/AdamW optimizer (`model.setup_optimizer(...)`, same call `base_train.py`
makes) for a given `--depth`, `torch.compile`s it, and runs 3 forward/
backward/`optimizer.step()` iterations on random token batches at
decreasing batch sizes. It reuses the real code paths on purpose, rather
than a from-scratch memory estimate, because activation memory, optimizer
state size, and what `torch.compile` decides to keep around are all easy to
get subtly wrong by hand-calculating.

Tested on a single GPU (`CUDA_VISIBLE_DEVICES=0`), not under `torchrun
--nproc_per_node=2` like real training. This is deliberate, not a shortcut
that loses accuracy: DDP gives each GPU its own full copy of the model and
optimizer state, so per-GPU memory depends only on the local (per-device)
batch size, not on how many GPUs are in the job. One GPU's answer is the
same answer `torchrun` would give per GPU, for a fraction of the setup cost.

The actual search is `accelerate.utils.find_executable_batch_size` (not a
hand-rolled loop): start at `--starting-batch-size` (32 here), and on a CUDA
OOM, clear the cache, `gc.collect()`, multiply the batch size by 0.9, and
retry -- returns the first batch size that completes without OOM.

### Results

| depth | model_dim | max working `--device-batch-size` | peak memory |
|---|---|---|---|
| 5 | 384 | 8 | 11,465 MiB / 14,912 MiB |
| 6 | 384 | 13 | 11,667 MiB / 14,912 MiB |
| 7 | 512 | 6 | 10,889 MiB / 14,912 MiB |
| 8 | 512 | 6 | 11,371 MiB / 14,912 MiB |

All four fit on a T4 (15GB). `d7`/`d8` (wider, `model_dim=512`) landing on a
smaller batch (6) than `d5`/`d6` (`model_dim=384`) is the expected direction
-- more memory per sample at a wider model.

### The d5-vs-d6 anomaly, and why it's a property of the search, not the model

`d5` and `d6` share the same `model_dim=384` (aspect_ratio rounding puts
both at the same width tier -- `d6` just has one more transformer layer).
More layers should need *more* memory per batch element, not less, so `d6`
finding a *larger* working batch (13) than `d5` (8) looks backwards at
first glance.

Likely explanation: `find_executable_batch_size` runs its whole search
*within one process*. Each OOM triggers `torch.cuda.empty_cache()` +
`gc.collect()`, which releases cached-but-unused memory back to the driver,
but doesn't defragment the memory that's still in use. If `d5`'s search
happened to hit an OOM at a batch size where the resulting retries landed
in a more fragmented allocator state, its final "working" answer can come
out lower than the model's *true* ceiling would allow in a fresh process --
which is consistent with `d5`'s peak memory (11,465 MiB) being lower than
`d6`'s (11,667 MiB) despite `d5` being the smaller model: if `d5` truly
maxed out around 11.5GB the way `d6` maxes out around 11.7GB, its real
ceiling batch size should have been higher than 8, not lower than `d6`'s 13.

Practical takeaway: treat these numbers as **safe, verified-working lower
bounds**, not exact ceilings -- correct to plan a training run around (won't
OOM), but `d5` in particular likely has more headroom than 8 if it were
re-probed in a clean process or verified directly in a real `torchrun`
launch. Not going to chase the exact true ceiling further -- diminishing
returns for a hobby-project GPU-hours budget.

### Decision

Chosen `d6` (73.53M params) over the originally-planned `d4v2` path: same
compute-optimal `--target-param-data-ratio=20` used for the first `d4` run
(more principled than deliberately overtraining a fixed small model, see the
entry above), confirmed to fit in VRAM with room to spare, and cheaper in
wall-clock (~4.3h estimated) than the `d4v2` alternative (~5.3h) while
carrying meaningfully more model capacity (73.53M vs 36.7M).

### Bug (again): device_batch_size not compatible with the auto-computed total_batch_size

First launch attempt of `d6` crashed immediately:
`total_batch_size (262144) must be a multiple of 49152`. Same class of bug
as the earlier `SMOKE_TEST` failure (`total_batch_size` must be a multiple
of `device_batch_size * max_seq_len * world_size`), just triggered
differently: this time `--target-param-data-ratio=20` auto-computed
`total_batch_size=262144` (nanochat's own muP-style-tuned value, not
something we set), and the VRAM probe's answer of 12/13 for `--device-batch-size`
doesn't divide into it evenly (`12*2048*2=49152`, and `262144/49152` isn't
an integer). `d4`'s original run got lucky here: its `--device-batch-size=16`
happens to divide the same auto-computed 262144 cleanly.

The general rule (worth remembering going forward, not just for `d6`):
`device_batch_size` must divide `total_batch_size / (max_seq_len *
world_size)` = `262144 / 4096` = `64`, when `total_batch_size` is left on
`auto`. Fixed by dropping to `--device-batch-size=8` (divides 64 cleanly,
still comfortably under the VRAM probe's verified-safe ceiling of 13) rather
than overriding `--total-batch-size` away from nanochat's own tuned value.

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

## 2026-08-10/11 -- d6 pretrain complete

1770/1770 steps, 464.0M tokens (`--target-param-data-ratio=20`), 255.72 min
on Kaggle T4x2 (`--device-batch-size=8`), peak memory 7,962.95MiB/15GiB --
well under the VRAM probe's 11,667MiB reading for this depth, since the
actual launch used `device_batch_size=8` (forced by the total_batch_size
divisibility bug above), not the probe's 13. **Min validation bpb: 0.9945**,
down from `d4`'s 1.0994 -- a real, if modest, improvement from doubling
model capacity at the same compute-optimal token ratio. Full log:
`kaggle/runs/2026-08-10_d6_pretrain_ratio20.ipynb`.

The Kaggle session ended on its own (idle timeout after the user fell
asleep) shortly after training finished -- but only after the run had
already completed all 1770 steps and the final Drive sync had reported
`exit code: 0`, so nothing was lost; this was a clean natural finish, not
an interrupted one.

Next: SFT on this checkpoint (A4).

## 2026-08-11 -- d6 SFT complete

1 SmolTalk epoch (460,341 rows, same mixture as `d4`'s SFT --
`--mmlu-epochs=0 --gsm8k-epochs=0`), 7.03 min on Kaggle T4x2, peak memory
7,960.36MiB. **Min validation bpb: 0.6169**, down from `d4`'s 0.6616. Full
log: `kaggle/runs/2026-08-11_d6_sft.ipynb`.

Chat test (in-Kaggle, same 2 prompts as the `d4` SFT test, this time with
the repetition-penalty/no-repeat-ngram fix from A2.5 active):

- "hi" -> *"I'm sorry for the misunderstanding, but as a student, I
  understand that sometimes we don't have access to any data or
  information. However, if you're looking for a specific instance of your
  learning style or class, I would recommend trying online courses or
  tutorials on various topics."* -- coherent, no loop, somewhat rambling/
  off-topic (never actually says hello).
- "What is your name?" -> *"My name is Jack, and I'm a developer. My name
  is Rachel Nick, and my work's been called"* -- no repetition loop, but
  internally inconsistent (claims two different names in the same answer)
  and cuts off mid-sentence (hit max_tokens).

Neither response degenerated into a literal token/n-gram loop -- consistent
with the A2.5 fix working, though still just a 2-prompt spot-check, not the
systematic A5 metric. The `d6` responses are not obviously more *coherent*
in content than `d4`'s were (both models still contradict themselves,
ramble, or fail to directly answer) -- the clearest, most measurable win
from `d6` so far is the bpb numbers (0.9945 vs 1.0994 pretrain, 0.6169 vs
0.6616 SFT), not a night-and-day difference in perceived chat quality. That
gap between "the loss numbers improved" and "the chat still isn't reliably
good" is itself worth remembering honestly, not smoothing over.

## 2026-08-11 -- A5: objective repetition-loop metric

Replaced eyeballing chat_cli transcripts with `scripts/eval_repetition.py`:
generates completions for 10 fixed varied prompts x 3 seeds (30 generations
per config), reports:
- **distinct-1 / distinct-2**: fraction of unique unigrams/bigrams among all
  generated tokens ([Li et al. 2016](https://arxiv.org/abs/1510.03055), a
  standard NLG diversity metric, not invented here). Lower = more repetitive.
- **max-4gram-repeat**: the most times any single 4-token sequence repeats
  within one generation -- a direct "did it get stuck" signal, since
  distinct-n can look acceptable on average while still missing one bad loop.
- **loop count**: generations where a 4-gram repeated >=3 times (an
  arbitrary but reasonable "this is clearly stuck" threshold).

Run locally on CPU against both SFT checkpoints, with and without the A2.5
fix (`--repetition-penalty=1.0 --no-repeat-ngram-size=0` = off,
`=1.2/=3` = on, matching the `chat_cli.py` defaults):

| model | repetition control | distinct-1 | distinct-2 | worst max-4gram-repeat | loops (of 30) |
|---|---|---|---|---|---|
| `d4` | off | 0.4380 | 0.6424 | 93 | 18 |
| `d4` | on | 0.7917 | 0.9848 | 1 | 0 |
| `d6` | off | 0.5254 | 0.7889 | 89 | 8 |
| `d6` | on | 0.8039 | 0.9864 | 1 | 0 |

Three things confirmed with real numbers instead of a handful of spot-checked
seeds:

1. **The A2.5 fix is not a marginal improvement -- it's the difference
   between 60% of generations looping (`d4` off: 18/30) and 0%.** This is a
   much larger, more decisive effect than the earlier 4-seed spot-check
   suggested.
2. **`d6` is more loop-resistant than `d4` even without the fix** (8/30 vs
   18/30, higher distinct-1/2) -- some support for the "bigger model helps
   with degenerate repetition" intuition from earlier, though it clearly
   doesn't solve it alone (8/30 is still bad).
3. **With the fix on, `d4` and `d6` are nearly indistinguishable on this
   metric** (0.7917/0.9848 vs 0.8039/0.9864, both 0/30 loops) -- for this
   specific failure mode, the decoding-time fix matters far more than the
   2x jump in model capacity did. Doesn't mean `d6` isn't better overall
   (its bpb numbers are better, and this metric says nothing about whether
   the *content* of a loop-free response is any good), just that
   loop-avoidance specifically was never really a capacity problem to begin
   with -- consistent with the original diagnosis that this was a decoding
   issue, not a training one.

## 2026-08-11 -- A6/A6.5: chat_eval + BLiMP, scripted and spot-checked

`scripts/chat_eval.py` is vendored upstream, unmodified -- ran directly.
Local CPU sample against `d6`, categorical tasks only, n=100 each
(`--max-problems 100`, no GSM8K/HumanEval yet since those are generative
and slow on CPU):

| task | accuracy | chance baseline |
|---|---|---|
| ARC-Easy | 22% | 25% |
| ARC-Challenge | 27% | 25% |
| MMLU | 32% | 25% |

All within noise of random guessing on a 100-sample size, as expected going
in -- this model has no realistic capacity for multi-choice
knowledge/reasoning tasks. Confirms rather than contradicts the plan's
stated expectation.

Wrote `scripts/eval_blimp.py` (BLiMP, see the A6.5 plan entry for the
method) with batched scoring (pads a batch of sentences, one forward pass,
masks out padding when summing log-probs) rather than one sentence at a
time, needed to make a full 67 x 1000-pair run (134,000 sentences per
model) tractable. Verified batched output is bit-identical to the
unbatched version on a small sample first. Local CPU spot-check (`d6`, 2
categories x 30 pairs, batch-size 16): adjunct_island 73.3%, passive_1
73.3% -- both well above the 50% chance level, unlike the MMLU/ARC numbers
above. Consistent with the original hypothesis: this model can plausibly
show real grammatical competence even though it can't do knowledge/reasoning.

Full unsampled runs (all 5 chat_eval tasks incl. GSM8K/HumanEval, and all
67 BLiMP categories x 1000 pairs, for both `d4` and `d6`) need GPU to
finish in reasonable time -- queued in `kaggle/kaggle_eval.ipynb`.

## 2026-08-11 -- A6/A6.5 full results (both models, full test sets)

Ran on Kaggle T4 (single GPU, no DDP needed for eval -- see the notebook's
own note on why). Full log:
`kaggle/runs/2026-08-11_chat_eval_blimp_d4_d6.ipynb`.

**chat_eval.py** (full test sets, not sampled):

| task | `d4` | `d6` | chance baseline |
|---|---|---|---|
| ARC-Easy (n=2376) | 25.34% (602/2376) | 24.87% (591/2376) | 25% |
| ARC-Challenge (n=1172) | 22.61% (265/1172) | 22.44% (263/1172) | 25% |
| MMLU (n=14042) | 22.90% (3215/14042) | 22.94% (3221/14042) | 25% |
| GSM8K (n=1319) | 0.08% (1/1319) | 0.00% (0/1319) | 0% |
| HumanEval (n=164) | 0.00% (0/164) | 0.00% (0/164) | 0% |
| **ChatCORE** | **-0.0109** | **-0.0127** | 0 |

Both models sit at or fractionally below random-guessing on every single
task, ChatCORE effectively zero for both. Not a surprise -- exactly what
was predicted going in (A6's own plan entry called this "expected to be
near-baseline") -- but now it's a measured fact across the *entire* test
sets, not an assumption. `d6` (2x the params, better bpb) is not
meaningfully better than `d4` here; more capacity in this range doesn't
buy any knowledge/reasoning capability, consistent with the earlier Qwen
comparison (that gap is about orders of magnitude of scale, not the
2x we have between `d4` and `d6`).

**eval_blimp.py** (all 67 categories x 1000 pairs = 67,000 pairs each):

- `d4`: **66.46%** overall
- `d6`: **70.31%** overall
- (50% = chance, ~96.4% = human agreement per Warstadt et al. 2020)

Both models score well above chance -- confirming the local spot-check
wasn't a fluke -- and `d6` is measurably ahead of `d4` by ~4 points here,
unlike on chat_eval. This is the clearest evidence yet for the working
hypothesis from A6.5's plan entry: at this scale, grammatical competence is
a real, learnable, measurably-improvable capability, while
knowledge/reasoning (MMLU/GSM8K-style) is not -- more params/data moves
the needle on the former and not on the latter, at least across the `d4`
to `d6` jump tested here.

## 2026-08-11 -- A7: chat_rl.py on d6 (bounded), confirms the low expectation

Full GSM8K train set (7473 examples) at default settings would have meant
hundreds of thousands of rollout generations across ~467 steps -- many
hours for a model already measured at ~0% GSM8K accuracy (A6). Added
`--max-train-examples` to `scripts/chat_rl.py` (wires up
`tasks.common.Task`'s existing `stop` kwarg, already used elsewhere for
similar slicing, just not exposed via this script's CLI) and ran a bounded
version instead: 480 examples, 8 samples/example, 128 max generated
tokens, `d6` only. Full log: `kaggle/runs/2026-08-11_d6_chat_rl.ipynb`.

Also hit a small, easy-to-repeat mistake: first launch used the `--`
separator (`python -m scripts.chat_rl -- --model-tag=...`), copied from the
`torchrun`-launched training commands elsewhere in this project. `--` is
`torchrun`'s own convention for splitting its args from the wrapped
script's; plain `python -m` doesn't use it, so argparse choked on a literal
`"--"` token. Fixed by dropping it for this single-GPU, non-torchrun script.

30/30 steps completed, checkpoint saved to `chatrl_checkpoints/d6` and
synced. Result: **508 of 510 logged "Average reward" values were exactly
0.0** (the 2 exceptions: 0.0078125 and 0.125). Pass@1 0%, Pass@8 2% at the
single eval pass (step 0; `eval_every=60` default never re-triggered within
30 total steps). This confirms the expectation stated when A7 was
planned -- RL sharpens existing capability rather than creating it, and at
~0% baseline GSM8K accuracy there was essentially nothing to sharpen. A
harmless `Exception ignored in: <generator object get_batch...>` /
`AttributeError` appeared in the log after the checkpoint save completed --
interpreter-shutdown noise from garbage-collecting the training loop's
infinite `itertools.cycle` generator, not a real failure (training and
saving had already finished successfully by that point).

## 2026-08-11 -- more qualitative d6 chat samples (local, CPU)

Broader spot-check beyond the 2 canned prompts used so far, 5 new English
prompts against `d6` SFT (repetition fix active, temperature=0.6, seed=42):

- "What is the capital of France?" -> *"The capital of Paris."* -- fluent
  word choice, but grammatically malformed (missing subject) and doesn't
  actually land the fact cleanly.
- "Can you write a short poem about the ocean?" -> genuinely
  poem-shaped output ("The sun's rays are casting an orange glow over the
  deep horizon...") -- fluent and on-genre, if generic.
- "Why do we need sleep?" -> plausible-sounding but empty/circular science
  ("the body's natural clock shifts from waking to awake... often referred
  to as 'neverting back'" -- "neverting back" isn't a real term, invented).
- "What is 2 plus 2?" -> incoherent pseudo-math ("The formula for an
  arithmetic series is: A = (1 - 2) + (2 - 3) + ... + 2"), no actual
  arithmetic -- consistent with the measured 0% GSM8K accuracy (A6).
- "Explain how a computer works, in simple terms." -> fluent but generic
  and drifts into unrelated territory (starts talking about "list
  comprehension" mid-explanation).

Consistent picture across all 7 prompts tested so far (these 5 plus the 2
from the SFT run): grammatically fluent, genre-appropriate output (poem
sounds like a poem, question gets question-shaped phrasing) but weak or
absent factual grounding and zero real reasoning -- exactly the
BLiMP-high/MMLU-low split A6.5 measured formally, now visible in actual
transcripts rather than just aggregate scores.

## 2026-08-11 -- Sizing A9/A10 against d6's budget, and a time estimate that changed the plan

Originally sized A9 (smaller vocab) and A10 (different aspect ratio) against
`d4`'s ~36.7M param budget, per the plan's original framing ("at `d4`,
embeddings are ~46% of total params"). Reconsidered: `d6` is our current
best model (better bpb, better BLiMP), and its embedding fraction is
similarly extreme (85.6% of total params are `wte`+`lm_head`+`value_embeds`,
vs `d4`'s 91.4%) -- so sizing these experiments against `d6`'s ~73.53M
budget instead is more relevant to picking a final architecture.

Recomputed via the meta-device `num_scaling_params()` method (same approach
as the original depth search):

- **A9**: `--vocab-size=16384 --depth=7` (default aspect-ratio=64) ->
  `model_dim=512`, **72.35M params** (close match to `d6`'s 73.53M),
  non-embedding (`transformer_matrices`) fraction 30.4% vs `d6`'s 14.4%.
- **A10**: `--aspect-ratio=48 --depth=7` (default vocab_size=32768) ->
  `model_dim=384` (**same width as `d6`**, since `d6` is
  `aspect_ratio=64,depth=6`), 7 layers instead of 6 -> **87.88M params**.
  Isolates "more depth at fixed width" from the width change that also
  happened between `d4` and `d6` -- a cleaner ablation than "aspect ratio"
  sounds like, because rounding `model_dim` to multiples of `head_dim=128`
  collapses many `(depth, aspect_ratio)` pairs onto identical architectures
  (e.g. `aspect_ratio=48, depth=4` is literally `d4` again -- not every
  aspect_ratio value produces a genuinely different model at a given depth).

**Time estimate, calibrated not guessed**: used `model.estimate_flops()`
(the same method `base_train.py` itself calls) plus the *measured*
FLOPs-to-wall-clock rate from the two real completed runs (`d4`:
2.177e16 FLOPs / 64.1min, `d6`: 9.086e16 FLOPs / 255.72min -- rates agree
within 4.4% of each other, giving confidence in linear extrapolation over
this range) to estimate pretrain time at `--target-param-data-ratio=20`:

- A9: ~608M target tokens, ~1.645e17 FLOPs -> **~7.9h**
- A10: ~499M target tokens, ~1.078e17 FLOPs -> **~5.2h**

Combined sequentially (as originally requested, one notebook, run
overnight unattended) that's ~13-14h with SFT/eval on top -- over both the
requested ~10h and the hard 12h-per-session Kaggle cap. Decided to run them
as two separate notebooks/nights instead of forcing both into one session
(the alternative -- dropping the token ratio to ~15 to fit one night --
was rejected in favor of keeping ratio=20 fixed across every architecture
variant, so any bpb/BLiMP difference we see is attributable to the
architecture change, not a confounded data-ratio change too).

A10 first (cheaper, reuses the existing tokenizer, and mechanically
continues the already-validated `d4`->`d6` direction rather than
introducing a new variable): `kaggle/kaggle_train_a10.ipynb`, one notebook
covering pretrain + SFT + a quick chat/repetition check, tag `a10`. A9
queued for after (separate notebook, not built yet -- needs a tokenizer
retrain step first).

**Honest expectation for A9/A10**, since it's worth stating up front rather
than only after the fact: BLiMP/bpb plausibly move (they were sensitive to
the `d4`->`d6` capacity change); MMLU/GSM8K almost certainly won't (that
gap is about orders of magnitude of scale per the Qwen comparison, not
budget allocation); qualitative chat coherence is genuinely uncertain --
A10 seems the safer bet of the two (same lever that already worked once),
A9 is more exploratory (smaller vocab means longer tokenized sequences per
unit of text, a real potential downside that could offset the "more
compute budget" upside).

### Aside: cloud GPU rental as a faster alternative to free Kaggle T4s

User asked what a paid GPU (e.g. Vast.ai, RunPod) would cost, to get
results faster than waiting hours on a free T4x2. Looked it up rather than
guessing: RTX 4090 on Vast.ai's marketplace runs roughly **$0.13-0.59/hr**,
so $5 buys somewhere in the range of 8-38 hours depending on which
listing. A single 4090 (Ampere/Ada) supports bf16 and Flash Attention 2 --
both unavailable on Kaggle's T4 (SM75, pre-Ampere), which is why nanochat
falls back to fp32 + PyTorch SDPA there. Real speedup over T4x2 for this
workload isn't measured, just plausible (bf16 alone is commonly ~2x on
tensor cores, plus materially higher raw FLOPS than a T4). Not pursued yet
-- would need adapting the `kaggle_*.ipynb` notebooks' Kaggle-specific bits
(the `kaggle_secrets` API, session-based GPU allocation) to a generic
SSH/Jupyter cloud box, and is a real-money commitment the user would need
to set up and fund directly, not something to build unprompted.
Source: [Vast.ai RTX 4090 pricing](https://vast.ai/pricing/gpu/RTX-4090).

## 2026-08-11 -- Funded Vast.ai, built the single-GPU adaptation for A10

User added $5 credit to a Vast.ai account and shared a live marketplace
listing dump to pick from. Key filter applied: this project's models are
tens of millions of params, so the 48GB/96GB/179GB listings (RTX PRO 6000,
H100, H200, B200) are wasted money -- a cheap single 12-24GB Ampere/Ada
card (RTX 4070 Ti/5070/3090, ~$0.09-0.16/hr per the listings shared) is the
right fit. $5 buys 30-50+ hours at that rate.

Looked up [docs.vast.ai](https://docs.vast.ai/documentation/get-started/quickstart)
rather than guessing the rental/SSH flow -- summarized into
`docs/VASTAI_SETUP.md`. SSH access is actually simpler than Kaggle for
this: no `kaggle_secrets` API, no notebook cells, just a shell with `tmux`
by default (survives SSH disconnects).

Wrote `vastai/run_a10.sh`: single-GPU version of `kaggle_train_a10.ipynb`'s
pipeline (clone+deps, rclone from plain env vars instead of Kaggle
Secrets, VRAM probe to pick a safe `--device-batch-size` for whichever GPU
actually gets rented, pretrain, SFT, quick eval). No `torchrun` (single
GPU, no DDP) -- caught the same `--`-separator mistake from the `chat_rl.py`
bug before it shipped this time (plain `python -m` doesn't use torchrun's
`--` convention). One real difference from the Kaggle notebooks: with
`world_size=1` instead of 2, `device_batch_size` must divide
`262144/2048=128` (not 64) for the auto-computed `total_batch_size` to
divide evenly.

**Honest time expectation, stated before running rather than after**: 3-6x
speedup over the T4x2 Kaggle numbers is a reasoned guess from the hardware
specs (bf16, no cross-GPU sync, more raw FLOPS), not a measurement. Applied
to the combined A9+A10 estimate (~13.1h), that's a 2.2-4.4h range -- the
user's hoped-for "~3 hours to close both" sits inside that range but on
the optimistic side, not a promise.

## 2026-08-11 -- A10 on Vast.ai: real speedup measured, a disk-full crash, and a 3rd vendored-code deviation

Rented a single RTX 5070 Ti on Vast.ai ($0.138/hr) and ran `vastai/run_a10.sh`
manually over the browser-based Jupyter Terminal (couldn't get a literal
`ssh -p PORT root@host` string out of the Vast.ai UI in the time available --
the Instance Portal's own web dashboard/tunnels kept surfacing instead, so
used that terminal in place of raw SSH; functionally equivalent, just typed
into a browser tab instead of a local terminal).

**Real speedup, measured this time, not estimated**: pretrain ran at
~285-295k tok/sec, dt~900ms/step, vs `d6`'s measured 8.67s/step on Kaggle
T4x2 (255.72min / 1770 steps) -- **~9-10x faster per step**, well above the
3-6x guess from hardware specs. `bf16` + PyTorch SDPA (Flash Attention 3
still unavailable even on this GPU -- `WARNING: Flash Attention 3 not
available, using PyTorch SDPA fallback` -- so this speedup comes from bf16
tensor cores and no cross-GPU DDP sync, not from FA3).

### Bug: disk full at step 1000/1905, corrupted an in-progress checkpoint write

`save_checkpoint()` in `nanochat/checkpoint_manager.py` has no retention
policy -- every `--save-every=100` interval keeps `model_STEP.pt` +
`meta_STEP.json` + `optim_STEP_rank0.pt` on disk forever (~627MB/checkpoint:
215MB model + 383MB optimizer). 10 checkpoints (steps 100-1000) filled the
rented box's 16GB disk exactly at step 1000's optimizer write:

```
RuntimeError: [enforce fail at inline_container.cc:858] . PytorchStreamWriter failed writing file data/7: file write failed
RuntimeError: [enforce fail at inline_container.cc:664] . unexpected pos 226497792 vs 226497680
```

`model_001000.pt`/`meta_001000.json` had already written successfully before
the optimizer write hit the full disk; `optim_001000_rank0.pt` was left at
248MB instead of the expected ~402MB (confirmed truncated by comparing
against the other steps' file sizes). The training process crashed (disk-full
`torch.save` isn't caught anywhere), but the background `sync_checkpoints.py`
poller kept running independently and pushed the truncated file to Drive 3
more times before it was noticed and killed.

Fixed by: killing the sync poller, deleting the corrupted step-1000 files
both locally and on Drive (`rclone delete ... --include "*001000*"`, since
`base_train.py`'s resume path picks the highest step it finds and would have
re-downloaded the bad file otherwise), pruning all older local checkpoints
except step 900 (already safely on Drive, freed disk from 16G/16G to
5.2G free), then resuming manually with `--resume-from-step=900
--save-every=300` (fewer future checkpoints -- ~4 more saves instead of ~9 --
to avoid refilling the same 5.2GB headroom before the run finishes). Loss
resumed exactly where it left off (3.4498 at step 900, matching the
pre-crash trajectory), confirming the optimizer state restore was correct.
Root cause is a real gap worth fixing later (checkpoint retention/pruning),
not something to silently work around again if it recurs.

### 3rd deviation from vendored code: added RTX 5070 Ti to get_peak_flops/get_peak_bandwidth

`nanochat/common.py`'s hardcoded GPU tables (`get_peak_flops`,
`get_peak_bandwidth`) didn't recognize "NVIDIA GeForce RTX 5070 Ti" (too new
for the upstream table), so `bf16_mfu` printed as 0.00 all run --
informational only, doesn't affect training correctness. Looked up real
specs rather than guessing (consumer Blackwell dense tensor-core numbers
aren't obvious from memory): 87.88 TFLOPS dense BF16/FP16 (280 5th-gen
tensor cores), 896GB/s (16GB GDDR7, 256-bit). Cross-checked against the
already-tabled RTX 5090 by tensor-core-count ratio (280/680 x 209.5e12 =
86.3e12, matches the looked-up 87.88e12 within rounding) before trusting it.
Also added the other two GPUs `docs/VASTAI_SETUP.md` explicitly recommends
renting for this project (RTX 4070 Ti, RTX 5070 -- RTX 3090/4090/5090 were
already in the upstream table): RTX 4070 Ti 80.18 TFLOPS dense/504GB/s, RTX
5070 61.7 TFLOPS dense/672GB/s. The RTX 5070 number needed a second check --
one source gave inconsistent FP16 (123.5) vs BF16 (61.7) figures for the
same tensor cores, which shouldn't differ; cross-checked via tensor-core-
count ratio against the now-verified 5070 Ti (192/280 x 87.88e12 = 60.3e12,
matches 61.7e12, not 123.5e12) before trusting the lower number. Didn't add
5080 or any Ampere/Ada data-center card not in this project's docs --
guessing GPU FLOPS tables is exactly the kind of thing worth avoiding.
Doesn't apply retroactively to the currently-running process (loaded at
start), only future runs on these GPUs. This is the 3rd deviation from
"unmodified vendored code" in this project (after `engine.py`'s repetition
controls and `chat_rl.py`'s `--max-train-examples` flag), same policy:
deliberate, logged, not silent.
Sources: [WareDB RTX 5070 Ti specs](https://www.waredb.com/processor/nvidia-geforce-rtx-5070-ti),
[VideoCardz RTX 5070/Ti specs](https://videocardz.com/newz/nvidia-confirms-full-geforce-rtx-5070-ti-specifications-featuring-gb203-and-gb205-gpus),
[WareDB RTX 4070 Ti specs](https://www.waredb.com/processor/nvidia-geforce-rtx-4070-ti),
[WareDB RTX 5070 specs](https://www.waredb.com/processor/nvidia-geforce-rtx-5070).

## 2026-08-11 -- A10 pretrain complete on Vast.ai (real numbers, not estimates)

Full console log recovered (`vastai/runs/2026-08-11_a10_pretrain_console.log`) covers the
complete pretrain from the step-900 resume through to the final checkpoint save -- **SFT is
not in this log** (the pasted console buffer ended right after pretrain finished; SFT either
didn't run yet or its output wasn't captured). Recording what's actually confirmed rather than
assuming SFT happened.

**Pretrain result**: 1905/1905 steps, 499,384,320 tokens (`--target-param-data-ratio=20`,
resumed once from step 900 after the disk-full crash), **28.74 min total training time** on a
single rented RTX 5070 Ti, peak memory **5,386.32 MiB** (well under the 16GB disk-constrained
box's VRAM, and notably lower than `d6`'s 7,960MiB Kaggle run -- smaller batch/no DDP replica
overhead). **Min validation bpb: 0.977060** (monotonic decrease every eval: 1.050740 @step1000,
1.025286 @step1250, 1.003059 @step1500, 0.984603 @step1750, 0.977060 @step1905 -- final = min).

This **beats both `d4` (1.0994) and `d6` (0.9945)** -- a real, if modest (~1.75% vs `d6`),
improvement, consistent with "more depth at fixed width" being a genuine lever, not just noise.

**Speedup, measured on the full run this time** (not just per-step `dt`): `d6` did 464.0M
tokens in 255.72 min on Kaggle T4x2 = 1.814M tok/min. `a10` did 499.4M tokens in 28.74 min on
the rented RTX 5070 Ti = 17.38M tok/min. **~9.6x faster**, in the same range as the earlier
per-step estimate (~9-10x), now confirmed end-to-end including the resume overhead, checkpoint
saves, and Drive sync.

**Bonus data point**: `base_train.py` runs nanochat's own CORE eval bundle automatically at
the end of pretrain (hellaswag, arc_easy, arc_challenge, piqa, winogrande, boolq, etc.,
few-shot, base model) -- something not previously captured for `d4`/`d6` (only `val_bpb` was
logged for those). **CORE metric: 0.0747.** Individual task accuracies mostly near their
respective chance baselines with a few standouts above it (piqa 61.4%, arc_easy 44.6%,
commonsense_qa 37.2%), consistent with the "some real capability, no reliable knowledge" story
already established via BLiMP/chat_eval on `d4`/`d6` -- though this uses a different
methodology (few-shot, base model, not chat-formatted SFT model) so it isn't a clean
apples-to-apples comparison to those numbers. Not going back to compute this for `d4`/`d6`
right now (would need re-running their archived pretrain configs) -- flagged as a gap, not
backfilled retroactively.

Qualitative base-model completions logged at the end (pre-SFT, plain continuation, not
chat-formatted) -- fluent but factually wrong across the board, same pattern as `d4`/`d6`'s
chat-test transcripts: *"The capital of France is the French capital..."*, *"The chemical
symbol of gold is silver..."*, *"The planets of the solar system are: Earth, Mars and
Jupiter."* -- confirms this is a property of scale/data, not something A10's architecture
change fixes.

**Still open**: SFT for `a10` hasn't been confirmed complete (log doesn't cover it) -- next
step is either getting the rest of the console output or re-running SFT via
`vastai/vastai_eval_a10.ipynb`-style notebook for a clean, fully-captured artifact instead of
another lossy terminal paste (this project is switching to real downloaded Jupyter notebooks
for Vast.ai runs going forward, same convention as `kaggle/runs/` -- see that notebook and
`vastai/runs/README.md`).

## Open questions / next up
- Consider a tied-embeddings experiment (`wte`==`lm_head`): at `d4`,
  embeddings are ~46% of total params (untied by upstream design, see
  `nanochat/gpt.py` docstring "untied weights for token embedding and lm_head").
  Tying them would free ~8.4M params' worth of budget at this depth -- unclear
  if that's better spent as more transformer capacity or just makes the
  embedding table itself weaker. Not started.
- Phase 2 (Russian) intentionally deferred until the English side is judged
  "as done as it's going to get" within the free-tier budget.
