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

## 2026-08-11 -- A10 SFT + full eval complete: best BLiMP of the three models, SFT bpb doesn't follow pretrain bpb

Running `vastai_eval_a10.ipynb` hit its own round of infra bugs before producing real numbers,
each fixed and logged rather than worked around silently (see the notebook's git history):
`rclone copy` of the whole `base_checkpoints/a10` Drive folder re-triggered a disk-full crash
(pulled all 12 saved steps instead of just the one SFT needs -- fixed with `--include`); bare
`!python3` in a shell cell resolved to a different interpreter than the one Cell 1 installed
deps into, so `chat_sft.py` failed with `ModuleNotFoundError: torch`/`wandb` (fixed by using
`!{sys.executable}` everywhere); Jupyter's own notebook-save kept failing with "database is
locked" (disk had been full recently), so the executed notebook's baked-in output never made
it to disk -- archived the pasted console output as a plain `.log` instead
(`vastai/runs/2026-08-11_a10_sft_eval_console.log`) rather than pretending a notebook artifact
exists that doesn't.

**SFT**: 32/32 steps (SmolTalk bestfit-packing exhausted the mixture in 32 steps at this
`device_batch_size`/`seq_len`, same phenomenon as `d4`'s 125-step cap), 0.31 min, peak memory
3,540MiB, **min val_bpb 0.6285**. Sits *between* `d4` (0.6616) and `d6` (0.6169) -- notably,
**`a10`'s pretrain bpb was the best of the three (0.977060) but its SFT bpb is worse than
`d6`'s**, so "more depth at fixed width" doesn't cleanly carry over from pretrain quality to
SFT quality. Worth remembering as a real, measured non-monotonicity rather than assuming
pretrain bpb predicts everything downstream.

**Chat test** (2 prompts, repetition-penalty fix active): both grammatically fluent but neither
answers the actual question -- "hi" produces a paragraph about correlation coefficients and
outlier analysis; "What is your name?" invents a rambling, self-contradictory biography
("a Greek philosopher... born in the 14th century... my wife, Emily... Leonhard Euler...").
Subjectively **less coherent than `d6`'s SFT chat test** (which also rambled but stayed
roughly on-topic) -- consistent with the SFT bpb result above, though this is a 2-prompt
spot-check, not the systematic `eval_repetition.py` metric (not run yet for `a10` in this
session -- the notebook cell was added afterward, see the open item below).

**`chat_eval.py`** (full, unsampled): ARC-Easy 25.04% (595/2376), ARC-Challenge 22.78%
(267/1172), MMLU 22.94% (3221/14042), GSM8K 0.00% (0/1319), HumanEval 0.00% (0/164),
**ChatCORE -0.0113** -- at/below the random-guessing baseline on every task, same pattern as
`d4`/`d6`, confirms once again that this scale range doesn't buy knowledge/reasoning capability
regardless of which architecture lever is pulled.

**`eval_blimp.py`** (full, 67 x 1000 pairs): **72.13% overall** -- the best of all three models
measured so far (`d4` 66.46%, `d6` 70.31%, `a10` 72.13%), and the one metric where `a10`'s
pretrain-bpb advantage *does* clearly show up downstream. Consistent with the established
BLiMP-high/chat_eval-low split, now with three data points instead of two: extra depth at fixed
width keeps buying grammatical competence, same direction as the `d4`-to-`d6` width increase did.

**Open items carried forward**: `eval_repetition.py` for `a10` still hasn't been run (now in
the notebook, just not executed this session) -- needed before A8's final comparison table can
call itself complete. A9 is next, via the new `vastai/vastai_train_a9.ipynb` (full
tokenizer-retrain -> pretrain -> SFT -> eval pipeline, with `vastai/prune_checkpoints.py`
running in the background to keep only the last 3 local steps -- direct fix for the disk-full
pattern hit twice now on `a10`).

## 2026-08-11 -- A9 complete: clean sweep across every metric, the best English model yet

Ran `vastai_train_a9.ipynb` on the same rented RTX 5070 Ti right after A10. Two real bugs
caught and fixed before/during the run (both logged, not silently worked around):
`base_train.py` has no `--vocab-size` flag (it reads vocab_size from whichever tokenizer is
in `NANOCHAT_BASE_DIR` -- Cell 3 already trains the right one there, so the flag was just an
argparse error); and Jupyter lost the SFT cell's displayed output a second time ("database is
locked", same as A10) -- recovered the real `val_bpb` directly from the saved checkpoint's
`meta_000032.json` rather than guessing or leaving it blank. Both notebooks now `tee` SFT
output to a log file on disk so this can't happen a third time.

**Pretrain**: `--vocab-size=16384 --depth=7` (default aspect_ratio -> `model_dim=512`),
72,351,976 params, 2320 steps, 608.17M tokens (ratio=20), **37.87 min** on the RTX 5070 Ti
(slower wall-clock than A10's 28.74 min despite fewer total params -- more tokens needed
(608M vs 499M) and heavier per-token compute at `model_dim=512` vs A10's 384, consistent with
A9's higher non-embedding fraction). Peak memory only 3,443MiB (much lower than `d6`/`a10` --
smaller vocab means much less embedding-table memory). **Min validation bpb: 0.956752** --
beats `d4` (1.0994), `d6` (0.9945), and `a10` (0.977060). Base-model CORE metric: **0.0949**,
also the best of the two data points we have this metric for (`a10`: 0.0747) -- a genuinely
comparable apples-to-apples comparison since both used the same eval_bundle methodology.

**SFT**: 32/32 steps (SmolTalk bestfit-packing exhausted at the same step count as `a10`,
coincidence of similar packing behavior, not a bug). **Min validation bpb: 0.612585** -- beats
`d4` (0.6616), `d6` (0.6169), and `a10` (0.6285). Unlike `a10`, where the pretrain-bpb
advantage didn't carry over to SFT, **A9's advantage holds at every stage measured so far**.

**Chat test** (2 prompts, repetition-penalty fix active): "hi" -> a coherent but off-topic
paragraph about a hydroponic-garden AI project; "What is your name?" -> grammatically fluent
but internally invented/incoherent ("My name is Emily Liam 'Meet' John... 19-year-old from the
Baltic Sea..."). Same qualitative pattern as every other model so far -- fluent, doesn't
reliably answer the actual question.

**`eval_repetition.py`** (first real run of this metric on an A9/A10-generation model):
distinct-1 **0.8167**, distinct-2 **0.9865**, 0/30 loops -- edges out `d4` (0.7917/0.9848) and
`d6` (0.8039/0.9864), the best of the three models with this metric measured. (`a10` still
doesn't have this number -- the notebook cell was added after that session ran.)

**`chat_eval.py`**, run with **`-x 200`** (sampled, not the full unsampled test sets `d4`/`d6`/
`a10` got -- GSM8K/HumanEval's generative evaluation dominated `a10`'s full run, ~46 of ~50
total minutes, for a task already floored at 0% on every model tried; capping every task to
200 problems keeps `chat_eval.py`'s `all_tasks_were_evaluated` gate satisfied so ChatCORE still
computes, on a smaller but still-informative sample): ARC-Easy 24.00% (48/200), ARC-Challenge
**31.00%** (62/200 -- ~6pp above the 25% baseline, but at n=200 the binomial standard deviation
is ~3pp, so this is plausibly noise, not a real signal, and not treated as one), MMLU 23.50%
(47/200), GSM8K 0.00% (0/200), HumanEval 0.00% (0/164), **ChatCORE +0.0093** -- the only
positive ChatCORE of the four models (`d4` -0.0109, `d6` -0.0127, `a10` -0.0113), though this
isn't a clean apples-to-apples comparison given the sampled vs full-test-set methodology
difference, and all four values are close enough to zero that "no real knowledge/reasoning
capability" remains the honest read regardless of sign.

**`eval_blimp.py`** (full, 67 x 1000 pairs, unsampled -- same methodology as the other three):
**73.48% overall** -- the best of all four English models (`d4` 66.46%, `d6` 70.31%, `a10`
72.13%, `a9` 73.48%), continuing the same trend BLiMP has shown at every step: reallocating
budget away from a 32768-entry vocab toward more transformer capacity (whether via `a10`'s
extra depth or `a9`'s smaller vocab) reliably buys grammatical competence, monotonically so
far across all three architecture variants tried.

**Overall read**: A9 is a genuinely clean sweep -- best pretrain bpb, best SFT bpb, best CORE,
best BLiMP, best repetition metric -- unlike A10, which won on pretrain bpb and BLiMP but
landed worse on SFT bpb than `d6`. Tentative conclusion (three data points, not enough to be
fully confident): reallocating the embedding-dominated budget toward *smaller vocabulary* was
a more effective lever at this scale than *more depth at fixed width*. Still true across every
model so far: none of this moves chat_eval/GSM8K/HumanEval off the random-guessing floor --
that gap remains a scale problem, not an architecture problem.

## 2026-08-11 -- Vast.ai session closed: real wall-clock vs actual GPU-compute time

Instance rented 06:22 PM, deleted 10:23 PM -- **~4h01m total rental**, ~$1 all-in (well under
the earlier ~$5 budget). Worth being honest about what that 4 hours actually contained, since
it's not 4 hours of GPU compute: A10 pretrain (28.74 min) + SFT (0.31 min) + full eval (~50
min) + A9 pretrain (37.87 min) + SFT (~1 min) + eval (~20 min, sampled) adds up to **roughly
2.3h of actual training/eval compute** -- the remaining ~1.7h was debugging (the disk-full
crash and resume, the credential/interpreter/`--vocab-size` bugs on the A9 notebook, hunting
for a working shell method after the SSH connect string wasn't easy to find in the Vast.ai UI)
and idle time between steps waiting on a human to paste the next command/log. Not a criticism
of the tooling -- this was the first real session on this backend, and every bug hit here is
now fixed in the scripts/notebooks for next time (see the entries above). Logging the real
ratio because "$1 for two full architecture experiments" undersells how much of that hour was
spent on infra shakedown rather than productive compute, and a repeat run on this now-debugged
pipeline should take meaningfully less wall-clock.

## 2026-08-11 -- Phase B planning research (corpus, tokenizer, SFT data, eval)

Researched before renting anything, so the next Vast.ai session can go straight to building
instead of researching mid-session. Web search + HF dataset-card fetches, not guessed.

### Pretraining corpus

Checked the three candidates already scouted (`docs/RESEARCH_LOG.md`/README's Phase 2 section):
**FineWeb-2** (`HuggingFaceFW/fineweb-2`, config `rus_Cyrl`), **CulturaX** (`ru`), **HPLT**.
Russian has "hundreds of billions of tokens" in FineWeb-2 alone -- data volume is a total
non-issue at this project's scale (we used ~500M-600M tokens per English run out of a
400-billion-token English corpus, i.e. a rounding error of what's available). No Russian
equivalent of ClimbMix (the clustering-based mixture that beat FineWeb-Edu for *English* on
this exact pipeline, see the 2026-08-10 entry) was found -- no comparison study exists for
Russian specifically, so there's no evidence-based reason to deviate from the same
well-documented, actively-maintained default this project already trusted for English
(FineWeb's pipeline, same team). **Leaning FineWeb-2 `rus_Cyrl`** unless something better turns
up, not a locked decision. Structure: `data/rus_Cyrl/{train,test}/*.parquet`, loadable via
`load_dataset("HuggingFaceFW/fineweb-2", "rus_Cyrl")` or direct parquet paths -- similar shape
to `nanochat/dataset.py`'s current hardcoded ClimbMix loader, so swapping it will need actual
code changes (a 4th deviation from vendored code, after `engine.py`/`chat_rl.py`/`common.py`),
not just a config flag.
Sources: [FineWeb-2 dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2),
[FineWeb2: One Pipeline to Scale Them All](https://www.researchgate.net/publication/393065691_FineWeb2_One_Pipeline_to_Scale_Them_All_--_Adapting_Pre-Training_Data_Processing_to_Every_Language).

### Tokenizer -- good news: no code changes needed, but vocab_size is a real open question

Checked `nanochat/tokenizer.py`'s `SPLIT_PATTERN` (`nanochat/tokenizer.py:26`): it's built on
Unicode property classes (`\p{L}`, `\p{N}`), not literal Latin character ranges, so it already
handles Cyrillic letter-runs correctly with zero code changes -- the only English-specific bit
(`'(?i:[sdmt]|ll|ve|re)`, contraction handling) simply won't match Russian text and falls
through to the general rule harmlessly. Confirmed against a live `karpathy/nanochat` GitHub
issue (#107, "multilingual tokenization support") raising exactly this as an open question for
the upstream project -- our vendored copy already clears it structurally.

**What isn't already settled**: `vocab_size`. Cyrillic/morphologically-rich languages are a
documented source of poor tokenizer fertility (more distinct wordforms per lemma than English
due to case/gender/number inflection) -- multiple papers on Ukrainian/Kazakh tokenization found
this pushes toward wanting a *larger* vocab for good compression, the opposite direction from
A9's finding that a *smaller* vocab (16384 vs 32768) won on English at this model scale. Since
that tokenizer is trained on the target corpus from scratch either way (not reusing a
multilingual pretrained one, so the "multilingual tokenizer wastes budget on other languages"
failure mode doesn't directly apply to us), the fertility question is really "how well does a
16384 or 32768 Russian-only BPE vocab compress Russian specifically" -- an empirical question,
not something to assume transfers from the English A9 result. **Recommendation: don't just
reuse A9's vocab_size=16384 on faith -- run a small vocab_size sweep (e.g. 16384 vs 32768) on
the Russian corpus, same architecture, and let val_bpb/RuBLiMP decide**, same spirit as A9 vs
A10 for English.
Sources: [SozKZ: Efficient Small LMs for Kazakh](https://arxiv.org/pdf/2603.20854),
[The Tokenizer Tax Across 25 European Languages](https://arxiv.org/pdf/2605.24718),
[nanochat issue #107](https://github.com/karpathy/nanochat/issues/107).

### SFT data

Upstream nanochat uses SmolTalk (English) for SFT here. Russian equivalents found, ranked by
fit: **`IlyaGusev/saiga_scored`** (41,609 rows, multilingual but includes Russian, needs
filtering to `language=="ru"`, scored 1-10 by quality -- can filter to high-scoring rows only,
actively maintained, used to train 130+ models per its HF card) is the closest match to
SmolTalk's role (general-purpose instruction/chat mixture, not a single narrow task). Bigger
but less curated alternatives if 41.6K filtered-to-Russian rows turns out too small:
**`d0rj/ru-instruct`** (combined from several sources, deduplicated) or Vikhr's **ruFLAN**
(~500K, per their paper). Worth remembering: **dataset size was never actually a bottleneck for
SFT on this project** -- SmolTalk's 460K rows still only produced 32-125 *packed* training
steps for our tiny models (bestfit-packing exhausts the mixture almost immediately at this
scale), so even the smallest of these Russian options is very unlikely to be the limiting
factor. Sources: [rulm](https://github.com/IlyaGusev/rulm),
[saiga_scored](https://huggingface.co/datasets/IlyaGusev/saiga_scored),
[Vikhr paper](https://arxiv.org/html/2405.13929v5).

### Eval -- BLiMP has a real Russian equivalent

**RuBLiMP** (`RussianNLP/rublimp`, [Taktasheva et al. 2024](https://aclanthology.org/2024.emnlp-main.522/))
exists and is structurally close to BLiMP: 45 linguistic phenomena, ~1000 pairs each (~45K
total), built the same way (grammatical vs. perturbed-ungrammatical sentence pairs) but via
UD-parser-driven perturbation rather than hand-written templates. Column names differ from
BLiMP (`source_sentence`/`target_sentence`, not `sentence_good`/`sentence_bad`) and which one is
grammatical needs verifying against the paper before assuming a direction -- but the same
batched log-prob-comparison method `scripts/eval_blimp.py` already uses should port over with a
new loader function, not a rewrite. MMLU/GSM8K/HumanEval have no meaningful Russian equivalent
worth chasing: this project's own English results already showed those tasks floor at 0%
regardless of architecture at this scale (A9 vs A10 vs `d4` vs `d6`, all four), so a Russian
version would almost certainly show the same floor for the same reason (scale, not language) --
not worth the eval time. **Planned Russian eval stack: val_bpb, RuBLiMP, `eval_repetition.py`
(already language-agnostic, no changes needed), qualitative chat test.** Skipping chat_eval.py
entirely for Russian unless there's a specific reason to revisit that later.

### Recommendation on structure (single notebook vs. ladder)

User's question: repeat the `d4`->`d6` ladder (ratchet up one variable at a time, build a
comparison table) or go straight to an A9-style single combined pipeline. Leaning toward a
**middle path**: architecture search is already answered for this scale (A9's vocab-reallocated
`depth=7` shape won cleanly for English, no strong reason to expect the opposite for Russian) --
re-running the full `d4`/`d6`/`a10`/`a9` ladder for Russian would mostly just re-derive an
already-known result at real GPU-rental cost. The one genuinely open question is the
Russian-specific `vocab_size` sweep above, which *is* worth a small ladder (2 vocab sizes,
same architecture) rather than assuming. So: **small vocab_size ladder (16384 vs 32768) at
A9's architecture shape, pick the winner via bpb/RuBLiMP, then one full pretrain+SFT+eval
pipeline on the winner** -- cheaper and more informative than either extreme (blind reuse of
A9's exact config, or re-running the entire English ladder from scratch).

## 2026-08-11 -- Phase B build: real data validated locally before ever renting a GPU

Built the actual code for Phase B (not just researched it) and tested every non-GPU-dependent
piece against real live data locally, catching real bugs before they'd cost rented-GPU time.

**4th deviation from vendored code**: `nanochat/dataset.py`'s `DATA_DIR` is now
`base_data_{NANOCHAT_CORPUS_NAME}` (env var, defaults to `"climbmix"` -- zero behavior change
for every existing English run). `list_parquet_files()`/`parquets_iter_batched()` were already
generic (just read `*.parquet` files with a `text` column from *some* directory); the only
ClimbMix-specific parts (`BASE_URL`, `MAX_SHARD`, the `__main__` downloader) are untouched and
simply unused for a different corpus. This one env var is enough for `tok_train.py`/
`base_train.py`/`nanochat/dataloader.py` to point at a Russian corpus with **zero further code
changes** in any of them -- checked by reading their import chains, not assumed.

**Corpus size, checked not guessed**: FineWeb-2's `rus_Cyrl` auto-export parquet shards are
**~4.84GB each** (verified via a real HTTP HEAD request -- `content-length: 4841277045`), not
ClimbMix-sized (~40MB). 440 shards exist; this project's whole token budget per run
(~500-600M tokens) fits comfortably inside 1-2 of them, so `scripts/download_ru_corpus.py`
(new, not a vendored-code change) defaults to downloading just 2 (1 train + 1 val, mirroring
`dataset.py`'s own "last shard = val" convention) -- ~9.7GB, still a lot more than ClimbMix's
footprint, so the next Vast.ai rental needs 50GB+ disk, not the ~16-30GB that sufficed for
English.

**Tokenizer**: `nanochat/tokenizer.py` needs zero changes (already covered in the planning
entry above) -- confirmed again here, no new findings, just re-verified nothing regressed.

**SFT dataset, tested against real data**: `IlyaGusev/saiga_scored` filtered to
`language=="Russian"` and `opus_score>=8` gives **28,237 candidate rows**. Found a real bug
while testing locally: the dataset uses `"bot"` as the assistant role (not `"assistant"`,
which `nanochat/tokenizer.py`'s renderer requires literally) -- confirmed by checking the
actual distinct role values present (`{"user", "system", "bot"}`), not guessed. Normalized
`"bot"` -> `"assistant"` in the new `tasks/saiga.py`. Also found **39 rows (~0.14%)** start
directly with an assistant turn (no preceding user message) and can't satisfy the strict
user/assistant alternation the renderer enforces -- dropped at init rather than left to crash
a rented-GPU training run mid-epoch. **28,198 rows survive, and every single one round-trips
through the full validation successfully** (tested exhaustively, not sampled). Added a
`--sft-dataset` flag to `scripts/chat_sft.py` (default `smoltalk`, `saiga_ru` swaps in this
new task with a 90/10 in-class held-out split since the source has no official train/test
split) -- existing English runs are unaffected by default.

**Eval, tested against real data -- and a real bug caught two ways**: `scripts/eval_rublimp.py`
(new, ported from `eval_blimp.py`'s exact log-prob-comparison method) -- confirmed via a live
datasets-server row that `source_sentence` is the grammatical original and `target_sentence`
is the perturbed one (e.g. "плечи" -> "плечники", a nonsense suffix), not assumed from the
dataset card alone. The category list was first transcribed from the GitHub README's prose
phenomena list -- **wrong in 2 places**, only caught by actually querying every config: `adp_
government_case` should be `adposition_government`, `nominalization_cas` should be
`nominalization_case` (missing an 'e'). Along the way, a second bug surfaced: the script had
no `if __name__ == "__main__":` guard, so a naive `from scripts.eval_rublimp import
RUBLIMP_CATEGORIES` (meant to just grab the category list for a validation check) actually
executed the *entire* eval -- loaded a local `d6` checkpoint and ran a real RuBLiMP pass against
it, on CPU, using whatever CLI args happened to be live at that moment. Fixed by wrapping the
whole body in `main()` behind the standard guard. Re-verified the corrected 45-name list
straight from the dataset's own metadata (`datasets-server.huggingface.co/splits`), not the
README, and **all 45/45 configs now load real data (1000 pairs each) confirmed** (2 initial
network blips on retry, not name errors -- both categories loaded fine on a second attempt).
Also added `--lang ru` to `scripts/eval_repetition.py` -- the existing 10 prompts were
English-only, and feeding an English-tuned prompt set to a Russian-only model would just
measure out-of-distribution garbage (this project's own English models already showed exactly
that failure mode on Russian input, see the 2026-08-10 entry), not a real repetition-loop
signal.

**Notebook**: `vastai/vastai_train_ru.ipynb` -- downloads the corpus once (shared across a
vocab_size sweep via symlink, same pattern A9 used for the English corpus), trains both
16384/32768 tokenizers into isolated base dirs + distinct Drive paths
(`gdrive:tokenizer_ru_v16384`/`_v32768`, never the shared `gdrive:tokenizer`), pretrains both,
prints val_bpb + a quick sampled RuBLiMP pass on both base checkpoints, then requires a manual
`WINNER_VOCAB` decision (deliberately not auto-picked -- same judgment-call policy as A9 vs
A10) before SFT-ing and fully evaluating the winner only.

## 2026-08-14 -- Phase B run complete: vocab_size sweep decided the opposite way from English, SFT + eval done

Ran `vastai/vastai_train_ru.ipynb` end to end on a newly rented RTX 5070 Ti (same GPU class as
the A9/A10 session). Full notebook output checked cell by cell for hidden failures (not just
skimmed) -- exactly one error output in the whole run, and it was the intentional
`WINNER_VOCAB = None` assertion guard (Cell 6, by design, requires a manual decision, see the
Phase B build entry above) -- everything else, including all 45/45 RuBLiMP categories on both
base checkpoints and the full SFT+eval pass, completed with no crashes, no silent bugs.

**Vocab_size sweep, both at A9's `depth=7` architecture shape**:

| | `vocab=16384` | `vocab=32768` |
|---|---|---|
| total params | 72,351,976 (identical to `a9`) | 122,683,624 |
| pretrain steps / tokens | 2320 / 608,174,080 (ratio=20) | 2960 / 775,946,240 (ratio=20) |
| pretrain wall-clock | 36.55 min | 55.71 min |
| peak memory | 3,443.53 MiB | 6,294.17 MiB |
| **min val_bpb** | 0.652795 | **0.616911** |
| base-model CORE metric | 0.0531 | **0.0630** |
| RuBLiMP (base, sampled 200/category) | 92.36% | **93.19%** |

**Both metrics agree: `vocab=32768` wins.** This is the *opposite* direction from A9's English
finding (`vocab=16384` beat `32768` there) -- confirming, not just plausibly supporting, the
prediction made in the Phase B planning entry above: Cyrillic/morphologically-rich Russian
needs more vocab capacity to compress well, so the embedding-vs-transformer-capacity tradeoff
that favored a smaller vocab for English tips the other way for Russian. Worth stating plainly:
**A9's architecture lever does not transfer across languages by default** -- the vocab_size
sweep was the right call, not a formality, since blindly reusing `vocab=16384` on faith would
have shipped the worse of the two options.

`WINNER_VOCAB = 32768` set manually, loser's (`16384`'s) local checkpoints freed per Cell 7 (already
safely on Drive under `gdrive:base_checkpoints/ru_v16384`, not deleted there, just off the
rented box's disk).

**SFT** (`saiga_ru`, `IlyaGusev/saiga_scored` filtered to Russian, 25,378 train rows after the
90/10 split): bestfit-packing exhausted the mixture in **32/32 steps, 0.42 min** -- same
phenomenon as every English SFT run so far (`d4` 125 steps, `d6`/`a10`/`a9` 32 steps), packing
efficiency scales with dataset size not language. **Min validation bpb: 0.4785**, peak memory
4,438.38MiB.

**Chat test** (2 prompts, repetition-penalty fix active): both responses are grammatically
Russian-shaped but semantically garbage -- code fragments, HTML tags, and invented pseudo-technical
terms mixed into the output rather than an answer to either prompt ("привет" -> a paragraph
about a `text-to-value` method mixing Python/JS syntax fragments; "Как тебя зовут?" -> similar
code-flavored non-answer). Same qualitative ceiling documented for every English model too
(fluent surface form, no reliable content) -- confirms this is a scale problem that transfers
across languages exactly as expected, not something SFT-data-language-specific.

**`eval_repetition.py --lang ru`**: avg distinct-1 **0.8567**, avg distinct-2 **0.9717**, **0/30
generations looped** (worst max-4gram-repeat: 1) -- as clean a result as any English model's,
confirming the repetition-penalty/no-repeat-ngram fix (`Engine.generate()`, English-agnostic by
construction) works identically for Russian.

**`eval_rublimp.py`** (full, all 45 categories x 1000 pairs, SFT checkpoint): **91.10% overall**
-- down from the base checkpoint's 93.19% (sampled) / would-be-full comparison, a real
regression from SFT, consistent with the general pattern that SFT trades some of the pure
grammatical competence measured by BLiMP/RuBLiMP for chat-format fluency (not previously
measurable this cleanly on the English side since `eval_blimp.py` was never run on both base
*and* SFT checkpoints of the same model to isolate the SFT effect specifically). Category-level
breakdown mirrors the base model's weak spots almost exactly (`external_possessor` still worst
at 60.1%, `transitive_verb_object`/`transitive_verb_subject` still the other weak cluster) --
the SFT regression is a roughly uniform shift, not a specific category collapsing.

**Verified for real, not just trusted from logs**: used the Google Drive MCP tools directly to
confirm `chatsft_checkpoints/ru_v32768/model_000032.pt` (322,985,075 bytes) actually landed on
Drive, timestamp matching the local save log to the second -- same "check the actual sync
target, not just the sync tool's own exit code" policy used earlier in this project's Drive
verification pass.

Instance rental: not yet closed out with a final duration figure at the time of this entry
(session still had cleanup/deletion pending) -- will note actual $ cost in a follow-up entry
once the instance is confirmed deleted, same honest-timing policy as the A9/A10 session.

**Bottom line for Phase B**: the pipeline built and locally-validated in the entry above ran
on real rented GPU time with zero code bugs surfacing (everything caught was caught locally,
before ever touching a GPU) -- the only genuinely new finding from the GPU run itself was the
vocab_size decision, which came out opposite to the English case exactly as hypothesized.
Phase B (B0-B8) is complete.

## Open questions / next up
- Consider a tied-embeddings experiment (`wte`==`lm_head`, A-opt-1): untied
  by upstream design (see `nanochat/gpt.py` docstring). Given A9's clean
  sweep, should be sized against **`a9`'s budget** (72.35M, `vocab_size=16384`)
  now that it's the leading architecture, not `d4`'s original 46%-embedding
  framing -- `a9` is already less embedding-dominated (smaller vocab), so
  the params freed by tying would be a smaller fraction here than the
  original `d4` estimate suggested. Not started, not sized yet.
- Phase 2 (Russian) intentionally deferred until the English side is judged
  "as done as it's going to get" within the free-tier/rented budget.
