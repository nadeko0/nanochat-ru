"""
Objective repetition-loop metric, to replace eyeballing chat_cli transcripts.

Generates completions for a fixed set of prompts and reports:
- distinct-1 / distinct-2: fraction of unique unigrams/bigrams among all generated
  (Li et al. 2016, "A Diversity-Promoting Objective Function for Neural Conversation
  Models" -- a standard, established diversity metric, not invented for this project).
  Lower distinct-n means more repetitive text.
- max_ngram_repeat: the highest number of times any single n-gram (n=4) repeats in one
  generation -- a direct, interpretable "did it get stuck in a loop" signal (distinct-n
  alone can look "fine" on average while still missing an obvious single bad loop).

Usage:
    python -m scripts.eval_repetition --source sft --model-tag d4 --repetition-penalty 1.0 --no-repeat-ngram-size 0
    python -m scripts.eval_repetition --source sft --model-tag d6 --repetition-penalty 1.2 --no-repeat-ngram-size 3
"""
import argparse
import json
from collections import Counter

from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

PROMPTS = [
    "hi",
    "What is your name?",
    "Tell me about your day.",
    "What do you like to do for fun?",
    "Can you help me plan a trip?",
    "What is the weather like today?",
    "Describe your favorite food.",
    "What are your thoughts on music?",
    "How do you make a sandwich?",
    "Tell me a short story.",
]

parser = argparse.ArgumentParser(description="Objective repetition-loop metric for chat generation")
parser.add_argument("-i", "--source", type=str, default="sft", help="sft|base|rl")
parser.add_argument("-g", "--model-tag", type=str, default=None)
parser.add_argument("-s", "--step", type=int, default=None)
parser.add_argument("-t", "--temperature", type=float, default=0.6)
parser.add_argument("-k", "--top-k", type=int, default=50)
parser.add_argument("--repetition-penalty", type=float, default=1.0)
parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
parser.add_argument("--max-tokens", type=int, default=100)
parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
parser.add_argument("--device-type", type=str, default="")
args = parser.parse_args()

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
engine = Engine(model, tokenizer)

bos = tokenizer.get_bos_token_id()
user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
assistant_start = tokenizer.encode_special("<|assistant_start|>")


def ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def score(token_ids):
    uni = ngrams(token_ids, 1)
    bi = ngrams(token_ids, 2)
    four = ngrams(token_ids, 4)
    distinct1 = len(set(uni)) / len(uni) if uni else 0.0
    distinct2 = len(set(bi)) / len(bi) if bi else 0.0
    max_repeat = max(Counter(four).values()) if four else 0
    return distinct1, distinct2, max_repeat


results = []
for prompt in PROMPTS:
    prompt_tokens = [bos, user_start] + tokenizer.encode(prompt) + [user_end, assistant_start]
    for seed in args.seeds:
        gen = []
        for token_col, _ in engine.generate(
            prompt_tokens, num_samples=1, max_tokens=args.max_tokens,
            temperature=args.temperature, top_k=args.top_k, seed=seed,
            repetition_penalty=args.repetition_penalty, no_repeat_ngram_size=args.no_repeat_ngram_size,
        ):
            gen.append(token_col[0])
        d1, d2, max_repeat = score(gen)
        results.append({
            "prompt": prompt, "seed": seed, "num_tokens": len(gen),
            "distinct1": d1, "distinct2": d2, "max_4gram_repeat": max_repeat,
        })

n = len(results)
avg_d1 = sum(r["distinct1"] for r in results) / n
avg_d2 = sum(r["distinct2"] for r in results) / n
worst_repeat = max(r["max_4gram_repeat"] for r in results)
looped = sum(1 for r in results if r["max_4gram_repeat"] >= 3)  # same 4-gram 3+ times = a real loop

print(f"\n=== {args.model_tag} ({args.source}) | repetition_penalty={args.repetition_penalty} no_repeat_ngram_size={args.no_repeat_ngram_size} ===")
print(f"generations: {n} ({len(PROMPTS)} prompts x {len(args.seeds)} seeds)")
print(f"avg distinct-1: {avg_d1:.4f}")
print(f"avg distinct-2: {avg_d2:.4f}")
print(f"worst max-4gram-repeat: {worst_repeat}")
print(f"generations with a loop (same 4-gram repeated >=3x): {looped}/{n}")
print(json.dumps(results, indent=2))
