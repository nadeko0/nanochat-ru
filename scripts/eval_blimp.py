"""
BLiMP grammar eval (Warstadt et al. 2020, https://arxiv.org/abs/1912.00582):
67 categories of grammatical phenomena, each with 1000 minimal pairs -- one
grammatical sentence, one with a single grammatical violation. Score: does
the model assign higher total log-likelihood to the grammatical sentence?
No fine-tuning, no generation -- just a forward pass per sentence, same as
loss_eval.py's bpb computation.

Dataset: nyu-mll/blimp on the HF Hub, loaded the same way tasks/*.py load
ARC/MMLU/etc (tasks.common.load_hub_dataset, auto-downloads parquet).

Usage:
    python -m scripts.eval_blimp -i sft -g d6 --max-pairs 1000
    python -m scripts.eval_blimp -i sft -g d6 --categories adjunct_island,passive_1 --max-pairs 50
"""
import argparse
import json
import torch
import torch.nn.functional as F

from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from tasks.common import load_hub_dataset

BLIMP_CATEGORIES = [
    "adjunct_island", "anaphor_gender_agreement", "anaphor_number_agreement",
    "animate_subject_passive", "animate_subject_trans", "causative",
    "complex_NP_island", "coordinate_structure_constraint_complex_left_branch",
    "coordinate_structure_constraint_object_extraction", "determiner_noun_agreement_1",
    "determiner_noun_agreement_2", "determiner_noun_agreement_irregular_1",
    "determiner_noun_agreement_irregular_2", "determiner_noun_agreement_with_adj_2",
    "determiner_noun_agreement_with_adj_irregular_1", "determiner_noun_agreement_with_adj_irregular_2",
    "determiner_noun_agreement_with_adjective_1", "distractor_agreement_relational_noun",
    "distractor_agreement_relative_clause", "drop_argument", "ellipsis_n_bar_1",
    "ellipsis_n_bar_2", "existential_there_object_raising", "existential_there_quantifiers_1",
    "existential_there_quantifiers_2", "existential_there_subject_raising",
    "expletive_it_object_raising", "inchoative", "intransitive",
    "irregular_past_participle_adjectives", "irregular_past_participle_verbs",
    "irregular_plural_subject_verb_agreement_1", "irregular_plural_subject_verb_agreement_2",
    "left_branch_island_echo_question", "left_branch_island_simple_question",
    "matrix_question_npi_licensor_present", "npi_present_1", "npi_present_2",
    "only_npi_licensor_present", "only_npi_scope", "passive_1", "passive_2",
    "principle_A_c_command", "principle_A_case_1", "principle_A_case_2",
    "principle_A_domain_1", "principle_A_domain_2", "principle_A_domain_3",
    "principle_A_reconstruction", "regular_plural_subject_verb_agreement_1",
    "regular_plural_subject_verb_agreement_2", "sentential_negation_npi_licensor_present",
    "sentential_negation_npi_scope", "sentential_subject_island", "superlative_quantifiers_1",
    "superlative_quantifiers_2", "tough_vs_raising_1", "tough_vs_raising_2", "transitive",
    "wh_island", "wh_questions_object_gap", "wh_questions_subject_gap",
    "wh_questions_subject_gap_long_distance", "wh_vs_that_no_gap", "wh_vs_that_no_gap_long_distance",
    "wh_vs_that_with_gap", "wh_vs_that_with_gap_long_distance",
]

parser = argparse.ArgumentParser(description="BLiMP grammatical-acceptability eval")
parser.add_argument("-i", "--source", type=str, default="sft", help="base|sft|rl")
parser.add_argument("-g", "--model-tag", type=str, default=None)
parser.add_argument("-s", "--step", type=int, default=None)
parser.add_argument("--categories", type=str, default=None, help="comma-separated subset of BLIMP_CATEGORIES; default = all 67")
parser.add_argument("--max-pairs", type=int, default=1000, help="pairs per category (max 1000)")
parser.add_argument("--batch-size", type=int, default=32, help="sentences per forward pass (good+bad interleaved, so use an even number)")
parser.add_argument("--device-type", type=str, default="")
args = parser.parse_args()

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
bos = tokenizer.get_bos_token_id()

categories = args.categories.split(",") if args.categories else BLIMP_CATEGORIES


@torch.no_grad()
def batch_logprobs(texts):
    """Sum log-likelihood of each text (with a BOS prefix), batched with right-padding."""
    all_ids = [[bos] + tokenizer.encode(t) for t in texts]
    lengths = [len(ids) for ids in all_ids]
    max_len = max(lengths)
    padded = [ids + [bos] * (max_len - len(ids)) for ids in all_ids]
    ids_t = torch.tensor(padded, dtype=torch.long, device=device)
    logits = model(ids_t)  # (B, T, V)
    logprobs = F.log_softmax(logits[:, :-1].float(), dim=-1)  # (B, T-1, V)
    targets = ids_t[:, 1:]  # (B, T-1)
    token_logprobs = logprobs.gather(2, targets.unsqueeze(2)).squeeze(2)  # (B, T-1)
    # mask out padding positions (anything beyond this sentence's real length - 1 targets)
    out = []
    for i, length in enumerate(lengths):
        out.append(token_logprobs[i, :length - 1].sum().item())
    return out


results = {}
for ci, category in enumerate(categories):
    ds = load_hub_dataset("nyu-mll/blimp", subset=category, split="train")
    n = min(len(ds), args.max_pairs)
    correct = 0
    half_batch = max(1, args.batch_size // 2)
    for start in range(0, n, half_batch):
        rows = [ds[i] for i in range(start, min(start + half_batch, n))]
        goods = [r["sentence_good"] for r in rows]
        bads = [r["sentence_bad"] for r in rows]
        good_lps = batch_logprobs(goods)
        bad_lps = batch_logprobs(bads)
        correct += sum(1 for g, b in zip(good_lps, bad_lps) if g > b)
    acc = correct / n
    results[category] = acc
    print(f"[{ci+1}/{len(categories)}] {category}: {100*acc:.1f}% ({correct}/{n})")

overall = sum(results.values()) / len(results)
print(f"\n=== BLiMP overall: {100*overall:.2f}% across {len(results)} categories ===")
print(f"(50% = chance, ~96.4% = human agreement per Warstadt et al. 2020)")
print(json.dumps(results, indent=2))
