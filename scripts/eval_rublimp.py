"""
RuBLiMP grammar eval (Taktasheva et al. 2024, https://aclanthology.org/2024.emnlp-main.522/):
Russian structural equivalent of BLiMP -- 44 phenomena (morphology/syntax/semantics), ~1000
minimal pairs each, built by applying expert perturbation rules to real Russian sentences
(Wikipedia/news/books) rather than hand-written templates. Score: does the model assign higher
total log-likelihood to the grammatical sentence? No fine-tuning, no generation -- same method
as scripts/eval_blimp.py, just a different dataset/column names.

Verified against a live datasets-server row (2026-08-11, see docs/RESEARCH_LOG.md): in
RussianNLP/rublimp, `source_sentence` is the grammatical original, `target_sentence` is the
perturbed/ungrammatical one (e.g. "плечи" -> "плечники", a nonsense suffix).

The category list below is the *authoritative* config list pulled from
https://datasets-server.huggingface.co/splits?dataset=RussianNLP/rublimp -- not transcribed
from the GitHub README's prose phenomena list, which turned out to have 2 wrong names
(adp_government_case -> adposition_government, nominalization_cas -> nominalization_case) that
only surfaced as a real HTTP 400 when actually queried. Lesson: verify category/config names
against the dataset's own metadata API, not a paper's or README's human-written phenomena list.

Dataset: RussianNLP/rublimp on the HF Hub, loaded the same way tasks/*.py load ARC/MMLU/etc
(tasks.common.load_hub_dataset, auto-downloads parquet).

Usage:
    python -m scripts.eval_rublimp -i sft -g ru_a9 --max-pairs 1000
    python -m scripts.eval_rublimp -i sft -g ru_a9 --categories add_new_suffix,negative_concord --max-pairs 50
"""
import argparse
import json

RUBLIMP_CATEGORIES = [
    "add_new_suffix", "add_verb_prefix", "adposition_government",
    "anaphor_agreement_gender", "anaphor_agreement_number", "change_declension_ending",
    "change_declension_ending_has_dep", "change_duration_aspect", "change_repetition_aspect",
    "change_verb_conjugation", "change_verb_prefixes_order",
    "clause_subj_predicate_agreement_gender", "clause_subj_predicate_agreement_number",
    "clause_subj_predicate_agreement_person", "conj_verb_tense", "deontic_imperative_aspect",
    "external_possessor", "floating_quantifier_agreement_case",
    "floating_quantifier_agreement_gender", "floating_quantifier_agreement_number",
    "genitive_subj_predicate_agreement_gender", "genitive_subj_predicate_agreement_number",
    "genitive_subj_predicate_agreement_person", "indefinite_pronoun_to_negative",
    "negative_concord", "negative_pronoun_to_indefinite", "nominalization_case",
    "noun_subj_predicate_agreement_gender", "noun_subj_predicate_agreement_number",
    "noun_subj_predicate_agreement_person", "np_agreement_case", "np_agreement_gender",
    "np_agreement_number", "single_verb_tense", "subj_predicate_agreement_gender_attractor",
    "subj_predicate_agreement_number_attractor", "tense_marker", "transitive_verb",
    "transitive_verb_iobject", "transitive_verb_object", "transitive_verb_passive",
    "transitive_verb_subject", "verb_acc_object", "verb_gen_object", "verb_ins_object",
]


def main():
    import torch
    import torch.nn.functional as F

    from nanochat.common import compute_init, autodetect_device_type
    from nanochat.checkpoint_manager import load_model
    from tasks.common import load_hub_dataset

    parser = argparse.ArgumentParser(description="RuBLiMP grammatical-acceptability eval (Russian)")
    parser.add_argument("-i", "--source", type=str, default="sft", help="base|sft|rl")
    parser.add_argument("-g", "--model-tag", type=str, default=None)
    parser.add_argument("-s", "--step", type=int, default=None)
    parser.add_argument("--categories", type=str, default=None, help="comma-separated subset of RUBLIMP_CATEGORIES; default = all 44")
    parser.add_argument("--max-pairs", type=int, default=1000, help="pairs per category (max ~1000)")
    parser.add_argument("--batch-size", type=int, default=32, help="sentences per forward pass (good+bad interleaved, so use an even number)")
    parser.add_argument("--device-type", type=str, default="")
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    bos = tokenizer.get_bos_token_id()

    categories = args.categories.split(",") if args.categories else RUBLIMP_CATEGORIES

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
        out = []
        for i, length in enumerate(lengths):
            out.append(token_logprobs[i, :length - 1].sum().item())
        return out

    results = {}
    for ci, category in enumerate(categories):
        ds = load_hub_dataset("RussianNLP/rublimp", subset=category, split="train")
        n = min(len(ds), args.max_pairs)
        correct = 0
        half_batch = max(1, args.batch_size // 2)
        for start in range(0, n, half_batch):
            rows = [ds[i] for i in range(start, min(start + half_batch, n))]
            goods = [r["source_sentence"] for r in rows]
            bads = [r["target_sentence"] for r in rows]
            good_lps = batch_logprobs(goods)
            bad_lps = batch_logprobs(bads)
            correct += sum(1 for g, b in zip(good_lps, bad_lps) if g > b)
        acc = correct / n
        results[category] = acc
        print(f"[{ci+1}/{len(categories)}] {category}: {100*acc:.1f}% ({correct}/{n})")

    overall = sum(results.values()) / len(results)
    print(f"\n=== RuBLiMP overall: {100*overall:.2f}% across {len(results)} categories ===")
    print("(50% = chance; see Taktasheva et al. 2024 for human/model reference points)")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
