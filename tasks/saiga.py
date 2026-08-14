"""
IlyaGusev/saiga_scored -- SFT dataset for the Saiga family of Russian chat models.
https://huggingface.co/datasets/IlyaGusev/saiga_scored

Multilingual (8 languages) collection scored 1-10 for quality by an LLM judge; filtered here to
Russian rows only (language=="Russian", confirmed via a live datasets-server row, see
docs/RESEARCH_LOG.md 2026-08-11) and to a minimum quality score, the closest available analog
to SmolTalk's role in the English pipeline (general-purpose instruction/chat mixture, not a
single narrow task). No official train/test split (one split, "train", only) -- the caller is
expected to carve out a held-out slice via the inherited start/stop/step (e.g.
SaigaRu(stop=N) for train, SaigaRu(start=N) for val), same mechanism tasks.common.Task already
exposes elsewhere in this project (e.g. chat_rl.py's --max-train-examples).
"""

import random

from tasks.common import Task, load_hub_dataset

class SaigaRu(Task):
    """ saiga_scored, filtered to Russian rows above a quality threshold, deterministically shuffled. """

    def __init__(self, min_score=8, **kwargs):
        super().__init__(**kwargs)
        full_ds = load_hub_dataset("IlyaGusev/saiga_scored", split="train")
        candidates = [
            full_ds[i] for i in range(len(full_ds))
            if full_ds[i]["language"] == "Russian" and (full_ds[i]["opus_score"] or 0) >= min_score
        ]
        # A small fraction (~0.14%, checked locally 2026-08-11) of rows start directly with a
        # "bot" turn -- no preceding user message, so they can't satisfy the strict
        # user/assistant alternation nanochat/tokenizer.py's renderer requires. Drop them here
        # at init rather than letting a rented-GPU training run crash on one mid-epoch.
        rows = []
        for row in candidates:
            messages = self._normalize(row["messages"])
            if self._is_valid(messages):
                rows.append(messages)
        random.Random(42).shuffle(rows)
        self.rows = rows
        self.length = len(rows)

    @staticmethod
    def _normalize(messages):
        # saiga_scored uses "bot" for the assistant turn (verified: {"user", "system", "bot"}
        # are the only roles present in the Russian rows) -- nanochat/tokenizer.py's renderer
        # requires the literal "assistant" string.
        return [{**m, "role": "assistant"} if m["role"] == "bot" else m for m in messages]

    @staticmethod
    def _is_valid(messages):
        if not messages:
            return False
        rest = messages[1:] if messages[0]["role"] == "system" else messages
        if len(rest) < 2:
            return False
        for i, message in enumerate(rest):
            expected_role = "user" if i % 2 == 0 else "assistant"
            if message["role"] != expected_role or not isinstance(message["content"], str):
                return False
        return True

    def num_examples(self):
        return self.length

    def get_example(self, index):
        return {"messages": self.rows[index]}
