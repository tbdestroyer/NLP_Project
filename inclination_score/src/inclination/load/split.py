"""Conversation-level train/val/test splits, stratified by outcome per dataset (spec rule 3).

The assignment is saved to disk and reused, so every later stage sees the same split.
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def make_splits(
    conversations: pd.DataFrame, fractions: dict, seed: int, batches: dict | None = None
) -> dict[str, str]:
    """Returns {conversation_id: split}. One row of `conversations` = one conversation.

    `batches` maps a conversation id to a batch name. Each batch of a dataset is split on its own,
    so adding a new batch never moves the conversations that were already assigned.
    """
    batch = conversations["conversation_id"].map(batches or {}).fillna("")
    assignment = {}
    for _, group in conversations.assign(batch=batch).groupby(["dataset", "batch"]):
        group = group.sort_values("conversation_id")
        train, rest = train_test_split(
            group, train_size=fractions["train"], stratify=group["outcome"], random_state=seed
        )
        val_share = fractions["val"] / (fractions["val"] + fractions["test"])
        val, test = train_test_split(
            rest, train_size=val_share, stratify=rest["outcome"], random_state=seed
        )
        for name, part in (("train", train), ("val", val), ("test", test)):
            assignment.update(dict.fromkeys(part["conversation_id"], name))
    return assignment


def load_or_make_splits(
    conversations: pd.DataFrame,
    fractions: dict,
    seed: int,
    splits_path: Path,
    batches: dict | None = None,
) -> dict[str, str]:
    """Reuses the saved assignment when it covers exactly these conversations."""
    if splits_path.exists():
        saved = json.loads(splits_path.read_text(encoding="utf-8"))
        if set(saved) == set(conversations["conversation_id"]):
            return saved
        print(f"  {splits_path.name} does not match the current conversations; rebuilding it")
    assignment = make_splits(conversations, fractions, seed, batches)
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(json.dumps(assignment, indent=0, sort_keys=True), encoding="utf-8")
    return assignment
