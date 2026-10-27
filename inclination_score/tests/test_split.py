"""Split rule (spec rule 3): by conversation, stratified, seeded, saved and reused."""
import json

import pandas as pd

from inclination.config import path
from inclination.load.split import load_or_make_splits, make_splits


def _conversations(n=400):
    return pd.DataFrame({
        "dataset": ["a"] * (n // 2) + ["b"] * (n // 2),
        "conversation_id": [f"c{i}" for i in range(n)],
        "outcome": [int(i % 4 == 0) for i in range(n)],
    })


def test_make_splits_fractions_and_stratification(cfg):
    conversations = _conversations()
    assignment = make_splits(conversations, cfg["data"]["split_fractions"], cfg["seed"])
    conversations["split"] = conversations["conversation_id"].map(assignment)
    assert conversations["split"].notna().all()
    for _, group in conversations.groupby("dataset"):
        shares = group["split"].value_counts(normalize=True)
        for name, fraction in cfg["data"]["split_fractions"].items():
            assert abs(shares[name] - fraction) < 0.01
        rates = group.groupby("split")["outcome"].mean()
        assert (rates - group["outcome"].mean()).abs().max() < 0.02


def test_make_splits_is_deterministic_and_order_independent(cfg):
    conversations = _conversations()
    fractions, seed = cfg["data"]["split_fractions"], cfg["seed"]
    shuffled = conversations.sample(frac=1, random_state=7)
    assert make_splits(conversations, fractions, seed) == make_splits(shuffled, fractions, seed)


def test_a_new_batch_does_not_move_existing_conversations(cfg):
    fractions, seed = cfg["data"]["split_fractions"], cfg["seed"]
    before = make_splits(_conversations(), fractions, seed)
    added = pd.DataFrame({
        "dataset": ["a"] * 100,
        "conversation_id": [f"new{i}" for i in range(100)],
        "outcome": [int(i % 2 == 0) for i in range(100)],
    })
    batches = dict.fromkeys(added["conversation_id"], "new")
    after = make_splits(pd.concat([_conversations(), added]), fractions, seed, batches)
    assert {cid: after[cid] for cid in before} == before
    new_splits = pd.Series([after[cid] for cid in added["conversation_id"]])
    for name, fraction in fractions.items():
        assert abs((new_splits == name).mean() - fraction) < 0.01


def test_saved_splits_are_reused(cfg, tmp_path):
    conversations = _conversations()
    fractions = cfg["data"]["split_fractions"]
    splits_path = tmp_path / "splits.json"
    first = load_or_make_splits(conversations, fractions, cfg["seed"], splits_path)
    # a different seed must not change anything once the file exists
    second = load_or_make_splits(conversations, fractions, cfg["seed"] + 1, splits_path)
    assert first == second == json.loads(splits_path.read_text(encoding="utf-8"))


def test_every_turn_of_a_conversation_is_in_one_split(processed):
    turns, conversations = processed
    assert conversations["split"].isin(["train", "val", "test"]).all()
    merged = turns.merge(conversations[["conversation_id", "split"]], on="conversation_id")
    assert len(merged) == len(turns)
    assert (merged.groupby("conversation_id")["split"].nunique() == 1).all()


def test_processed_splits_match_the_saved_file(processed):
    _, conversations = processed
    saved = json.loads(path("splits").read_text(encoding="utf-8"))
    assert conversations.set_index("conversation_id")["split"].to_dict() == saved


def test_processed_splits_are_stratified(processed, cfg):
    _, conversations = processed
    for _, group in conversations.groupby("dataset"):
        shares = group["split"].value_counts(normalize=True)
        for name, fraction in cfg["data"]["split_fractions"].items():
            assert abs(shares[name] - fraction) < 0.01
        rates = group.groupby("split")["outcome"].mean()
        assert (rates - group["outcome"].mean()).abs().max() < 0.01
