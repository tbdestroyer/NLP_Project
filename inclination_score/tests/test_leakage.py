"""Leakage rule (spec rule 2) checked on the processed tables."""
import json

from inclination.load.standardise import CONVERSATION_COLUMNS, TURN_COLUMNS

# Raw fields that reveal the outcome; none may appear as a column or a metadata key.
FORBIDDEN = {
    "outcome", "intent", "price", "target", "bottomline", "dialogue_acts", "b6", "donation",
    "probability_trajectory", "customer_engagement", "sales_effectiveness", "scenario",
    "conversation_flow", "expected_outcome", "expected_conversion_probability",
}
ALLOWED_META_KEYS = {
    "category", "listing_price", "charity",
    "company_name", "product_name", "product_type", "communication_channel",
}


def test_turn_table_has_only_the_standard_columns(processed):
    turns, _ = processed
    assert list(turns.columns) == TURN_COLUMNS
    assert not FORBIDDEN & {c.lower() for c in turns.columns}


def test_outcome_lives_only_in_the_conversation_table(processed):
    _, conversations = processed
    assert list(conversations.columns) == CONVERSATION_COLUMNS
    assert set(conversations["outcome"].unique()) <= {0, 1}


def test_metadata_is_display_only(processed):
    _, conversations = processed
    keys = set()
    for meta in conversations["meta"].unique():
        keys |= set(json.loads(meta))
    assert keys <= ALLOWED_META_KEYS
    assert not FORBIDDEN & keys


def test_no_action_or_system_turns_survive(processed):
    turns, _ = processed
    assert set(turns["role"].unique()) == {"customer", "agent"}
    assert (turns["text"].str.strip().str.len() > 0).all()
    templated = turns["text"].str.fullmatch(r"<(offer|accept|reject|quit)>", case=False)
    assert not templated.any()
    assert not turns["text"].str.fullmatch(r"-+ .* -+").any()  # "--- a week later ---" markers


def test_turns_are_numbered_from_zero_without_gaps(processed):
    turns, conversations = processed
    per_conversation = turns.groupby("conversation_id")["turn"].agg(["min", "max", "count"])
    assert (per_conversation["min"] == 0).all()
    assert (per_conversation["max"] == per_conversation["count"] - 1).all()
    n_turns = conversations.set_index("conversation_id")["n_turns"]
    assert per_conversation["count"].sort_index().equals(n_turns.sort_index().rename("count"))


def test_conversation_ids_are_prefixed_and_unique(processed):
    _, conversations = processed
    assert conversations["conversation_id"].is_unique
    prefixed = [
        cid.startswith(f"{dataset}-")
        for cid, dataset in zip(conversations["conversation_id"], conversations["dataset"])
    ]
    assert all(prefixed)


def test_only_saas_is_synthetic(processed):
    _, conversations = processed
    synthetic = conversations.groupby("dataset")["is_synthetic"].all().to_dict()
    assert synthetic == {"craigslist": False, "p4g": False, "saas": True}
