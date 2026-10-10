"""Loader tests on tiny hand-made records in each dataset's raw format."""
import json

import pandas as pd

from inclination.load import craigslist, p4g, saas
from inclination.load.standardise import TURN_COLUMNS, clean_text, to_tables


def _craigslist_row(speakers, intents, utterances):
    return {
        "agent_info": {"Role": ["buyer", "seller"], "Target": [50.0, 80.0], "Bottomline": ["None"] * 2},
        "agent_turn": speakers,
        "dialogue_acts": {"intent": intents, "price": [-1.0] * len(intents)},
        "utterance": utterances,
        "items": {"Category": ["bike"] * 2, "Title": ["Red bike"] * 2, "Price": [80.0] * 2},
    }


def _craigslist_records(cfg, rows):
    return craigslist.standardise(
        {"train": pd.DataFrame(rows)}, cfg["data"]["sources"]["craigslist"]
    )


def test_craigslist_roles_outcome_and_action_turns(cfg):
    accepted = _craigslist_row(
        [0, 1, 0, 1], ["init-price", "agree", "offer", "accept"], ["60?", "ok", "", ""]
    )
    rejected = _craigslist_row(
        [1, 0, 0, 1, 0], ["intro", "offer", "unknown", "reject", "quit"], ["hi", "", "pls", "", ""]
    )
    first, second = _craigslist_records(cfg, [accepted, rejected])
    assert first["outcome"] == 1 and second["outcome"] == 0
    assert first["turns"] == [("customer", "60?"), ("agent", "ok")]
    # the text typed after a (rejected) offer is kept; the action turns are not
    assert second["turns"] == [("agent", "hi"), ("customer", "pls")]
    assert first["meta"] == {"category": "bike", "listing_price": 80.0}


def test_craigslist_drops_everything_after_a_terminal_act(cfg):
    row = _craigslist_row(
        [0, 1, 0, 1, 0], ["init-price", "offer", "accept", "unknown", "unknown"],
        ["60?", "", "", "thanks for buying", "see you"],
    )
    (record,) = _craigslist_records(cfg, [row])
    assert record["turns"] == [("customer", "60?")]


def test_craigslist_excludes_dialogues_without_acts(cfg):
    unlabelled = _craigslist_row([0, 1, 0], ["", "", ""], ["hi", "hello", ""])
    assert _craigslist_records(cfg, [unlabelled]) == []


def test_p4g_roles_and_actual_donation(cfg):
    dialog = pd.DataFrame({
        "Unit": ["Hello", "Hi", "Donate?", "Sure $1", "Hello", "No"],
        "Turn": [0, 0, 1, 1, 0, 0],
        "B4": [0, 1, 0, 1, 0, 1],
        "B2": ["a", "a", "a", "a", "b", "b"],
    })
    info = pd.DataFrame({
        "B2": ["a", "a", "b", "b"],
        "B4": [0, 1, 0, 1],
        "B6": [5.0, 1.0, 2.0, 0.0],  # persuader donations (5.0, 2.0) must be ignored
        "age.x": [30, 40, 50, 60],
    })
    first, second = p4g.standardise(dialog, info, cfg["data"]["sources"]["p4g"])
    assert (first["outcome"], second["outcome"]) == (1, 0)
    assert first["turns"][0] == ("agent", "Hello") and first["turns"][1] == ("customer", "Hi")
    assert first["conversation_id"] == "p4g-a"
    assert "B6" not in json.dumps(first["meta"]) and "age" not in json.dumps(first["meta"])


def _saas_frame(n=1):
    conversation = json.dumps([
        {"speaker": "customer", "message": "hi"},
        {"speaker": "system", "message": "--- a week later ---"},
        {"speaker": "sales_rep", "message": "hello"},
    ])
    return pd.DataFrame({
        "conversation_id": [f"saas-1-conv-{i}" for i in range(n)],
        "conversation": [conversation] * n,
        "outcome": [i % 2 for i in range(n)],
        "company_name": ["Acme"] * n,
        "product_name": ["Widget"] * n,
        "product_type": ["Logistics"] * n,
        "communication_channel": ["email"] * n,
        "probability_trajectory": ['{"0": 0.9}'] * n,
    })


def test_saas_drops_system_messages_and_annotations(cfg):
    (record,) = saas.standardise(_saas_frame(), cfg["data"]["sources"]["saas"])
    assert record["turns"] == [("customer", "hi"), ("agent", "hello")]
    assert record["is_synthetic"] is True
    assert record["conversation_id"] == "saas-1-conv-0"
    assert set(record["meta"]) == set(cfg["data"]["sources"]["saas"]["meta_columns"])


def test_saas_subsample_is_stratified_and_deterministic(cfg):
    frame = _saas_frame(200)
    first = saas.subsample(frame, 50, cfg["seed"])
    second = saas.subsample(frame.sample(frac=1, random_state=1), 50, cfg["seed"])
    assert len(first) == 50 and first["outcome"].mean() == 0.5
    assert list(first["conversation_id"]) == list(second["conversation_id"])


def test_to_tables_cleans_renumbers_and_drops_short_conversations():
    base = {"dataset": "p4g", "outcome": 1, "title": "t", "meta": {}, "is_synthetic": False}
    records = [
        {**base, "conversation_id": "p4g-x",
         "turns": [("agent", "  hi \n there "), ("customer", "   "), ("customer", "ok")]},
        {**base, "conversation_id": "p4g-y", "turns": [("agent", "only one")]},
    ]
    turns, conversations = to_tables(records, min_turns=2)
    assert list(turns.columns) == TURN_COLUMNS
    assert list(turns["text"]) == ["hi there", "ok"] and list(turns["turn"]) == [0, 1]
    assert list(conversations["conversation_id"]) == ["p4g-x"]
    assert conversations["n_turns"].iloc[0] == 2
    assert clean_text(float("nan")) == ""
