"""Checks on turning points, explanations, insights and the exported dashboard files."""
import numpy as np
import pandas as pd
import pytest

from inclination import explain, insights
from inclination.config import path
from inclination.export import lead_status, trend_slope
from inclination.turning_points import delta_turning_points

EXPORTED = {
    "scores.parquet": ["dataset", "conversation_id", "turn", "model", "score"],
    "cues.parquet": ["dataset", "conversation_id", "turn", "role", "cues", "tagger"],
    "turning_points.parquet": ["dataset", "conversation_id", "turn", "model", "method", "direction",
                               "delta", "cues", "rationale", "faithfulness", "weak"],
    "leads.parquet": ["final_score", "max_score", "trend", "status", "n_turning_points", "outcome",
                      "split", "title", "is_synthetic"],
    "insights.parquet": ["mean_delta", "ci_low", "ci_high", "count", "share_rise", "share_drop"],
}


def _scores(values_by_conversation):
    rows = [
        {"dataset": "d", "conversation_id": cid, "turn": k, "model": "v1", "score": s}
        for cid, values in values_by_conversation.items() for k, s in enumerate(values)
    ]
    return pd.DataFrame(rows)


def test_delta_turning_points_never_cross_conversations():
    scores = _scores({"a": [0.5, 0.7, 0.72, 0.4], "b": [0.9, 0.85]})
    points = delta_turning_points(scores, threshold=0.15)
    found = [(p.conversation_id, p.turn, p.direction) for p in points.itertuples()]
    assert found == [("a", 1, "rise"), ("a", 3, "drop")]  # b turn 0 (0.4 -> 0.9) is not a point
    assert points["delta"].round(2).tolist() == [0.2, -0.32]


def test_template_rationale():
    sentence, trigger = explain.template_rationale(
        "drop", "customer", ["positive_sentiment", "price_objection"], ["states_price"]
    )
    assert sentence == "Interest dropped after a price objection, following the agent stating the price."
    assert trigger == "a price objection"
    assert explain.template_rationale("rise", "customer", [], [])[0] == (
        "Interest rose after the customer's reply."
    )
    assert "the agent offering a discount" in explain.template_rationale(
        "rise", "agent", ["gives_discount"], []
    )[0]


def test_removal_variants_drop_only_the_trigger():
    # one conversation of 5 turns at rows 10..14
    position = np.r_[np.zeros(10, int), np.arange(5)]
    sizes = np.r_[np.ones(10, int), np.full(5, 5)]
    rows, groups, evaluation = explain.removal_variants(np.array([12, 14]), None, sizes, position)
    assert rows.tolist() == [10, 11, 13] + [10, 11, 12, 13]
    assert groups.tolist() == [0, 0, 0, 1, 1, 1, 1]
    assert evaluation.tolist() == [13, 14]


def test_next_customer_rows_stay_inside_the_conversation():
    roles = np.array(["agent", "customer", "agent", "agent", "customer", "agent"])
    position = np.array([0, 1, 2, 0, 1, 2])
    assert insights.next_customer_rows(roles, position).tolist() == [1, -1, -1, 4, -1, -1]


def test_status_rules(cfg):
    status = cfg["status"]
    assert lead_status(0.8, -0.2, status) == "Hot"
    assert lead_status(0.5, 0.1, status) == "Rising"
    assert lead_status(0.5, -0.1, status) == "Cooling"
    assert lead_status(0.5, 0.0, status) == "Stalled"
    assert trend_slope(np.array([0.1, 0.2, 0.4, 0.6]), 3) == pytest.approx(0.2)
    assert trend_slope(np.array([0.3]), 3) == 0.0


@pytest.fixture(scope="module")
def exported():
    out = path("outputs")
    if not (out / "leads.parquet").exists():
        pytest.skip("outputs/ is empty; run `make export` first")
    return {name: pd.read_parquet(out / name) for name in EXPORTED}


def test_exported_files_have_the_spec_columns(exported):
    for name, columns in EXPORTED.items():
        assert set(columns) <= set(exported[name].columns), name


def test_one_calibrated_score_per_turn(exported, processed):
    turns, _ = processed
    scores = exported["scores.parquet"]
    assert scores["score"].between(0, 1).all()
    known = set(zip(turns["conversation_id"], turns["turn"]))
    for model, per_model in scores.groupby("model"):
        assert not per_model.duplicated(["conversation_id", "turn"]).any()
        assert set(zip(per_model["conversation_id"], per_model["turn"])) <= known
        if model == "v1":  # the baseline scores every conversation; v3 only some splits
            assert len(per_model) == len(turns)


def test_turning_points_match_the_threshold(exported, cfg):
    points = exported["turning_points.parquet"]
    delta_points = points[points["method"] == "delta"]
    assert (delta_points["delta"].abs() >= cfg["turning"]["delta_threshold"] - 1e-9).all()
    assert (delta_points["turn"] > 0).all()
    assert ((delta_points["delta"] > 0) == (delta_points["direction"] == "rise")).all()
    assert delta_points["rationale"].str.startswith("Interest ").all()


def test_insights_only_report_frequent_moves(exported, cfg):
    effects = exported["insights.parquet"]
    assert (effects["count"] >= cfg["insights"]["min_count"]).all()
    assert (effects["ci_low"] <= effects["mean_delta"]).all()
    assert (effects["mean_delta"] <= effects["ci_high"]).all()
