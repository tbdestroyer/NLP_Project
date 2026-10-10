"""Method B turning points and the Phase 7 evaluation helpers."""
import json

import pandas as pd
import pytest

import evaluate
from inclination.config import path
from inclination.turning_points import pelt_turning_points


def _scores(values_by_conversation):
    rows = [
        {"dataset": "d", "conversation_id": cid, "turn": k, "model": "v1", "score": s}
        for cid, values in values_by_conversation.items() for k, s in enumerate(values)
    ]
    return pd.DataFrame(rows)


def test_pelt_finds_a_sustained_shift_and_ignores_flat_curves(cfg):
    turning = cfg["turning"]
    scores = _scores({
        "step": [0.3, 0.3, 0.32, 0.3, 0.7, 0.72, 0.7, 0.7],
        "flat": [0.5, 0.52, 0.5, 0.51, 0.5, 0.5],
        "high": [0.9, 0.9, 0.9, 0.9],  # a jump from "flat" to "high" must not count
    })
    points = pelt_turning_points(scores, turning["pelt_penalty"], turning["pelt_min_size"])
    assert [(p.conversation_id, p.turn, p.direction) for p in points.itertuples()] == [("step", 4, "rise")]
    assert points["delta"].iloc[0] == pytest.approx(0.4, abs=0.01)
    assert (points["method"] == "pelt").all()


def test_turning_point_matching_uses_the_tolerance():
    labelled = pd.DataFrame({"conversation_id": ["a", "a", "b"], "turn": [3, 9, 2]})
    predicted = pd.DataFrame({"conversation_id": ["a", "a", "c"], "turn": [4, 6, 2]})
    result = evaluate.match_turning_points(labelled, predicted, tolerance=1)
    assert result["n_predicted"] == 3 and result["n_labelled"] == 3
    assert result["precision"] == pytest.approx(1 / 3)  # only a:4 is near a labelled point
    assert result["recall"] == pytest.approx(1 / 3)     # only a:3 is found
    empty = evaluate.match_turning_points(labelled, predicted.iloc[:0], tolerance=1)
    assert empty["precision"] is None and empty["recall"] == 0


def test_direction_agreement():
    scores = _scores({"a": [0.5, 0.6, 0.6, 0.4, 0.41]})
    labels = pd.DataFrame({
        "conversation_id": "a", "dataset": "d", "turn": [0, 1, 2, 3, 4],
        "role": ["customer", "customer", "agent", "customer", "customer"],
        "direction": ["more", "more", None, "less", "more"], "turning_point": False,
    })
    overall = evaluate.evaluate_directions(labels, scores, epsilon=0.02)[0]
    # turn 0 is skipped; turn 1 up (correct), turn 3 down (correct), turn 4 same (label: more)
    assert overall["dataset"] == "all" and overall["n_turns"] == 3
    assert overall["accuracy"] == pytest.approx(2 / 3)
    assert overall["always_same_accuracy"] == 0


def test_llm_annotators_are_never_called_human():
    assert evaluate.annotator_kind("auto_llm") == "LLM"
    assert evaluate.annotator_kind("shantanu") == "human"


def test_faithfulness_compares_trigger_with_random():
    points = pd.DataFrame({
        "dataset": "d", "model": "v1", "method": ["delta", "delta", "pelt"],
        "faithfulness": [0.2, -0.4, None], "faithfulness_random": [0.1, 0.05, None],
        "weak": [False, False, False],
    })
    (row,) = evaluate.evaluate_faithfulness(points)
    assert row["n"] == 2
    assert row["trigger_removed"] == pytest.approx(0.3) and row["random_removed"] == pytest.approx(0.075)
    assert row["ratio"] == pytest.approx(4.0) and row["trigger_larger_share"] == 1.0


def test_metrics_file_has_every_section_and_labels_llm_annotators():
    target = path("outputs") / "metrics.json"
    if not target.exists():
        pytest.skip("no metrics; run `make eval` first")
    metrics = json.loads(target.read_text(encoding="utf-8"))
    for section in ("outcome", "turning_points", "faithfulness", "cross_domain", "labels", "case_studies"):
        assert section in metrics
    for annotator in metrics["labels"]["annotators"]:
        assert annotator["kind"] == evaluate.annotator_kind(annotator["annotator"])
