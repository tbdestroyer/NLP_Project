"""Scoring a new conversation (inclination.analyze), with the toy v1 model of test_scoring."""
import json

import joblib
import pandas as pd
import pytest

from inclination import analyze, export
from inclination.config import path
from test_scoring import BUY, PASS, toy_model  # noqa: F401  (toy_model is a fixture)

UI_FIELDS = {"conversation_id", "title", "dataset", "model", "outcome", "final_score", "status",
             "status_reason", "n_turning_points", "turns", "scores", "scores_by_model",
             "turning_points", "available_methods", "method"}
POINT_FIELDS = {"turn", "direction", "delta", "cues", "rationale", "trigger", "faithfulness",
                "weak", "role", "prev_agent_turn", "top_features", "rationale_llm"}


def as_turns(conversation):
    return [{"role": role, "text": text, "speaker": role.title()} for role, text in conversation]


@pytest.fixture
def backend(toy_model, tmp_path, monkeypatch, cfg):
    """The toy model loaded through load_backend, as the API loads a trained one."""
    target = tmp_path / "v1_saas.joblib"
    joblib.dump(toy_model, target)
    monkeypatch.setattr(export, "model_path", lambda model, dataset: target)
    return analyze.load_backend("v1", "saas", cfg)


def test_scores_match_the_trained_model(backend, toy_model, cfg):
    for conversation in (BUY, PASS):
        result = analyze.analyze(as_turns(conversation), "saas", backend, cfg)
        assert result["scores"] == pytest.approx(toy_model.score(conversation), abs=1e-9)


def test_score_at_turn_k_ignores_later_turns(backend, cfg):
    original = analyze.analyze(as_turns(BUY), "saas", backend, cfg)["scores"]
    altered = BUY[:2] + [("customer", "No thanks, not interested, too expensive.")]
    assert analyze.analyze(as_turns(altered), "saas", backend, cfg)["scores"][:2] == pytest.approx(original[:2])


@pytest.mark.parametrize("method", analyze.METHODS)
def test_result_has_every_field_the_ui_reads(backend, cfg, method):
    conversation = PASS + BUY[1:]  # low, then rises sharply
    result = analyze.analyze(as_turns(conversation), "saas", backend, cfg, title=" Acme ", method=method)
    json.dumps(result)  # plain JSON values only
    assert UI_FIELDS <= result.keys()
    assert result["title"] == "Acme" and result["outcome"] is None and result["method"] == method
    assert len(result["scores"]) == len(result["turns"]) == len(conversation)
    assert result["turns"][1] == {"turn": 1, "role": "customer", "speaker": "Customer",
                                  "text": PASS[1][1], "cues": result["turns"][1]["cues"]}
    assert result["n_turning_points"] == len(result["turning_points"])
    for point in result["turning_points"]:
        assert POINT_FIELDS <= point.keys()
        assert point["direction"] in ("rise", "drop")


def test_turning_points_have_explanations(backend, cfg):
    result = analyze.analyze(as_turns(PASS + BUY[1:]), "saas", backend, cfg)
    assert result["turning_points"], "the toy conversation should move sharply at least once"
    point = result["turning_points"][0]
    assert point["rationale"].startswith("Interest ")
    assert point["faithfulness"] is not None


def test_empty_messages_are_dropped(backend, cfg):
    turns = as_turns(BUY)
    turns.insert(1, {"role": "agent", "text": "   "})
    assert len(analyze.analyze(turns, "saas", backend, cfg)["turns"]) == len(BUY)


@pytest.mark.parametrize("turns, message", [
    ([{"role": "agent", "text": "Hello"}], "at least 2"),
    ([{"role": "agent", "text": "Hello"}, {"role": "agent", "text": "Anyone?"}], "customer"),
    ([{"role": "agent", "text": "Hello"}, {"role": "buyer", "text": "Hi"}], "role"),
    ([{"role": "agent", "text": "Hello"}, {"role": "customer", "text": "x" * 2001}], "characters"),
    ([{"role": "agent", "text": "Hi"}, {"role": "customer", "text": "Hi"}] * 101, "limit is 200"),
])
def test_invalid_conversations_are_rejected(backend, cfg, turns, message):
    with pytest.raises(analyze.InvalidConversation, match=message):
        analyze.analyze(turns, "saas", backend, cfg)


def test_unknown_method_is_rejected(backend, cfg):
    with pytest.raises(analyze.InvalidConversation, match="method"):
        analyze.analyze(as_turns(BUY), "saas", backend, cfg, method="magic")


@pytest.mark.parametrize("model, dataset, message", [
    ("v2", "saas", "cannot score new"),
    ("v1", "amazon", "Unknown dataset"),
])
def test_unusable_models_are_rejected(cfg, model, dataset, message):
    with pytest.raises(analyze.ModelUnavailable, match=message):
        analyze.load_backend(model, dataset, cfg)


def test_untrained_model_is_rejected(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(export, "model_path", lambda model, dataset: tmp_path / "missing.joblib")
    with pytest.raises(analyze.ModelUnavailable, match="make train"):
        analyze.load_backend("v1", "saas", cfg)


@pytest.mark.parametrize("dataset", ["craigslist", "p4g", "saas"])
def test_matches_exported_scores(cfg, dataset):
    """An exported test conversation, re-scored as an upload, gets its exported v1 scores."""
    scores_file, turns_file = path("outputs") / "scores.parquet", path("outputs") / "turns.parquet"
    if not (scores_file.exists() and turns_file.exists() and export.V1Backend.available(dataset)):
        pytest.skip("needs `make all` (trained v1 models and outputs/)")
    scores = pd.read_parquet(scores_file)
    scores = scores[(scores["dataset"] == dataset) & (scores["model"] == "v1")]
    conversation_id = scores["conversation_id"].iloc[0]
    turns = pd.read_parquet(turns_file)
    turns = turns[turns["conversation_id"] == conversation_id].sort_values("turn")
    expected = scores[scores["conversation_id"] == conversation_id].sort_values("turn")["score"]

    backend = analyze.load_backend("v1", dataset, cfg)
    result = analyze.analyze(turns[["role", "text"]].to_dict("records"), dataset, backend, cfg)
    assert result["scores"] == pytest.approx(expected.tolist(), abs=1e-6)
