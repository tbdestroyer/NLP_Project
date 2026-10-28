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


LEAD_FIELDS = {"conversation_id", "title", "final_score", "status", "status_reason", "n_turns",
               "n_turning_points", "curve", "outcome"}


def batch_of(*conversations):
    return [{"title": f"Chat {i}", "turns": as_turns(c)} for i, c in enumerate(conversations)]


def test_batch_scores_match_scoring_each_conversation_alone(backend, cfg):
    conversations = [BUY, PASS, PASS + BUY[1:], BUY + PASS[1:]]
    result = analyze.analyze_batch(batch_of(*conversations), "saas", backend, cfg)
    assert [lead["title"] for lead in result["leads"]] == [f"Chat {i}" for i in range(4)]
    for lead, conversation in zip(result["leads"], conversations):
        alone = analyze.analyze(as_turns(conversation), "saas", backend, cfg)
        detail = result["conversations"][lead["conversation_id"]]
        assert lead["curve"] == detail["scores"] == pytest.approx(alone["scores"], abs=1e-9)
        for field in ("final_score", "status", "status_reason", "n_turning_points"):
            assert lead[field] == detail[field] == alone[field]
        assert [p["turn"] for p in detail["turning_points"]] == [p["turn"] for p in alone["turning_points"]]


@pytest.mark.parametrize("method", analyze.METHODS)
def test_batch_has_every_field_the_ui_reads(backend, cfg, method):
    result = analyze.analyze_batch(batch_of(BUY, PASS + BUY[1:]), "saas", backend, cfg, method=method)
    json.dumps(result)  # plain JSON values only
    assert result["n_conversations"] == len(result["leads"]) == len(result["conversations"]) == 2
    for lead in result["leads"]:
        assert LEAD_FIELDS <= lead.keys()
        assert UI_FIELDS <= result["conversations"][lead["conversation_id"]].keys()
        assert result["conversations"][lead["conversation_id"]]["method"] == method


def test_batch_skips_conversations_it_cannot_score(backend, cfg):
    conversations = batch_of(BUY, PASS)
    conversations.insert(1, {"title": "Only the rep", "turns": [{"role": "agent", "text": "Hi"}] * 3})
    result = analyze.analyze_batch(conversations, "saas", backend, cfg)
    assert [lead["title"] for lead in result["leads"]] == ["Chat 0", "Chat 1"]
    assert result["skipped"] == [{"index": 1, "title": "Only the rep",
                                  "reason": "Mark at least one speaker as the customer and one as the agent."}]


def test_batch_with_nothing_scorable_is_rejected(backend, cfg):
    with pytest.raises(analyze.InvalidConversation, match="None of the conversations"):
        analyze.analyze_batch([{"turns": [{"role": "agent", "text": "Hi"}]}], "saas", backend, cfg)
    with pytest.raises(analyze.InvalidConversation, match="at least one conversation"):
        analyze.analyze_batch([], "saas", backend, cfg)


def test_batch_size_is_limited(backend, cfg):
    limit = cfg["analyze"]["max_conversations"]
    with pytest.raises(analyze.InvalidConversation, match=f"limit is {limit}"):
        analyze.analyze_batch(batch_of(*[BUY] * (limit + 1)), "saas", backend, cfg)


def test_batch_script_insights_and_cue_summary(backend, cfg):
    conversations = [BUY, PASS] * 5
    result = analyze.analyze_batch(batch_of(*conversations), "saas", backend, cfg)
    assert result["insights"], "10 conversations repeat each agent move often enough"
    for row in result["insights"]:
        assert row["count"] >= result["insights_min_count"]
        assert row["ci_low"] <= row["mean_delta"] <= row["ci_high"]
    example = result["insight_examples"][0]
    detail = result["conversations"][example["conversation_id"]]
    assert example["agent_text"] == detail["turns"][example["agent_turn"]]["text"]
    assert example["customer_text"] == detail["turns"][example["customer_turn"]]["text"]

    counts = {row["cue"]: row for row in result["cue_summary"]}
    assert counts["explicit_refusal"]["n_conversations"] == 5 and counts["explicit_refusal"]["kind"] == "customer"
    assert all(row["n_conversations"] <= 10 for row in result["cue_summary"])


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


@pytest.mark.parametrize("model", analyze.MODELS)
def test_batch_matches_exported_scores(cfg, model):
    """Exported saas test conversations, uploaded together as a batch, get their exported scores."""
    scores_file, turns_file = path("outputs") / "scores.parquet", path("outputs") / "turns.parquet"
    if not (scores_file.exists() and turns_file.exists() and export.BACKENDS[model].available("saas")):
        pytest.skip(f"needs `make all` (a trained {model} model and outputs/)")
    try:
        backend = analyze.load_backend(model, "saas", cfg)
    except ImportError:
        pytest.skip(f"{model} needs requirements-ml.txt")
    scores = pd.read_parquet(scores_file, filters=[("dataset", "==", "saas"), ("model", "==", model)])
    ids = scores["conversation_id"].drop_duplicates().iloc[:5].tolist()
    turns = pd.read_parquet(turns_file, filters=[("conversation_id", "in", ids)]).sort_values("turn")
    conversations = [{"title": cid, "turns": turns[turns["conversation_id"] == cid][["role", "text"]]
                      .to_dict("records")} for cid in ids]

    result = analyze.analyze_batch(conversations, "saas", backend, cfg)
    for lead in result["leads"]:
        expected = scores[scores["conversation_id"] == lead["title"]].sort_values("turn")["score"]
        assert lead["curve"] == pytest.approx(expected.tolist(), abs=1e-4)
