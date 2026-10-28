"""POST /api/analyze and GET /api/analyze/models, with the toy v1 model of test_scoring."""
import joblib
import pytest
from fastapi.testclient import TestClient

from api import server
from inclination import export
from test_scoring import BUY, PASS, toy_model  # noqa: F401  (toy_model is a fixture)

client = TestClient(server.app)


@pytest.fixture
def trained_saas(toy_model, tmp_path, monkeypatch):
    """Only saas has a 'trained' v1 model: the toy model."""
    target = tmp_path / "v1_saas.joblib"
    joblib.dump(toy_model, target)
    monkeypatch.setattr(export, "model_path", lambda model, dataset: target if dataset == "saas"
                        else tmp_path / "missing.joblib")
    # v3 checks outputs/models/ directly, so hide any real v3 models on this machine
    monkeypatch.setattr(export.V3Backend, "available", staticmethod(lambda dataset: False))
    server.scoring_backend.cache_clear()
    yield
    server.scoring_backend.cache_clear()


def body(conversation, **extra):
    return {"dataset": "saas", "model": "v1", "title": "Test chat",
            "turns": [{"role": role, "text": text, "speaker": role.title()} for role, text in conversation],
            **extra}


def test_lists_trained_models(trained_saas):
    response = client.get("/api/analyze/models")
    assert response.status_code == 200
    assert response.json()["saas"] == ["v1"]
    assert response.json()["craigslist"] == []


def test_scores_an_uploaded_conversation(trained_saas, toy_model):
    conversation = PASS + BUY[1:]
    response = client.post("/api/analyze", json=body(conversation))
    assert response.status_code == 200
    result = response.json()
    assert result["title"] == "Test chat" and result["dataset"] == "saas" and result["model"] == "v1"
    assert result["scores"] == pytest.approx(toy_model.score(conversation), abs=1e-9)
    assert result["turns"][0]["speaker"] == "Agent"
    assert result["status"] in ("Hot", "Rising", "Cooling", "Stalled")


def test_change_point_method(trained_saas):
    response = client.post("/api/analyze", json=body(PASS + BUY[1:], method="pelt"))
    assert response.status_code == 200
    assert response.json()["method"] == "pelt"


@pytest.mark.parametrize("extra, status, message", [
    ({"dataset": "craigslist"}, 400, "No trained v1 model for craigslist"),
    ({"model": "v2"}, 400, "cannot score new"),
    ({"turns": [{"role": "agent", "text": "Hi"}, {"role": "agent", "text": "Hello?"}]}, 422, "customer"),
])
def test_explains_what_went_wrong(trained_saas, extra, status, message):
    response = client.post("/api/analyze", json={**body(BUY), **extra})
    assert response.status_code == status
    assert message in response.json()["detail"]


def test_rejects_unknown_roles(trained_saas):
    response = client.post("/api/analyze", json=body([("agent", "Hi"), ("buyer", "Hello")]))
    assert response.status_code == 422


def batch_body(*conversations, **extra):
    return {"dataset": "saas", "model": "v1",
            "conversations": [{"title": f"Chat {i}", "turns": body(c)["turns"]} for i, c in enumerate(conversations)],
            **extra}


def test_scores_a_batch(trained_saas, toy_model):
    response = client.post("/api/analyze/batch", json=batch_body(BUY, PASS))
    assert response.status_code == 200
    result = response.json()
    assert [lead["title"] for lead in result["leads"]] == ["Chat 0", "Chat 1"]
    assert result["leads"][1]["curve"] == pytest.approx(toy_model.score(PASS), abs=1e-9)
    assert result["skipped"] == []


@pytest.mark.parametrize("extra, status, message", [
    ({"dataset": "craigslist"}, 400, "No trained v1 model for craigslist"),
    ({"conversations": []}, 422, "at least one conversation"),
    ({"conversations": [{"turns": [{"role": "agent", "text": "Hi"}]}]}, 422, "None of the conversations"),
])
def test_batch_explains_what_went_wrong(trained_saas, extra, status, message):
    response = client.post("/api/analyze/batch", json={**batch_body(BUY), **extra})
    assert response.status_code == status
    assert message in response.json()["detail"]
