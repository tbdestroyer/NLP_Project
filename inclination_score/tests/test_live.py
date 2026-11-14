"""Live mode: transcript cleaning, the coaching playbook, coaching and the /api/live/* endpoints.
Whisper is never loaded: the endpoint tests use a fake transcriber."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import server
from inclination import live
from test_api import body, trained_saas  # noqa: F401  (trained_saas is a fixture)
from test_scoring import BUY, PASS, toy_model  # noqa: F401  (toy_model is a fixture)

client = TestClient(server.app)


# ---- transcripts ---------------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    (" Honestly, that's too expensive. ", "Honestly, that's too expensive."),
    ("[BLANK_AUDIO]", ""),
    ("(upbeat music) ", ""),
    ("Thanks for watching!", ""),
    ("Sure [inaudible] that works.", "Sure that works."),
    ("Thank you.", "Thank you."),  # can be real speech, so it is kept
])
def test_cleans_whisper_output(raw, expected):
    assert live.clean_transcript(raw) == expected


def test_decodes_float32_audio():
    samples = np.array([0.0, 0.5, -0.25], dtype="<f4")
    assert live.decode_pcm(samples.tobytes()).tolist() == [0.0, 0.5, -0.25]
    with pytest.raises(ValueError):
        live.decode_pcm(b"abc")
    with pytest.raises(ValueError):
        live.decode_pcm(np.array([np.nan], dtype="<f4").tobytes())


def test_quiet_audio_is_not_transcribed(cfg):
    engine = live.Transcriber(cfg)
    engine.load = lambda: pytest.fail("silence must not load the model")
    out = engine.transcribe(np.zeros(live.SAMPLE_RATE, dtype=np.float32))
    assert out["text"] == ""
    with pytest.raises(ValueError):
        engine.transcribe(np.zeros(live.SAMPLE_RATE * 31, dtype=np.float32))


# ---- the playbook --------------------------------------------------------------------------------

def scored_chats(n: int = 30) -> pd.DataFrame:
    """n chats in which, after a price objection, a discount is followed by a rise and stating
    the price again by a drop."""
    rows = []
    for i in range(n):
        discount = i % 2 == 0
        bump = 0.2 + 0.01 * (i % 5)
        turns = [
            ("agent", "It's $50.", ["states_price"], 0.5),
            ("customer", "Too expensive.", ["price_objection"], 0.4),
            ("agent", "I can do $40." if discount else "It's $50, final.",
             ["gives_discount"] if discount else ["states_price"], 0.4),
            ("customer", "Deal." if discount else "No.", ["agreement"] if discount else ["explicit_refusal"],
             0.4 + bump if discount else 0.4 - bump),
        ]
        for turn, (role, text, cues, score) in enumerate(turns):
            rows.append({"conversation_id": f"c{i}", "turn": turn, "role": role, "text": text,
                         "cues": np.array(cues), "score": score})
    return pd.DataFrame(rows)


@pytest.fixture
def small_cfg(cfg):
    return {**cfg, "live": {**cfg["live"], "min_count": 5, "shrinkage": 5}}


def test_playbook_measures_replies_per_situation(small_cfg):
    playbook = live.build_playbook(scored_chats(), small_cfg)
    effects = playbook["effects"].set_index(["situation", "move"])
    discount = effects.loc[("price_objection", "gives_discount")]
    assert discount["count"] == 15 and discount["mean_delta"] == pytest.approx(0.22, abs=0.01)
    assert effects.loc[("price_objection", "states_price"), "mean_delta"] == pytest.approx(-0.22, abs=0.01)
    # the first agent turn opens the call: its reply is the price objection, 0.1 lower
    assert effects.loc[(live.OPENING, "states_price"), "mean_delta"] == pytest.approx(-0.1)
    assert playbook["baseline"]["price_objection"]["count"] == 30
    assert playbook["n_conversations"] == 30
    example = playbook["examples"][("price_objection", "gives_discount")][0]
    assert example == {"customer": "Too expensive.", "agent": "I can do $40.", "reply": "Deal.",
                       "delta": pytest.approx(0.24)}
    assert playbook["examples"][("price_objection", "states_price")] == []  # no rises to show


def test_playbook_skips_rare_pairs(small_cfg):
    playbook = live.build_playbook(scored_chats(8), small_cfg)
    assert ("price_objection", "gives_discount") not in set(zip(playbook["effects"]["situation"], playbook["effects"]["move"]))


# ---- coaching ------------------------------------------------------------------------------------

def test_coach_answers_the_latest_customer_cue(small_cfg):
    playbook = live.build_playbook(scored_chats(), small_cfg)
    turns = [{"role": "agent", "text": "It's $50.", "cues": ["states_price"]},
             {"role": "customer", "text": "Too expensive, I'm not sure.", "cues": ["price_objection", "doubt_or_distrust"]}]
    out = live.coach(turns, [0.5, 0.4], "Cooling", playbook, small_cfg)
    assert out["situation"] == "price_objection"  # objections outrank doubt
    assert out["waiting_for"] == "agent" and out["quote"] == "Too expensive, I'm not sure."
    assert [s["move"] for s in out["suggestions"]] == ["gives_discount"]
    top = out["suggestions"][0]
    assert top["evidence"] == "strong" and top["examples"][0]["agent"] == "I can do $40."
    assert top["say"] == live.SITUATION_SAY[("price_objection", "gives_discount")]
    assert "after a price objection" in top["why"]
    assert [a["move"] for a in out["avoid"]] == ["states_price"]
    assert out["avoid"][0]["just_used"]


def test_coach_falls_back_to_every_reply(small_cfg):
    playbook = live.build_playbook(scored_chats(), small_cfg)
    turns = [{"role": "agent", "text": "Hi", "cues": []},
             {"role": "customer", "text": "Is this legit?", "cues": ["doubt_or_distrust"]}]
    out = live.coach(turns, [0.5, 0.4], "Stalled", playbook, small_cfg)
    assert out["situation"] == "doubt_or_distrust" and out["evidence_from"] == live.ANY


def test_coach_without_playbook_or_customer(cfg):
    out = live.coach([{"role": "agent", "text": "Hello!", "cues": ["builds_rapport"]}], None, None, None, cfg)
    assert out["situation"] == live.OPENING and out["waiting_for"] == "customer"
    assert out["suggestions"] == [] and out["headline"]


# ---- endpoints -----------------------------------------------------------------------------------

class FakeTranscriber:
    loaded = False

    def status(self):
        return {"installed": True, "loaded": self.loaded, "model": "fake", "device": "cpu", "error": ""}

    def load(self):
        self.loaded = True

    def transcribe(self, audio):
        return {"text": f"{len(audio)} samples", "duration_s": len(audio) / live.SAMPLE_RATE, "elapsed_s": 0.0}


@pytest.fixture
def fake_whisper(monkeypatch):
    fake = FakeTranscriber()
    monkeypatch.setattr(server, "transcriber", lambda: fake)
    return fake


def test_load_and_transcribe(fake_whisper):
    assert client.get("/api/live/status").json()["loaded"] is False
    assert client.post("/api/live/load").json()["loaded"] is True
    audio = np.zeros(1600, dtype="<f4").tobytes()
    response = client.post("/api/live/transcribe", content=audio,
                           headers={"Content-Type": "application/octet-stream"})
    assert response.status_code == 200 and response.json()["text"] == "1600 samples"
    assert client.post("/api/live/transcribe", content=b"abc").status_code == 422


def test_live_score_before_both_sides_speak(trained_saas, monkeypatch):
    monkeypatch.setattr(server, "live_playbook", lambda dataset, model: None)
    response = client.post("/api/live/score", json=body(BUY[:1]))
    assert response.status_code == 200
    out = response.json()
    assert out["ready"] is False and out["detail"] is None and out["message"]
    assert out["turns"][0]["cues"] and out["coach"]["situation"] == live.OPENING


def test_live_score_and_coach(trained_saas, toy_model, monkeypatch, small_cfg):
    playbook = live.build_playbook(scored_chats(), small_cfg)
    monkeypatch.setattr(server, "live_playbook", lambda dataset, model: playbook)
    response = client.post("/api/live/score", json=body(PASS[:2]))
    assert response.status_code == 200
    out = response.json()
    assert out["ready"] is True and out["dropped_turns"] == 0
    assert out["detail"]["scores"] == pytest.approx(toy_model.score(PASS[:2]), abs=1e-9)
    assert out["coach"]["situation"] == "price_objection"
    assert out["coach"]["suggestions"][0]["move"] == "gives_discount"


def test_live_score_keeps_the_latest_messages(trained_saas, monkeypatch, cfg):
    monkeypatch.setattr(server, "live_playbook", lambda dataset, model: None)
    too_long = (BUY * (cfg["analyze"]["max_turns"] // len(BUY) + 1))[: cfg["analyze"]["max_turns"] + 3]
    out = client.post("/api/live/score", json=body(too_long)).json()
    assert out["ready"] is True and out["dropped_turns"] == 3
    assert len(out["detail"]["turns"]) == cfg["analyze"]["max_turns"]


def test_live_score_needs_a_trained_model(trained_saas):
    response = client.post("/api/live/score", json=body(BUY, dataset="craigslist"))
    assert response.status_code == 400
