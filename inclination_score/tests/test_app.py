"""Smoke tests: every dashboard page runs without an exception for every dataset."""
import pytest
from streamlit.testing.v1 import AppTest

from inclination.config import ROOT, load_config, path

APP = ROOT / "app"
PAGES = ["Home.py", "pages/1_Lead_List.py", "pages/2_Conversation_View.py", "pages/3_Script_Insights.py"]
TIMEOUT_S = 120


@pytest.fixture(scope="module", autouse=True)
def needs_outputs():
    if not (path("outputs") / "leads.parquet").exists():
        pytest.skip("outputs/ is empty; run `make export` first")


def _run(page: str, **session) -> AppTest:
    app = AppTest.from_file(str(APP / page), default_timeout=TIMEOUT_S)
    for key, value in session.items():
        app.session_state[key] = value
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    return app


@pytest.mark.parametrize("page", PAGES)
@pytest.mark.parametrize("dataset", load_config()["data"]["datasets"])
def test_page_runs(page, dataset):
    _run(page, dataset=dataset)


def test_outcomes_are_hidden_unless_evaluation_mode_is_on():
    hidden = _run("pages/1_Lead_List.py", dataset="p4g")
    shown = _run("pages/1_Lead_List.py", dataset="p4g", eval_mode=True)
    assert "Outcome" not in hidden.dataframe[0].value.columns
    assert "Outcome" in shown.dataframe[0].value.columns
    assert any("actually converted" in c.value for c in shown.caption)
    assert not any("actually converted" in c.value for c in hidden.caption)


def test_synthetic_badge_only_on_saas():
    def badges(dataset):
        app = _run("Home.py", dataset=dataset)
        return [m.value for m in app.markdown if "Synthetic data" in m.value]
    assert badges("saas") and not badges("p4g")


def test_replay_hides_later_turns():
    full = _run("pages/2_Conversation_View.py", dataset="p4g")
    conversation_id = full.session_state["conv_select"]
    replay = _run("pages/2_Conversation_View.py", dataset="p4g",
                  **{f"replay_{conversation_id}": 2})
    transcript = max((m.value for m in replay.markdown), key=len)
    assert "turn 2" in transcript and "turn 3" not in transcript
