"""Gemini features, tested without any network call (a stub stands in for the client)."""
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from inclination import llm, llm_jobs
from inclination.config import ROOT, path
from inclination.cues import llm_tagger
from inclination.cues.taxonomy import ALL_NAMES, DESCRIPTIONS
from inclination.scoring import llm_rubric

ROLES = ["agent", "customer", "agent", "customer"]
TEXTS = ["The bike is $200.", "Too expensive.", "I can do $170.", "Deal, I'll take it."]


class StubClient:
    """Returns canned (parsed, raw) answers and records the prompts it was given."""

    def __init__(self, answers):
        self.answers, self.prompts = list(answers), []

    def generate(self, prompt, model, schema, job):
        self.prompts.append(prompt)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return (None if answer is None else schema.model_validate(answer)), str(answer)


def test_every_taxonomy_name_has_a_description():
    assert set(DESCRIPTIONS) == set(ALL_NAMES)


def test_tagger_drops_unknown_names_and_wrong_role_names():
    answer = {"turns": [
        {"turn": 0, "role": "agent", "cues": ["states_price", "price_objection", "made_up_cue"]},
        {"turn": 1, "role": "customer", "cues": ["price_objection", "states_price"]},
        {"turn": 3, "role": "customer", "cues": ["commitment", "agreement"]},
        {"turn": 9, "role": "customer", "cues": ["commitment"]},  # no such turn
    ]}
    cues, _ = llm_tagger.tag_conversation(StubClient([answer]), ROLES, TEXTS)
    assert cues == [["states_price"], ["price_objection"], [], ["commitment", "agreement"]]
    unusable, _ = llm_tagger.tag_conversation(StubClient([None]), ROLES, TEXTS)
    assert unusable is None


def test_tagger_prompt_lists_every_name_and_turn():
    prompt = llm_tagger.build_prompt(ROLES, TEXTS)
    assert all(name in prompt for name in ALL_NAMES)
    assert "[3] CUSTOMER: Deal, I'll take it." in prompt


def test_rubric_sees_only_the_turns_so_far():
    stub = StubClient([{"score": 50, "reason": "a"}, {"score": 20, "reason": "b"},
                       None, {"score": 95, "reason": "d"}])
    results = llm_rubric.score_conversation(stub, "craigslist", ROLES, TEXTS)
    assert [score for score, _ in results] == [0.5, 0.2, 0.2, 0.95]  # unusable answer: carried over
    for k, prompt in enumerate(stub.prompts):
        assert TEXTS[k] in prompt
        assert all(later not in prompt for later in TEXTS[k + 1:])


def test_rubric_rejects_scores_outside_the_scale():
    assert llm._parse('{"score": 140, "reason": "x"}', llm_rubric.RubricScore) is None
    assert llm._parse("not json", llm_rubric.RubricScore) is None
    assert llm._parse('{"score": 70, "reason": "x"}', llm_rubric.RubricScore).score == 70


def test_rationale_prompt_stops_at_the_trigger_turn():
    prompt = llm_jobs.rationale_prompt(ROLES, TEXTS, 1, "drop", -0.2, 30)
    assert "dropped by 20 points at turn 1" in prompt and "Too expensive." in prompt
    assert "I can do $170." not in prompt


def test_samples_are_deterministic_and_test_split_first(processed, cfg):
    _, conversations = processed
    scoring = llm_jobs.scoring_sample(conversations, cfg)
    assert scoring == llm_jobs.scoring_sample(conversations.sample(frac=1, random_state=5), cfg)
    split = conversations.set_index("conversation_id")["split"]
    for dataset, ids in scoring.items():
        assert len(ids) == len(set(ids)) == cfg["llm"]["score_sample_size"]
        assert all(split[cid] == "test" for cid in ids)
    tagging = llm_jobs.tagging_sample(conversations, cfg)
    for dataset, ids in tagging.items():
        assert len(ids) == len(set(ids)) == cfg["llm"]["tag_sample_size"]
        n_test = (conversations["dataset"].eq(dataset) & conversations["split"].eq("test")).sum()
        assert sum(split[cid] == "test" for cid in ids) == min(n_test, len(ids))


def test_a_stopped_job_keeps_what_it_has(tmp_path, monkeypatch, processed, cfg):
    turns, conversations = processed
    monkeypatch.setattr(llm_jobs, "path", lambda key: tmp_path)
    limited = {**cfg, "llm": {**cfg["llm"], "score_sample_size": 2}}
    first = llm_jobs.scoring_sample(conversations, limited)["craigslist"][0]
    n_turns = int((turns["conversation_id"] == first).sum())
    answers = [{"score": 60, "reason": "r"}] * n_turns + [llm.LLMStopped("quota")]
    llm_jobs.run_scoring(StubClient(answers), turns, conversations, limited)
    saved = pd.read_parquet(tmp_path / llm_jobs.SCORES_FILE)
    assert list(saved["conversation_id"].unique()) == [first] and len(saved) == n_turns


def test_dashboard_runs_with_the_v2_model_when_it_exists():
    leads_file = path("outputs") / "leads.parquet"
    if not leads_file.exists() or "v2" not in set(pd.read_parquet(leads_file, columns=["model"])["model"]):
        pytest.skip("no v2 scores exported")
    for page in ("Home.py", "pages/1_Lead_List.py", "pages/2_Conversation_View.py", "pages/3_Script_Insights.py"):
        app = AppTest.from_file(str(ROOT / "app" / page), default_timeout=120)
        app.session_state["dataset"] = "p4g"
        app.session_state["model"] = "v2"
        app.run()
        assert not app.exception, (page, [e.value for e in app.exception])
