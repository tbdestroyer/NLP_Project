"""Annotation sample and label files."""
import json

import pytest
from streamlit.testing.v1 import AppTest

from inclination import annotation
from inclination.config import ROOT, path


def test_sample_is_test_split_only_even_and_deterministic(processed, cfg):
    _, conversations = processed
    sample = annotation.build_sample(conversations, cfg)
    assert sample == annotation.build_sample(conversations.sample(frac=1, random_state=3), cfg)
    per_dataset = cfg["annotation"]["n_conversations"] // len(cfg["data"]["datasets"])
    split = conversations.set_index("conversation_id")["split"]
    for dataset, ids in sample.items():
        assert len(ids) == len(set(ids)) == per_dataset
        assert all(cid.startswith(dataset) and split[cid] == "test" for cid in ids)


def test_saved_sample_matches_the_seeded_sample(processed, cfg):
    if not annotation.sample_path().exists():
        pytest.skip("no annotation sample; run `make export` first")
    _, conversations = processed
    saved = json.loads(annotation.sample_path().read_text(encoding="utf-8"))
    assert saved == annotation.build_sample(conversations, cfg)


def test_labels_round_trip_and_latest_save_wins(tmp_path):
    first = [{"turn": 1, "role": "customer", "direction": "more", "turning_point": False}]
    second = [{"turn": 1, "role": "customer", "direction": "less", "turning_point": True}]
    annotation.save_labels("Ann Lee!", "p4g-x", "p4g", first, tmp_path)
    target = annotation.save_labels("ann lee", "p4g-x", "p4g", second, tmp_path)
    annotation.save_labels("bob", "p4g-x", "p4g", first, tmp_path)
    assert target.name == "ann_lee.jsonl"
    assert len(target.read_text(encoding="utf-8").splitlines()) == 2
    assert annotation.load_labels("Ann Lee", tmp_path)["p4g-x"]["labels"] == second
    assert annotation.list_annotators(tmp_path) == ["ann_lee", "bob"]
    assert annotation.load_labels("nobody", tmp_path) == {}


def test_annotate_page_hides_scores_and_requires_a_name():
    if not (path("outputs") / "turns.parquet").exists() or not annotation.sample_path().exists():
        pytest.skip("run `make export` first")
    page = str(ROOT / "app" / "pages" / "9_Annotate.py")
    anonymous = AppTest.from_file(page, default_timeout=120).run()
    assert not anonymous.exception
    assert any("annotator name" in m.value for m in anonymous.info)

    named = AppTest.from_file(page, default_timeout=120)
    named.session_state["annotator"] = "pytest_probe"
    named.run()
    assert not named.exception
    assert len(named.radio) > 0 and len(named.checkbox) >= len(named.radio)
    assert all(radio.value is None for radio in named.radio)  # nothing pre-selected
    assert not named.metric  # no scores anywhere on the page
    assert not (annotation.labels_dir() / "pytest_probe.jsonl").exists()  # nothing saved on load
