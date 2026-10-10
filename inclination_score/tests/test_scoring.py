"""Scoring tests, including the leakage CHECK: a score at turn k ignores turns after k."""
import joblib
import numpy as np
import pytest

from inclination.cues import rules
from inclination.cues.taxonomy import ALL_NAMES, AGENT_MOVE_NAMES, CUSTOMER_CUE_NAMES
from inclination.scoring.cue_model import CueModel, last_operators, prefix_operators, prefix_weights
from inclination.train import model_path

BUY = [("agent", "Hello, it is in great condition and costs $50."),
       ("customer", "Sounds good, how do I pay?"),
       ("agent", "Cash works. Would you like to buy it?"),
       ("customer", "Yes, I'll take it. Deal.")]
PASS = [("agent", "Hello, the price is $50."),
        ("customer", "That is too expensive, I am not sure."),
        ("agent", "I understand, but it saves you money."),
        ("customer", "No thanks, not interested.")]


@pytest.fixture(scope="module")
def toy_model(cfg):
    """A tiny v1 model trained on 40 toy conversations (30 train, 10 val)."""
    roles, texts, groups, labels, splits = [], [], [], [], []
    for i in range(40):
        conversation, label = (BUY, 1) if i % 2 == 0 else (PASS, 0)
        for role, text in conversation:
            roles.append(role)
            texts.append(f"{text} ref{i % 5}")
            groups.append(i)
            labels.append(label)
            splits.append("train" if i < 30 else "val")
    model = CueModel(cfg)
    matrix = rules.tags_to_matrix(rules.tag_turns(roles, texts))
    model.train(roles, texts, matrix, np.array(groups), np.array(labels), np.array(splits))
    return model


def _assert_causal(model, conversation):
    original = model.score(conversation)
    for k in range(len(conversation) - 1):
        altered = conversation[: k + 1] + [
            ("customer", "No thanks, not interested, too expensive."),
            ("agent", "Here is a 50% discount, deal?"),
            ("customer", "I'll take it! Sold. Great."),
        ]
        assert model.score(altered)[: k + 1] == pytest.approx(original[: k + 1], abs=1e-9)
        truncated = model.score(conversation[: k + 1])
        assert truncated == pytest.approx(original[: k + 1], abs=1e-9)


def test_score_at_turn_k_ignores_later_turns(toy_model):
    _assert_causal(toy_model, BUY)
    _assert_causal(toy_model, PASS)


@pytest.mark.parametrize("dataset", ["craigslist", "p4g", "saas"])
def test_trained_models_ignore_later_turns(dataset):
    target = model_path("v1", dataset)
    if not target.exists():
        pytest.skip("no trained model; run `make train` first")
    model = joblib.load(target)
    _assert_causal(model, BUY + PASS)


@pytest.mark.parametrize("dataset", ["craigslist", "p4g", "saas"])
def test_trained_transformers_ignore_later_turns(dataset, cfg):
    from inclination.export import V3Backend

    if not V3Backend.available(dataset):
        pytest.skip("no trained v3 model; run `make train-v3` first")
    from inclination.scoring.transformer import TransformerScorer

    scorer = TransformerScorer.load(dataset, cfg)
    conversation = BUY + PASS
    original = scorer.score(conversation)
    assert len(original) == len(conversation) and all(0 <= s <= 1 for s in original)
    for k in (0, 3, 6):
        altered = conversation[: k + 1] + [("customer", "No thanks, not interested."),
                                           ("agent", "Here is a 50% discount, deal?")]
        # batches are padded differently, so allow floating-point noise only
        assert scorer.score(altered)[: k + 1] == pytest.approx(original[: k + 1], abs=1e-4)


def test_transformer_packing_keeps_the_most_recent_turns():
    from inclination.scoring.transformer import TransformerScorer, sample_prefix_ends, token_batches

    class FakeTokenizer:
        cls_token_id, sep_token_id, pad_token_id = 0, 2, 1

    scorer = TransformerScorer.__new__(TransformerScorer)
    scorer.tokenizer, scorer.max_length = FakeTokenizer(), 8
    assert scorer.pack([[10, 11], [20, 21, 22], [30, 31]]) == [0, 20, 21, 22, 30, 31, 2]
    assert scorer.pack([[10], list(range(40, 50))]) == [0, 44, 45, 46, 47, 48, 49, 2]
    assert sample_prefix_ends(5, 8).tolist() == [0, 1, 2, 3, 4]
    ends = sample_prefix_ends(30, 8)
    assert len(ends) == 8 and ends[0] == 0 and ends[-1] == 29
    batches = token_batches([0, 1, 2, 3], [100, 100, 400, 50], max_items=3, max_tokens=400)
    assert batches == [[0, 1], [2], [3]]


def test_scores_are_probabilities_one_per_turn(toy_model):
    scores = toy_model.score(BUY)
    assert len(scores) == len(BUY)
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert toy_model.score(BUY)[-1] > toy_model.score(PASS)[-1]


def test_last_operators_match_the_final_prefix_row():
    groups = np.array([0, 0, 0, 0, 0, 1, 1, 2])
    values = np.arange(len(groups), dtype=np.float32)
    cumulative, recent, position = prefix_operators(groups, window=3)
    total, tail, last_position = last_operators(groups, window=3)
    last_rows = [4, 6, 7]
    assert (total @ values).tolist() == (cumulative @ values)[last_rows].tolist() == [10, 11, 7]
    assert (tail @ values).tolist() == (recent @ values)[last_rows].tolist() == [9, 11, 7]
    assert last_position.tolist() == position[last_rows].tolist()


def test_prefix_weights_sum_to_one_per_conversation():
    weights = prefix_weights(np.array([0, 0, 0, 0, 1, 2, 2]))
    assert weights.tolist() == [0.25] * 4 + [1.0] + [0.5] * 2


def test_rule_tagger_basics():
    assert "price_objection" in rules.tag_turn("customer", "That's too expensive for me")
    assert "commitment" in rules.tag_turn("customer", "Ok, I'll take it")
    assert "explicit_refusal" in rules.tag_turn("customer", "No thanks, not interested")
    assert rules.tag_turn("customer", "hmm") == ["disengagement"]
    assert "states_price" in rules.tag_turn("agent", "It is $40")
    assert "asks_question" in rules.tag_turn("agent", "What do you need?")


def test_rule_tagger_only_uses_taxonomy_names_for_the_right_role():
    samples = ["Great, how much is it? Can you go lower?", "I'll think about it", "ok", ""]
    for text in samples:
        assert set(rules.tag_turn("customer", text)) <= set(CUSTOMER_CUE_NAMES)
        assert set(rules.tag_turn("agent", text)) <= set(AGENT_MOVE_NAMES)
    assert rules.tags_to_matrix([["commitment"], []]).shape == (2, len(ALL_NAMES))
