from inclination.config import load_config


def test_config_has_required_sections():
    cfg = load_config()
    assert cfg["seed"] == 42
    for section in ("paths", "data", "model", "turning", "status", "insights", "eval", "llm"):
        assert section in cfg


def test_split_fractions_sum_to_one():
    fractions = load_config()["data"]["split_fractions"]
    assert abs(sum(fractions.values()) - 1.0) < 1e-9
