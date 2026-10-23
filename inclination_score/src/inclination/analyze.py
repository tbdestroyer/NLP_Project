"""Scores one new conversation the way `make export` scores the datasets: cues, a score after
every turn, turning points with explanations, and a lead status. Used by POST /api/analyze.

Nothing is written to disk. The scorer is one of the per-dataset models in outputs/models/, so
the caller picks the dataset whose model fits the conversation best.
"""
import json

import numpy as np
import pandas as pd

from inclination import export
from inclination.cues import rules
from inclination.load.standardise import clean_text
from inclination.turning_points import delta_turning_points, pelt_turning_points

MODELS = ("v1", "v3")  # v2 needs a paid LLM call per turn, so it cannot score new chats
METHODS = ("delta", "pelt")
ROLES = ("agent", "customer")
CONVERSATION_ID = "uploaded"


class InvalidConversation(ValueError):
    """The conversation cannot be scored as given (the message says why)."""


class ModelUnavailable(LookupError):
    """The requested model has not been trained for the requested dataset."""


def load_backend(model: str, dataset: str, cfg: dict):
    """The export backend that scores with `model` trained on `dataset`."""
    if dataset not in cfg["data"]["datasets"]:
        raise ModelUnavailable(f"Unknown dataset {dataset!r}.")
    if model not in MODELS:
        raise ModelUnavailable(f"Model {model!r} cannot score new conversations; use one of {', '.join(MODELS)}.")
    if not export.BACKENDS[model].available(dataset):
        command = "make train" if model == "v1" else "make train-v3"
        raise ModelUnavailable(f"No trained {model} model for {dataset}. Run `{command}` in inclination_score/.")
    return export.BACKENDS[model](dataset, cfg)


def to_frame(turns: list[dict], dataset: str, limits: dict) -> pd.DataFrame:
    """One conversation as the turn table export uses: one row per non-empty message."""
    rows = []
    for raw in turns:
        role = str(raw.get("role", "")).lower()
        if role not in ROLES:
            raise InvalidConversation(f"Every message needs a role of 'agent' or 'customer', got {role!r}.")
        text = clean_text(raw.get("text"))
        if not text:
            continue
        if len(text) > limits["max_chars_per_turn"]:
            raise InvalidConversation(
                f"Message {len(rows)} has {len(text):,} characters; the limit is {limits['max_chars_per_turn']:,}.")
        rows.append({"role": role, "text": text, "speaker": clean_text(raw.get("speaker")) or role})
    if len(rows) < limits["min_turns"]:
        raise InvalidConversation(f"A conversation needs at least {limits['min_turns']} non-empty messages.")
    if len(rows) > limits["max_turns"]:
        raise InvalidConversation(f"The conversation has {len(rows)} messages; the limit is {limits['max_turns']}.")
    if {row["role"] for row in rows} != set(ROLES):
        raise InvalidConversation("Mark at least one speaker as the customer and one as the agent.")

    part = pd.DataFrame(rows)
    n = len(part)
    part.insert(0, "dataset", dataset)
    part.insert(1, "conversation_id", CONVERSATION_ID)
    part.insert(2, "turn", np.arange(n))
    part["group"] = 0
    part["position"] = np.arange(n)
    part["size"] = n
    return part


def _records(frame: pd.DataFrame) -> list[dict]:
    """pandas turns numpy types and NaN (as null) into plain JSON values."""
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def analyze(turns: list[dict], dataset: str, backend, cfg: dict, title: str = "",
            method: str = "delta") -> dict:
    """Scores `turns` ([{role, text, speaker?}, ...]) with `backend` (see load_backend).

    Returns the same fields as GET /api/conversations/{id}, without an outcome.
    """
    if method not in METHODS:
        raise InvalidConversation(f"Unknown turning-point method {method!r}.")
    part = to_frame(turns, dataset, cfg["analyze"])
    tags = rules.tag_turns(part["role"], part["text"])
    cue_matrix = rules.tags_to_matrix(tags)
    part["score"] = np.asarray(backend.score(part, cue_matrix), dtype=float)
    part["model"] = backend.name

    # Method A points are always needed: the lead status uses them, as in leads.parquet.
    turning = cfg["turning"]
    delta_points = export.explain_points(
        delta_turning_points(part, turning["delta_threshold"]), part, tags, backend, cfg,
        with_faithfulness=backend.supports_removal and method == "delta",
    )
    if method == "delta":
        points = delta_points
    else:
        points = pelt_turning_points(part, turning["pelt_penalty"], turning["pelt_min_size"])
        points = export.explain_points(points, part, tags, backend, cfg, with_faithfulness=False)
    points["rationale_llm"] = None

    last_point = next(delta_points.iloc[::-1].itertuples(index=False), None)
    scores = part["score"].to_numpy()
    status_cfg = cfg["status"]
    trend = export.trend_slope(scores, status_cfg["trend_window"])
    status = export.lead_status(scores[-1], trend, status_cfg)

    transcript = part[["turn", "role", "speaker", "text"]].assign(cues=[list(t) for t in tags])
    return {
        "conversation_id": CONVERSATION_ID,
        "title": title.strip() or "Uploaded conversation",
        "dataset": dataset,
        "model": backend.name,
        "outcome": None,
        "n_turns": len(part),
        "final_score": float(scores[-1]),
        "max_score": float(scores.max()),
        "trend": trend,
        "status": status,
        "status_reason": export.status_reason(status, last_point),
        "n_turning_points": len(points),
        "turns": _records(transcript),
        "scores": scores.tolist(),
        "scores_by_model": {backend.name: scores.tolist()},
        "turning_points": _records(points),
        "available_methods": list(METHODS),
        "available_taggers": ["rules"],
        "method": method,
        "tagger": "rules",
    }
