"""Scores new conversations the way `make export` scores the datasets: cues, a score after every
turn, turning points with explanations, and a lead status. Used by POST /api/analyze (one chat)
and POST /api/analyze/batch (many chats: a lead list plus script insights).

Nothing is written to disk. The scorer is one of the per-dataset models in outputs/models/, so
the caller picks the dataset whose model fits the conversations best.
"""
import json

import numpy as np
import pandas as pd

from inclination import export, insights
from inclination.cues import rules
from inclination.cues.taxonomy import CUSTOMER_CUE_NAMES
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


def to_frame(turns: list[dict], dataset: str, limits: dict, conversation_id: str = CONVERSATION_ID,
             group: int = 0) -> pd.DataFrame:
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
    part.insert(1, "conversation_id", conversation_id)
    part.insert(2, "turn", np.arange(n))
    part["group"] = group
    part["position"] = np.arange(n)
    part["size"] = n
    return part


def _records(frame: pd.DataFrame) -> list[dict]:
    """pandas turns numpy types and NaN (as null) into plain JSON values."""
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def _score(part: pd.DataFrame, backend, cfg: dict, method: str):
    """Scores the conversations in `part` (rows sorted by group, then turn) in one model pass.

    Returns (part with score, cue tags, method A points, points of `method`). Method A points are
    always needed: the lead status uses them, as in leads.parquet.
    """
    if method not in METHODS:
        raise InvalidConversation(f"Unknown turning-point method {method!r}.")
    part = part.reset_index(drop=True)
    tags = rules.tag_turns(part["role"], part["text"])
    cue_matrix = rules.tags_to_matrix(tags)
    part["score"] = np.asarray(backend.score(part, cue_matrix), dtype=float)
    part["model"] = backend.name

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
    return part, tags, cue_matrix, delta_points, points


def _detail(part: pd.DataFrame, tags, delta_points: pd.DataFrame, points: pd.DataFrame,
            dataset: str, backend, cfg: dict, conversation_id: str, title: str, method: str) -> dict:
    """The fields of GET /api/conversations/{id} for one scored conversation, without an outcome."""
    rows = np.flatnonzero(part["conversation_id"].to_numpy() == conversation_id)
    conversation = part.iloc[rows]
    own_delta = delta_points[delta_points["conversation_id"] == conversation_id]
    own_points = points[points["conversation_id"] == conversation_id]
    last_point = next(own_delta.iloc[::-1].itertuples(index=False), None)
    scores = conversation["score"].to_numpy()
    status_cfg = cfg["status"]
    trend = export.trend_slope(scores, status_cfg["trend_window"])
    status = export.lead_status(scores[-1], trend, status_cfg)

    transcript = conversation[["turn", "role", "speaker", "text"]].assign(cues=[list(tags[row]) for row in rows])
    return {
        "conversation_id": conversation_id,
        "title": title.strip() or "Uploaded conversation",
        "dataset": dataset,
        "model": backend.name,
        "outcome": None,
        "n_turns": len(conversation),
        "final_score": float(scores[-1]),
        "max_score": float(scores.max()),
        "trend": trend,
        "status": status,
        "status_reason": export.status_reason(status, last_point),
        "n_turning_points": len(own_points),
        "turns": _records(transcript),
        "scores": scores.tolist(),
        "scores_by_model": {backend.name: scores.tolist()},
        "turning_points": _records(own_points),
        "available_methods": list(METHODS),
        "available_taggers": ["rules"],
        "method": method,
        "tagger": "rules",
    }


def analyze(turns: list[dict], dataset: str, backend, cfg: dict, title: str = "",
            method: str = "delta") -> dict:
    """Scores `turns` ([{role, text, speaker?}, ...]) with `backend` (see load_backend).

    Returns the same fields as GET /api/conversations/{id}, without an outcome.
    """
    if method not in METHODS:
        raise InvalidConversation(f"Unknown turning-point method {method!r}.")
    part = to_frame(turns, dataset, cfg["analyze"])
    part, tags, _, delta_points, points = _score(part, backend, cfg, method)
    return _detail(part, tags, delta_points, points, dataset, backend, cfg, CONVERSATION_ID, title, method)


def batch_id(index: int) -> str:
    return f"upload-{index + 1:03d}"


def cue_summary(part: pd.DataFrame, tags) -> list[dict]:
    """How many conversations (and turns) show each cue, customer cues and agent moves alike."""
    counts = {}
    for conversation_id, role, turn_tags in zip(part["conversation_id"], part["role"], tags):
        for cue in turn_tags:
            entry = counts.setdefault(cue, {"conversations": set(), "turns": 0})
            entry["conversations"].add(conversation_id)
            entry["turns"] += 1
    return sorted(
        ({"cue": cue, "kind": "customer" if cue in CUSTOMER_CUE_NAMES else "agent",
          "n_conversations": len(entry["conversations"]), "n_turns": entry["turns"]}
         for cue, entry in counts.items()),
        key=lambda row: (-row["n_conversations"], -row["n_turns"], row["cue"]),
    )


def script_insights(part: pd.DataFrame, cue_matrix: np.ndarray, tags, cfg: dict):
    """Agent-move effects over every uploaded conversation (all of them are in scope), with the
    agent and customer messages of each example filled in for the page."""
    settings = {**cfg["insights"], "min_count": cfg["analyze"]["batch_insights_min_count"]}
    effects, examples = insights.move_effects(
        part.assign(in_scope=True), cue_matrix[:, len(CUSTOMER_CUE_NAMES):], settings, cfg["seed"])
    row_of = {(c, int(t)): row for row, (c, t) in enumerate(zip(part["conversation_id"], part["turn"]))}
    records = _records(examples)
    for example in records:
        for role in ("agent", "customer"):
            row = row_of[(example["conversation_id"], example[f"{role}_turn"])]
            example[f"{role}_text"] = part["text"].iat[row]
            example[f"{role}_cues"] = list(tags[row])
    return _records(effects), records


def analyze_batch(conversations: list[dict], dataset: str, backend, cfg: dict,
                  method: str = "delta") -> dict:
    """Scores many conversations ([{title?, turns}, ...]) in one model pass.

    Returns a lead list (one row per scored conversation, in upload order, with its score curve),
    every conversation's detail (as `analyze` returns it), script insights over the whole batch and
    how often each cue appeared. A conversation that cannot be scored is listed in `skipped` with
    the reason instead of failing the batch.
    """
    if method not in METHODS:
        raise InvalidConversation(f"Unknown turning-point method {method!r}.")
    limits = cfg["analyze"]
    if not conversations:
        raise InvalidConversation("Add at least one conversation.")
    if len(conversations) > limits["max_conversations"]:
        raise InvalidConversation(
            f"The batch has {len(conversations)} conversations; the limit is {limits['max_conversations']}.")

    frames, titles, skipped = [], {}, []
    for index, conversation in enumerate(conversations):
        title = str(conversation.get("title") or "").strip() or f"Conversation {index + 1}"
        try:
            frame = to_frame(conversation.get("turns") or [], dataset, limits, batch_id(index), len(frames))
        except InvalidConversation as error:
            skipped.append({"index": index, "title": title, "reason": str(error)})
            continue
        frames.append(frame)
        titles[batch_id(index)] = title
    if not frames:
        raise InvalidConversation("None of the conversations could be scored. " + skipped[0]["reason"])

    part, tags, cue_matrix, delta_points, points = _score(pd.concat(frames, ignore_index=True), backend, cfg, method)
    details = {
        conversation_id: _detail(part, tags, delta_points, points, dataset, backend, cfg,
                                 conversation_id, title, method)
        for conversation_id, title in titles.items()
    }
    lead_fields = ("conversation_id", "title", "dataset", "model", "outcome", "n_turns", "final_score",
                   "max_score", "trend", "status", "status_reason", "n_turning_points")
    leads = [{**{field: detail[field] for field in lead_fields}, "curve": detail["scores"]}
             for detail in details.values()]
    effects, examples = script_insights(part, cue_matrix, tags, cfg)
    return {
        "dataset": dataset,
        "model": backend.name,
        "method": method,
        "n_conversations": len(details),
        "n_turns": len(part),
        "leads": leads,
        "conversations": details,
        "insights": effects,
        "insight_examples": examples,
        "insights_min_count": limits["batch_insights_min_count"],
        "cue_summary": cue_summary(part, tags),
        "skipped": skipped,
    }
