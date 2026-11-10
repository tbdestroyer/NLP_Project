"""Data server for the React UI in web/. It only reads the files `make export` and `make eval`
write into outputs/, the same files the Streamlit dashboard reads. It never trains or calls an LLM.
Live mode (/api/live/*) also runs a local Whisper model for speech-to-text.

Run:  make api      (or: .venv/bin/python -m uvicorn api.server:app --port 8000 --reload)
"""
import json
import os
import re
import threading
from functools import lru_cache
from pathlib import Path
from datetime import datetime, timezone
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from pydantic import BaseModel
from inclination import analyze as analysis
from inclination import live
from inclination.config import load_config
from inclination.export import BACKENDS

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = Path(os.environ.get("INCLINATION_OUTPUTS", ROOT / "outputs"))
REQUIRED = [
    "leads.parquet",
    "scores.parquet",
    "turns.parquet",
    "cues.parquet",
    "turning_points.parquet",
    "insights.parquet",
    "insight_examples.parquet",
]

app = FastAPI(title="Inclination Score data API")
_UNSAFE_ANNOTATOR = re.compile(r"[^a-z0-9_-]+")


class TurnLabel(BaseModel):
    turn: int
    role: Literal["agent", "customer"]
    direction: Literal["more", "same", "less"] | None
    turning_point: bool


class AnalyzeTurn(BaseModel):
    role: Literal["agent", "customer"]
    text: str
    speaker: str = ""


class AnalyzeRequest(BaseModel):
    dataset: str
    model: str = "v1"
    method: Literal["delta", "pelt"] = "delta"
    title: str = ""
    turns: list[AnalyzeTurn]


class BatchConversation(BaseModel):
    title: str = ""
    turns: list[AnalyzeTurn]


class AnalyzeBatchRequest(BaseModel):
    dataset: str
    model: str = "v1"
    method: Literal["delta", "pelt"] = "delta"
    conversations: list[BatchConversation]


class LiveRequest(BaseModel):
    dataset: str
    model: str = "v1"
    turns: list[AnalyzeTurn]


class AnnotationSubmission(BaseModel):
    annotator: str
    conversation_id: str
    dataset: str
    labels: list[TurnLabel]


def as_json(frame_or_obj) -> Response:
    """pandas handles numpy types, list cells and NaN (as null) when it writes JSON."""
    if isinstance(frame_or_obj, pd.DataFrame):
        body = frame_or_obj.to_json(orient="records", force_ascii=False)
    else:
        body = json.dumps(frame_or_obj)
    return Response(body, media_type="application/json")


def _mtime(name: str) -> float:
    target = OUTPUTS / name
    if not target.exists():
        raise HTTPException(
            503, f"outputs/{name} is missing. Run `make all` in inclination_score/ (or copy a "
                 "teammate's outputs/ folder) and restart the server.")
    return target.stat().st_mtime


@lru_cache(maxsize=16)
def _read(name: str, modified: float) -> pd.DataFrame:
    return pd.read_parquet(OUTPUTS / name)


def load(name: str) -> pd.DataFrame:
    """Reads a parquet file once and re-reads it only when the file changes."""
    return _read(name, _mtime(name))


def pick(frame: pd.DataFrame, **filters) -> pd.DataFrame:
    for column, value in filters.items():
        frame = frame[frame[column] == value]
    return frame


def safe_annotator(name: str) -> str:
    return _UNSAFE_ANNOTATOR.sub("_", name.strip().lower()).strip("_")


def annotation_paths():
    config = load_config()
    return (
        ROOT / config["annotation"]["sample_file"],
        ROOT / config["paths"]["annotation_labels"],
    )


def load_annotation_labels(annotator: str, labels_dir: Path) -> dict[str, dict]:
    target = labels_dir / f"{safe_annotator(annotator)}.jsonl"
    if not target.exists():
        return {}
    records = {}
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["conversation_id"]] = record
    return records


@app.get("/api/health")
def health():
    missing = [name for name in REQUIRED if not (OUTPUTS / name).exists()]
    return {"ok": not missing, "outputs": str(OUTPUTS), "missing": missing}


@app.get("/api/options")
def options():
    """Which datasets and models exist in the exported files."""
    leads = load("leads.parquet")
    pairs = leads[["dataset", "model"]].drop_duplicates()
    return {d: sorted(group["model"].tolist()) for d, group in pairs.groupby("dataset")}


@app.get("/api/ui-config")
def ui_config():
    config = load_config()
    return {
        "insights": {
            "min_count": config["insights"]["min_count"],
            "position_buckets": config["insights"]["position_buckets"],
        },
        "eval": {"precision_at": config["eval"]["precision_at"]},
        "annotation": {"directions": config["annotation"]["directions"]},
    }


@app.get("/api/leads")
def leads(dataset: str, model: str, split: str = "test"):
    frame = pick(load("leads.parquet"), dataset=dataset, model=model)
    if split != "all":
        frame = frame[frame["split"] == split]
    scores = pick(load("scores.parquet"), dataset=dataset, model=model)
    scores = scores[scores["conversation_id"].isin(frame["conversation_id"])]
    curves = scores.sort_values("turn").groupby("conversation_id")["score"].agg(list)
    frame = frame.drop(columns=["meta"], errors="ignore").copy()
    frame["curve"] = frame["conversation_id"].map(curves)
    return as_json(frame)


@app.get("/api/search")
def search(dataset: str, q: str):
    turns = pick(load("turns.parquet"), dataset=dataset)
    hits = turns.loc[turns["text"].str.contains(q, case=False, regex=False), "conversation_id"]
    return sorted(set(hits))


@app.get("/api/conversations/{conversation_id}")
def conversation(
    conversation_id: str,
    model: str,
    method: str = "delta",
    tagger: str = "rules",
    models: str = "",
):
    lead = pick(load("leads.parquet"), conversation_id=conversation_id, model=model)
    if lead.empty:
        raise HTTPException(404, f"No {model} scores for conversation {conversation_id}.")
    record = json.loads(lead.drop(columns=["meta"], errors="ignore").to_json(orient="records"))[0]

    turns = pick(load("turns.parquet"), conversation_id=conversation_id).sort_values("turn")
    all_cues = pick(load("cues.parquet"), conversation_id=conversation_id)
    available_taggers = sorted(all_cues["tagger"].dropna().unique().tolist())
    if tagger not in available_taggers:
        tagger = "rules" if "rules" in available_taggers else (available_taggers[0] if available_taggers else "")
    cues = pick(all_cues, tagger=tagger) if tagger else pd.DataFrame(columns=["turn", "cues"])
    cue_map = {int(t): list(c) for t, c in zip(cues["turn"], cues["cues"])}
    turns = turns[["turn", "role", "text"]].copy()
    turns["cues"] = [cue_map.get(int(t), []) for t in turns["turn"]]

    all_scores = pick(load("scores.parquet"), conversation_id=conversation_id)
    available_models = sorted(all_scores["model"].dropna().unique().tolist())
    overlay_models = [entry for entry in models.split(",") if entry in available_models]
    if model not in overlay_models:
        overlay_models.insert(0, model)
    score_map = {
        score_model: pick(all_scores, model=score_model).sort_values("turn")["score"].tolist()
        for score_model in overlay_models
    }
    scores = score_map[model]

    all_points = pick(load("turning_points.parquet"), conversation_id=conversation_id, model=model)
    dataset_points = pick(
        load("turning_points.parquet"),
        dataset=record["dataset"],
        model=model,
    )
    available_methods = sorted(dataset_points["method"].dropna().unique().tolist())
    if method not in available_methods:
        method = "delta" if "delta" in available_methods else (available_methods[0] if available_methods else "")
    points = pick(all_points, method=method) if method else all_points.iloc[0:0]
    points = points.sort_values("turn")

    record["turns"] = json.loads(turns.to_json(orient="records", force_ascii=False))
    record["scores"] = scores
    record["scores_by_model"] = score_map
    record["turning_points"] = json.loads(points.to_json(orient="records", force_ascii=False))
    record["available_methods"] = available_methods
    record["available_taggers"] = available_taggers
    record["method"] = method
    record["tagger"] = tagger
    return record


@app.get("/api/insights")
def insights(dataset: str, model: str):
    if not (OUTPUTS / "insights.parquet").exists():
        return []
    return as_json(pick(load("insights.parquet"), dataset=dataset, model=model))


@app.get("/api/insight-examples")
def insight_examples(dataset: str, model: str, move: str, position: str):
    examples = pick(
        load("insight_examples.parquet"),
        dataset=dataset,
        model=model,
        move=move,
        position=position,
    ).sort_values("delta", ascending=False)
    if examples.empty:
        return as_json(examples)
    turns = pick(load("turns.parquet"), dataset=dataset)
    cues = pick(load("cues.parquet"), dataset=dataset, tagger="rules")
    by_conversation = {
        conversation_id: group.set_index("turn")
        for conversation_id, group in turns.groupby("conversation_id")
    }
    cue_maps = {
        conversation_id: dict(zip(group["turn"], group["cues"]))
        for conversation_id, group in cues.groupby("conversation_id")
    }
    records = json.loads(examples.to_json(orient="records", force_ascii=False))
    for row in records:
        transcript = by_conversation.get(row["conversation_id"])
        if transcript is None:
            continue
        for role in ("agent", "customer"):
            turn = row["agent_turn"] if role == "agent" else row["customer_turn"]
            message = transcript.loc[turn]
            row[f"{role}_text"] = message["text"]
            row[f"{role}_cues"] = list(cue_maps.get(row["conversation_id"], {}).get(turn, []))
    return as_json(records)


@app.get("/api/metrics")
def metrics(dataset: str, model: str):
    target = OUTPUTS / "metrics.json"
    if not target.exists():
        return None
    outcome = json.loads(target.read_text(encoding="utf-8")).get("outcome", {})
    found = outcome.get(dataset, {}).get(model)
    return None if found is None else {"roc_auc": found["roc_auc"], "roc_auc_ci": found["roc_auc_ci"]}


@app.get("/api/annotations")
def annotations(annotator: str, dataset: str = "all"):
    name = safe_annotator(annotator)
    if not name:
        raise HTTPException(422, "Enter an annotator name.")
    sample_path, labels_dir = annotation_paths()
    if not sample_path.exists():
        raise HTTPException(503, "annotation/sample.json is missing. Run `make export` in inclination_score/.")
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    if dataset != "all" and dataset not in sample:
        raise HTTPException(404, f"No annotation sample for dataset {dataset}.")
    saved = load_annotation_labels(name, labels_dir)
    queue = [
        (dataset_name, conversation_id)
        for dataset_name, conversation_ids in sample.items()
        if dataset == "all" or dataset == dataset_name
        for conversation_id in conversation_ids
    ]
    turns_frame = load("turns.parquet")
    conversations = []
    for dataset_name, conversation_id in queue:
        turns = pick(
            turns_frame,
            dataset=dataset_name,
            conversation_id=conversation_id,
        ).sort_values("turn")
        record = saved.get(conversation_id)
        conversations.append({
            "dataset": dataset_name,
            "conversation_id": conversation_id,
            "turns": json.loads(turns[["turn", "role", "text"]].to_json(orient="records", force_ascii=False)),
            "labels": record["labels"] if record else [],
            "labelled": record is not None,
        })
    return {
        "annotator": name,
        "dataset": dataset,
        "total": len(conversations),
        "completed": sum(conversation["labelled"] for conversation in conversations),
        "has_saved": (labels_dir / f"{name}.jsonl").exists(),
        "conversations": conversations,
    }


@app.post("/api/annotations")
def save_annotations(submission: AnnotationSubmission):
    annotator = safe_annotator(submission.annotator)
    if not annotator:
        raise HTTPException(422, "Enter an annotator name.")
    sample_path, labels_dir = annotation_paths()
    if not sample_path.exists():
        raise HTTPException(503, "annotation/sample.json is missing. Run `make export` in inclination_score/.")
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    if submission.conversation_id not in sample.get(submission.dataset, []):
        raise HTTPException(404, "This conversation is not in the fixed annotation sample.")

    turns = pick(
        load("turns.parquet"),
        dataset=submission.dataset,
        conversation_id=submission.conversation_id,
    ).sort_values("turn")
    expected_roles = {int(row.turn): row.role for row in turns.itertuples()}
    submitted = {label.turn: label for label in submission.labels}
    if len(submitted) != len(submission.labels) or set(submitted) != set(expected_roles):
        raise HTTPException(422, "Submit exactly one label for every transcript turn.")
    for turn, label in submitted.items():
        role = expected_roles[turn]
        if label.role != role:
            raise HTTPException(422, f"Turn {turn} must be labelled as {role}.")
        if role == "customer" and label.direction is None:
            raise HTTPException(422, f"Choose more / same / less for customer turn {turn}.")
        if role == "agent" and label.direction is not None:
            raise HTTPException(422, f"Agent turn {turn} cannot have an inclination direction.")

    record = {
        "annotator": annotator,
        "conversation_id": submission.conversation_id,
        "dataset": submission.dataset,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "labels": [submitted[turn].model_dump() for turn in expected_roles],
    }
    try:
        labels_dir.mkdir(parents=True, exist_ok=True)
        target = labels_dir / f"{annotator}.jsonl"
        with target.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as error:
        raise HTTPException(500, f"Could not save annotation labels: {error}") from error
    return {"saved": True, "annotator": annotator, "conversation_id": submission.conversation_id}


@lru_cache(maxsize=8)
def scoring_backend(model: str, dataset: str):
    """Loads a trained model once. Restart the server after retraining to pick up the new one."""
    return analysis.load_backend(model, dataset, load_config())


# A backend keeps the last scored turns on itself for the faithfulness re-scoring, so two
# requests must not score with it at the same time.
_scoring_lock = threading.Lock()


def backend_for(model: str, dataset: str):
    try:
        return scoring_backend(model, dataset)
    except analysis.ModelUnavailable as error:
        raise HTTPException(400, str(error)) from error
    except ImportError as error:  # v3 without torch / transformers installed
        raise HTTPException(400, f"{model} needs requirements-ml.txt installed: {error}") from error


@app.get("/api/analyze/models")
def analyze_models():
    """Which trained models can score a new conversation, per dataset. Reads outputs/models/ only,
    so it works before `make export`."""
    config = load_config()
    return {
        dataset: [model for model in analysis.MODELS if BACKENDS[model].available(dataset)]
        for dataset in config["data"]["datasets"]
    }


@app.post("/api/analyze")
def analyze_conversation(request: AnalyzeRequest):
    """Scores an uploaded conversation with a trained model. Nothing is stored."""
    backend = backend_for(request.model, request.dataset)
    turns = [turn.model_dump() for turn in request.turns]
    try:
        with _scoring_lock:
            return analysis.analyze(turns, request.dataset, backend, load_config(),
                                    title=request.title, method=request.method)
    except analysis.InvalidConversation as error:
        raise HTTPException(422, str(error)) from error


@app.post("/api/analyze/batch")
def analyze_batch(request: AnalyzeBatchRequest):
    """Scores many uploaded conversations: a lead list, every conversation's detail and script
    insights over the batch. Conversations that cannot be scored come back in `skipped`.
    Nothing is stored."""
    backend = backend_for(request.model, request.dataset)
    conversations = [conversation.model_dump() for conversation in request.conversations]
    try:
        with _scoring_lock:
            return analysis.analyze_batch(conversations, request.dataset, backend, load_config(),
                                          method=request.method)
    except analysis.InvalidConversation as error:
        raise HTTPException(422, str(error)) from error


# ---- Live mode ---------------------------------------------------------------------------------

_transcriber = None
_transcriber_lock = threading.Lock()


def transcriber() -> live.Transcriber:
    """The one Whisper instance. It is loaded on POST /api/live/load or the first transcription."""
    global _transcriber
    with _transcriber_lock:
        if _transcriber is None:
            _transcriber = live.Transcriber(load_config())
        return _transcriber


PLAYBOOK_INPUTS = ("leads.parquet", "turns.parquet", "cues.parquet", "scores.parquet")


@lru_cache(maxsize=8)
def _playbook(dataset: str, model: str, modified: tuple) -> dict:
    config = load_config()
    leads = pick(load("leads.parquet"), dataset=dataset, model=model)
    held_out = set(leads.loc[leads["split"].isin(config["live"]["playbook_splits"]), "conversation_id"])
    keys = ["conversation_id", "turn"]
    turns = pick(load("turns.parquet"), dataset=dataset)
    frame = (turns[turns["conversation_id"].isin(held_out)][[*keys, "role", "text"]]
             .merge(pick(load("cues.parquet"), dataset=dataset, tagger="rules")[[*keys, "cues"]], on=keys)
             .merge(pick(load("scores.parquet"), dataset=dataset, model=model)[[*keys, "score"]], on=keys))
    return live.build_playbook(frame, config)


def live_playbook(dataset: str, model: str) -> dict | None:
    """Coaching evidence from the exported outputs, built once per dataset and model (a few
    seconds) and rebuilt when the outputs change. None when outputs/ has not been exported."""
    try:
        return _playbook(dataset, model, tuple(_mtime(name) for name in PLAYBOOK_INPUTS))
    except HTTPException:
        return None


@app.get("/api/live/status")
def live_status():
    return transcriber().status()


@app.post("/api/live/load")
def live_load():
    """Loads Whisper (downloading it the first time) and warms it up. Takes about 20-30 s."""
    try:
        transcriber().load()
    except Exception as error:
        raise HTTPException(503, f"The speech model could not be loaded: {error}") from error
    return transcriber().status()


@app.post("/api/live/transcribe")
async def live_transcribe(request: Request):
    """One utterance as little-endian float32 mono samples at 16 kHz (the request body) -> text."""
    body = await request.body()
    try:
        audio = live.decode_pcm(body)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    engine = transcriber()
    try:
        return await run_in_threadpool(engine.transcribe, audio)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except Exception as error:
        raise HTTPException(503, f"Transcription failed: {error}") from error


@app.post("/api/live/score")
def live_score(request: LiveRequest):
    """The call so far: score after every message (once both sides have spoken) and coaching for
    the salesperson's next message. Nothing is stored."""
    backend = backend_for(request.model, request.dataset)
    turns = [turn.model_dump() for turn in request.turns]
    playbook = live_playbook(request.dataset, request.model)
    try:
        with _scoring_lock:
            return live.live_update(turns, request.dataset, backend, playbook, load_config())
    except analysis.InvalidConversation as error:
        raise HTTPException(422, str(error)) from error
