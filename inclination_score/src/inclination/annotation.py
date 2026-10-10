"""Hand-labelling support: the fixed sample of conversations and the label files.

Label files are JSON lines in annotation/labels/{annotator}.jsonl, one line per saved
conversation. Saving again appends a new line; the latest line for a conversation wins.
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from inclination.config import ROOT, load_config

_UNSAFE = re.compile(r"[^a-z0-9_-]+")


def sample_path() -> Path:
    return ROOT / load_config()["annotation"]["sample_file"]


def labels_dir() -> Path:
    return ROOT / load_config()["paths"]["annotation_labels"]


def build_sample(conversations: pd.DataFrame, cfg: dict) -> dict[str, list[str]]:
    """Random test-split conversations, the same number per dataset, fixed by the seed."""
    datasets = cfg["data"]["datasets"]
    per_dataset = cfg["annotation"]["n_conversations"] // len(datasets)
    test = conversations[conversations["split"] == "test"].sort_values("conversation_id")
    sample = {}
    for dataset in datasets:
        pool = test[test["dataset"] == dataset]
        chosen = pool.sample(n=min(per_dataset, len(pool)), random_state=cfg["seed"])
        sample[dataset] = chosen["conversation_id"].tolist()
    return sample


def ensure_sample(conversations: pd.DataFrame, cfg: dict) -> dict[str, list[str]]:
    """Creates the sample once. An existing file is never changed, so labels stay valid."""
    target = sample_path()
    if target.exists():
        return json.loads(target.read_text(encoding="utf-8"))
    sample = build_sample(conversations, cfg)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(sample, indent=1), encoding="utf-8")
    return sample


def safe_name(annotator: str) -> str:
    """File-safe annotator id: lower case letters, digits, '-' and '_'."""
    return _UNSAFE.sub("_", annotator.strip().lower()).strip("_")


def save_labels(annotator: str, conversation_id: str, dataset: str, labels: list[dict],
                folder: Path | None = None) -> Path:
    """`labels`: one dict per turn with turn, role, direction (customer turns) and turning_point."""
    folder = folder or labels_dir()
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{safe_name(annotator)}.jsonl"
    record = {
        "annotator": safe_name(annotator),
        "conversation_id": conversation_id,
        "dataset": dataset,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "labels": labels,
    }
    with open(target, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return target


def load_labels(annotator: str, folder: Path | None = None) -> dict[str, dict]:
    """{conversation_id: latest saved record} for one annotator."""
    target = (folder or labels_dir()) / f"{safe_name(annotator)}.jsonl"
    if not target.exists():
        return {}
    records = {}
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["conversation_id"]] = record
    return records


def list_annotators(folder: Path | None = None) -> list[str]:
    folder = folder or labels_dir()
    return sorted(p.stem for p in folder.glob("*.jsonl")) if folder.exists() else []


_COMPACT_DIRECTIONS = {"m": "more", "s": "same", "l": "less"}
_COMPACT_PREFIX = {"C": "craigslist", "P": "p4g", "S": "saas"}
_TOKEN = re.compile(r"(\d+)([msl]?)(\*?)$")


def import_compact(source: Path, annotator: str, turns: pd.DataFrame, sample: dict,
                   folder: Path | None = None) -> int:
    """Imports labels written in the compact text format (see annotation/llm_labels.txt).

    Every customer turn of every listed conversation must have a direction, and a direction may
    only be given to a customer turn; otherwise nothing is written.
    """
    records = []
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        code, *tokens = line.split()
        dataset = _COMPACT_PREFIX[code[0]]
        conversation_id = sample[dataset][int(code[1:]) - 1]
        rows = turns[turns["conversation_id"] == conversation_id].sort_values("turn")
        roles = dict(zip(rows["turn"], rows["role"]))
        given = {}
        for token in tokens:
            turn, direction, point = _TOKEN.match(token).groups()
            given[int(turn)] = (_COMPACT_DIRECTIONS.get(direction), bool(point))
        for turn, (direction, _) in given.items():
            if turn not in roles or (direction is not None) != (roles[turn] == "customer"):
                raise ValueError(f"{code}: turn {turn} does not match the transcript")
        missing = [t for t, role in roles.items() if role == "customer" and t not in given]
        if missing:
            raise ValueError(f"{code}: customer turns without a direction: {missing}")
        labels = [
            {"turn": int(turn), "role": role, "direction": given.get(turn, (None, False))[0],
             "turning_point": given.get(turn, (None, False))[1]}
            for turn, role in roles.items()
        ]
        records.append((conversation_id, dataset, labels))
    for conversation_id, dataset, labels in records:
        save_labels(annotator, conversation_id, dataset, labels, folder)
    return len(records)


if __name__ == "__main__":
    import sys

    from inclination.config import path

    source_file, annotator_name = Path(sys.argv[1]), sys.argv[2]
    sample_ids = json.loads(sample_path().read_text(encoding="utf-8"))
    all_turns = pd.read_parquet(path("outputs") / "turns.parquet")
    target = labels_dir() / f"{safe_name(annotator_name)}.jsonl"
    if target.exists():
        target.unlink()  # a re-import replaces this annotator's file
    count = import_compact(source_file, annotator_name, all_turns, sample_ids)
    print(f"Imported {count} conversations for annotator '{safe_name(annotator_name)}' into {target}")
