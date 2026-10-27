"""CraigslistBargains loader: buyer = customer, seller = agent, outcome = the dataset's own reward.

Reads the original CoCoA release (train/dev/test.json), which has an outcome for every dialogue.
Structured action events (offer / accept / reject / quit) are dropped, and nothing after an
accept or quit is kept. Conversation ids use the Hugging Face split names and row positions, so
they are the same as when the parquet copy was used.
"""
import json
from pathlib import Path

from inclination.config import ROOT

DATASET = "craigslist"


def read_raw(raw_dir: Path, src: dict) -> dict[str, list[dict]]:
    """Returns {id_split: dialogues}; the split name only serves to build stable ids."""
    return {
        id_split: json.loads((raw_dir / filename).read_text(encoding="utf-8"))
        for id_split, filename in src["files"].items()
    }


def read_recovered(recovered_path: Path) -> set[str]:
    """Ids of the dialogues that have no dialogue acts in the Hugging Face copy."""
    by_split = json.loads(recovered_path.read_text(encoding="utf-8"))
    return {f"{DATASET}-{split}-{i}" for split, indices in by_split.items() for i in indices}


def standardise(dialogues: dict[str, list[dict]], src: dict, recovered: set) -> list[dict]:
    action_events = set(src["action_events"])
    terminal_events = set(src["terminal_events"])
    records = []
    for id_split, rows in dialogues.items():
        for index, row in enumerate(rows):
            kbs = row["scenario"]["kbs"]
            roles = [kb["personal"]["Role"] for kb in kbs]
            turns = []
            for event in row["events"]:
                if event["action"] in action_events:
                    if event["action"] in terminal_events:
                        break
                    continue
                speaker = int(event["agent"])
                role = "customer" if roles[speaker] == src["customer_role"] else "agent"
                turns.append((role, event["data"]))
            item = kbs[0]["item"]  # both agents see the same title, category and listing price
            conversation_id = f"{DATASET}-{id_split}-{index}"
            record = {
                "dataset": DATASET,
                "conversation_id": conversation_id,
                "outcome": int(row["outcome"]["reward"]),
                "title": str(item["Title"]),
                "meta": {
                    "category": str(item["Category"]),
                    "listing_price": float(item["Price"]),
                },
                "is_synthetic": False,
                "turns": turns,
            }
            if conversation_id in recovered:
                # split on their own, so the dialogues used before keep their train/val/test split
                record["split_batch"] = src["recovered_batch"]
            records.append(record)
    return records


def load(raw_dir: Path, cfg: dict) -> list[dict]:
    src = cfg["data"]["sources"][DATASET]
    return standardise(read_raw(raw_dir, src), src, read_recovered(ROOT / src["recovered_file"]))
