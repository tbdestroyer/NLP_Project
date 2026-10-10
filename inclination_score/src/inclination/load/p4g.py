"""Persuasion for Good loader: persuadee = customer, persuader = agent.

Outcome = the persuadee's actual donation (`B6` in full_info.csv) is above zero. Nothing else
from full_info.csv (survey scores, demographics, amounts) reaches the standard tables.
"""
from pathlib import Path

import pandas as pd

DATASET = "p4g"


def read_raw(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(raw_dir / "full_dialog.csv"), pd.read_csv(raw_dir / "full_info.csv")


def standardise(dialog: pd.DataFrame, info: pd.DataFrame, src: dict) -> list[dict]:
    persuadee = info[info["B4"] == src["customer_flag"]]
    donated = (persuadee.set_index("B2")["B6"] > 0).astype(int).to_dict()
    role_of = {src["customer_flag"]: "customer", src["agent_flag"]: "agent"}
    records = []
    # File order is message order; sort=False keeps dialogues in file order too.
    for number, (dialog_id, rows) in enumerate(dialog.groupby("B2", sort=False), start=1):
        if dialog_id not in donated:
            continue
        records.append({
            "dataset": DATASET,
            "conversation_id": f"{DATASET}-{dialog_id}",
            "outcome": donated[dialog_id],
            "title": f"Donation chat #{number}",
            "meta": {"charity": "Save the Children"},
            "is_synthetic": False,
            "turns": [(role_of[flag], text) for flag, text in zip(rows["B4"], rows["Unit"])],
        })
    return records


def load(raw_dir: Path, cfg: dict) -> list[dict]:
    dialog, info = read_raw(raw_dir)
    return standardise(dialog, info, cfg["data"]["sources"][DATASET])
