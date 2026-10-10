"""SaaS Sales Conversations loader (SYNTHETIC, GPT-4 generated): customer / sales_rep = agent.

Only the `conversation` messages, the `outcome` label and a few descriptive columns are read.
The generated annotations (probability_trajectory, engagement and effectiveness scores, scenario,
conversation_flow) reveal the outcome and are never touched.
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATASET = "saas"


def read_raw(raw_dir: Path, src: dict) -> pd.DataFrame:
    columns = ["conversation_id", "conversation", "outcome", *src["meta_columns"]]
    parts = [pd.read_parquet(p, columns=columns) for p in sorted(raw_dir.glob("*.parquet"))]
    return pd.concat(parts, ignore_index=True)


def subsample(df: pd.DataFrame, max_conversations: int, seed: int) -> pd.DataFrame:
    """Outcome-stratified sample, deterministic for a given seed."""
    if len(df) <= max_conversations:
        return df
    df = df.sort_values("conversation_id")
    kept, _ = train_test_split(
        df, train_size=max_conversations, stratify=df["outcome"], random_state=seed
    )
    return kept.sort_values("conversation_id")


def standardise(df: pd.DataFrame, src: dict) -> list[dict]:
    speaker_roles = src["speaker_roles"]
    records = []
    for row in df.to_dict("records"):
        turns = [
            (speaker_roles[m["speaker"]], m["message"])
            for m in json.loads(row["conversation"])
            if m.get("speaker") in speaker_roles
        ]
        raw_id = str(row["conversation_id"])
        records.append({
            "dataset": DATASET,
            "conversation_id": raw_id if raw_id.startswith(f"{DATASET}-") else f"{DATASET}-{raw_id}",
            "outcome": int(row["outcome"]),
            "title": f"{row['product_name']} · {raw_id}",
            "meta": {column: row[column] for column in src["meta_columns"]},
            "is_synthetic": True,
            "turns": turns,
        })
    return records


def load(raw_dir: Path, cfg: dict) -> list[dict]:
    src = cfg["data"]["sources"][DATASET]
    df = subsample(read_raw(raw_dir, src), cfg["data"]["saas_max_conversations"], cfg["seed"])
    return standardise(df, src)
