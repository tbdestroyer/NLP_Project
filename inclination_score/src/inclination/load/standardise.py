"""Builds data/processed/turns.parquet and conversations.parquet from the three raw datasets.

Run: python -m inclination.load.standardise
"""
import json
import re
import sys

import pandas as pd

from inclination.config import load_config, path
from inclination.load import craigslist, p4g, saas
from inclination.load.split import load_or_make_splits

LOADERS = {"craigslist": craigslist, "p4g": p4g, "saas": saas}
TURN_COLUMNS = ["dataset", "conversation_id", "turn", "role", "text"]
CONVERSATION_COLUMNS = [
    "dataset", "conversation_id", "outcome", "split", "n_turns", "title", "meta", "is_synthetic",
]
SPLITS = ["train", "val", "test"]
_WHITESPACE = re.compile(r"\s+")


def clean_text(text) -> str:
    if not isinstance(text, str):
        return ""
    return _WHITESPACE.sub(" ", text).strip()


def to_tables(records: list[dict], min_turns: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cleans text, drops empty turns, renumbers turns from 0 and drops too-short conversations."""
    turn_rows, conversation_rows = [], []
    for record in records:
        turns = [(role, clean_text(text)) for role, text in record["turns"]]
        turns = [(role, text) for role, text in turns if text]
        if len(turns) < min_turns:
            continue
        for k, (role, text) in enumerate(turns):
            turn_rows.append((record["dataset"], record["conversation_id"], k, role, text))
        conversation_rows.append({
            "dataset": record["dataset"],
            "conversation_id": record["conversation_id"],
            "outcome": record["outcome"],
            "n_turns": len(turns),
            "title": clean_text(record["title"]),
            "meta": json.dumps(record["meta"], ensure_ascii=False),
            "is_synthetic": record["is_synthetic"],
        })
    return pd.DataFrame(turn_rows, columns=TURN_COLUMNS), pd.DataFrame(conversation_rows)


def dataset_stats(conversations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in conversations.groupby("dataset"):
        for split, part in [("all", group)] + [(s, group[group["split"] == s]) for s in SPLITS]:
            rows.append({
                "dataset": dataset,
                "split": split,
                "conversations": len(part),
                "avg_turns": round(part["n_turns"].mean(), 2),
                "outcome_rate": round(part["outcome"].mean(), 3),
            })
    return pd.DataFrame(rows)


def to_markdown(stats: pd.DataFrame) -> str:
    lines = ["| dataset | split | conversations | avg turns | outcome rate |", "|---|---|---|---|---|"]
    for row in stats.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.split} | {row.conversations:,} | {row.avg_turns:.2f} "
            f"| {row.outcome_rate:.1%} |"
        )
    return "\n".join(lines)


def print_examples(turns: pd.DataFrame, conversations: pd.DataFrame) -> None:
    """One standardised example conversation per dataset."""
    for dataset, group in conversations.groupby("dataset"):
        conversation = group.sort_values("conversation_id").iloc[0]
        print(f"\n--- {dataset}: {conversation['conversation_id']} | {conversation['title']} ---")
        print(f"outcome={conversation['outcome']} split={conversation['split']} "
              f"n_turns={conversation['n_turns']} synthetic={conversation['is_synthetic']} "
              f"meta={conversation['meta']}")
        rows = turns[turns["conversation_id"] == conversation["conversation_id"]]
        for row in rows.itertuples(index=False):
            print(f"  [{row.turn:>2}] {row.role:<8} {row.text}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    data_cfg = cfg["data"]
    all_turns, all_conversations, batches = [], [], {}
    for name in data_cfg["datasets"]:
        records = LOADERS[name].load(path("raw") / name, cfg)
        batches.update(
            {r["conversation_id"]: r["split_batch"] for r in records if "split_batch" in r}
        )
        turns, conversations = to_tables(records, data_cfg["min_turns"])
        print(f"{name}: {len(records):,} labelled conversations loaded, "
              f"{len(conversations):,} kept (>= {data_cfg['min_turns']} turns)")
        all_turns.append(turns)
        all_conversations.append(conversations)
    turns = pd.concat(all_turns, ignore_index=True)
    conversations = pd.concat(all_conversations, ignore_index=True)

    assignment = load_or_make_splits(
        conversations, data_cfg["split_fractions"], cfg["seed"], path("splits"), batches
    )
    conversations["split"] = conversations["conversation_id"].map(assignment)
    conversations = conversations[CONVERSATION_COLUMNS]

    processed = path("processed")
    processed.mkdir(parents=True, exist_ok=True)
    turns.to_parquet(processed / "turns.parquet", index=False)
    conversations.to_parquet(processed / "conversations.parquet", index=False)

    stats = dataset_stats(conversations)
    stats.to_csv(processed / "dataset_stats.csv", index=False)
    print("\n" + to_markdown(stats))
    print_examples(turns, conversations)
    print(f"\nWrote {len(turns):,} turns and {len(conversations):,} conversations to {processed}")


if __name__ == "__main__":
    main()
