"""Prints the real schema and example records of each raw dataset (spec rule 4).

Run: python -m inclination.load.inspect_raw
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from inclination.config import ROOT, load_config, path

MAX_STR = 400  # display truncation only


def _plain(value):
    """Converts numpy containers to plain Python and truncates long strings for printing."""
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, str) and len(value) > MAX_STR:
        return value[:MAX_STR] + f"... [{len(value)} chars]"
    return value


def show(name: str, df: pd.DataFrame, n_examples: int) -> None:
    print(f"\n{'=' * 100}\n{name}: {len(df):,} rows x {df.shape[1]} columns\n{'=' * 100}")
    print(df.dtypes.to_string())
    for i in range(min(n_examples, len(df))):
        print(f"\n--- example {i} ---")
        print(json.dumps(_plain(df.iloc[i].to_dict()), indent=2, default=str, ensure_ascii=False))


def inspect_craigslist(raw: Path, n: int) -> None:
    src = load_config()["data"]["sources"]["craigslist"]
    frames = {
        id_split: pd.DataFrame(json.loads((raw / name).read_text(encoding="utf-8")))
        for id_split, name in src["files"].items()
    }
    recovered = json.loads((ROOT / src["recovered_file"]).read_text(encoding="utf-8"))
    for id_split, df in frames.items():
        rewards = df["outcome"].map(lambda o: o["reward"])
        was_unlabelled = rewards.iloc[recovered[id_split]]
        print(f"\ncraigslist {src['files'][id_split]} ({id_split} ids): {len(df):,} dialogues | "
              f"reward=1: {rewards.mean():.3f} | without acts in the HF copy: "
              f"{len(was_unlabelled):,} (reward=1: {was_unlabelled.mean():.3f})")
    df = pd.concat(frames.values(), ignore_index=True)
    show("craigslist (all files)", df, n)

    actions, last_action = Counter(), Counter()
    accept_vs_reward, text_after_action = Counter(), 0
    for events, outcome in zip(df["events"], df["outcome"]):
        seq = [e["action"] for e in events]
        actions.update(seq)
        last_action[seq[-1] if seq else "<none>"] += 1
        accept_vs_reward[("accept" in seq, outcome["reward"])] += 1
        is_action = [a != "message" for a in seq]
        if True in is_action and not all(is_action[is_action.index(True):]):
            text_after_action += 1
    print("\nevent actions over all events:", dict(actions.most_common()))
    print("action of the LAST event:", dict(last_action.most_common()))
    print("(accept event present, reward):", dict(accept_vs_reward))
    print("events per dialogue:", df["events"].map(len).describe().round(2).to_dict())
    print(f"dialogues with message events after the first action event: {text_after_action:,}")
    print("categories:", dict(Counter(s["category"] for s in df["scenario"]).most_common()))
    print("role of first speaker:", Counter(
        s["kbs"][int(events[0]["agent"])]["personal"]["Role"] if len(events) else "<empty>"
        for s, events in zip(df["scenario"], df["events"])
    ))


def inspect_p4g(raw: Path, n: int) -> None:
    dialog = pd.read_csv(raw / "full_dialog.csv")
    info = pd.read_csv(raw / "full_info.csv")
    show("p4g full_dialog.csv", dialog, n)
    show("p4g full_info.csv", info, n)
    print("\nfull_info columns:", list(info.columns))
    print("dialogues in full_dialog:", dialog["B2"].nunique(), "| in full_info:", info["B2"].nunique())
    print("role flag B4 in full_dialog:", dialog["B4"].value_counts().to_dict())
    print("role flag B4 in full_info:", info["B4"].value_counts().to_dict())
    print("turn index range:", dialog["Turn"].min(), "-", dialog["Turn"].max())
    persuadee = info[info["B4"] == 1]
    for role, rows in (("persuader", info[info["B4"] == 0]), ("persuadee", persuadee)):
        values = pd.to_numeric(rows["B6"], errors="coerce")
        print(f"{role} B6: missing={values.isna().sum()}, >0 share={(values > 0).mean():.3f}, "
              f"describe={values.describe().round(2).to_dict()}")
    print("B7 (turns) describe:", info["B7"].describe().round(2).to_dict())
    per_dialog = dialog.groupby("B2").agg(rows=("Unit", "size"), max_turn=("Turn", "max"))
    print("rows per dialogue:", per_dialog["rows"].describe().round(2).to_dict())
    print("first speaker flag B4:", dialog.groupby("B2")["B4"].first().value_counts().to_dict())
    first = dialog["B2"].iloc[0]
    print(f"\nfirst dialogue ({first}) in full:")
    print(dialog[dialog["B2"] == first].to_string())


def inspect_saas(raw: Path, n: int) -> None:
    df = pd.concat([pd.read_parquet(p) for p in sorted(raw.glob("*.parquet"))], ignore_index=True)
    show("saas", df, n)
    print("\noutcome:", df["outcome"].value_counts().to_dict(), f"rate={df['outcome'].mean():.3f}")
    print("conversation_id unique:", df["conversation_id"].is_unique)
    print("conversation_length:", df["conversation_length"].describe().round(2).to_dict())
    for col in ("conversation_style", "conversation_flow", "communication_channel", "product_type"):
        print(f"{col}:", df[col].value_counts().head(8).to_dict())
    speakers, keys, bad = Counter(), Counter(), 0
    for raw_conv in df["conversation"].head(2000):
        try:
            for message in json.loads(raw_conv):
                keys.update(message.keys())
                speakers[message.get("speaker", message.get("role"))] += 1
        except (json.JSONDecodeError, AttributeError, TypeError):
            bad += 1
    print("message keys (first 2000 conversations):", dict(keys))
    print("speakers:", dict(speakers), "| unparseable:", bad)
    print("leak check, corr(outcome, customer_engagement / sales_effectiveness):",
          round(df["outcome"].corr(df["customer_engagement"]), 3),
          round(df["outcome"].corr(df["sales_effectiveness"]), 3))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 200, "display.max_colwidth", 120)
    n = load_config()["data"]["inspect_examples"]
    raw = path("raw")
    inspect_craigslist(raw / "craigslist", n)
    inspect_p4g(raw / "p4g", n)
    inspect_saas(raw / "saas", n)


if __name__ == "__main__":
    main()
