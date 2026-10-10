"""Prints the real schema and example records of each raw dataset (spec rule 4).

Run: python -m inclination.load.inspect_raw
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from inclination.config import load_config, path

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
    frames = {p.stem.split("-")[-1]: pd.read_parquet(p) for p in sorted(raw.rglob("*.parquet"))}
    for split, df in frames.items():
        no_acts = df["dialogue_acts"].map(lambda a: not any(list(a["intent"])))
        ends_empty = df["utterance"].map(lambda u: len(u) > 0 and not str(u[-1]).strip())
        print(f"\ncraigslist split {split}: {len(df):,} dialogues | "
              f"with NO dialogue acts at all: {no_acts.sum():,} | "
              f"of those ending in an empty utterance: {(no_acts & ends_empty).sum():,}")
    df = pd.concat(frames.values(), ignore_index=True)
    show("craigslist (all splits)", df, n)

    intents = Counter()
    last_intent = Counter()
    empty_with_intent = Counter()
    for acts, utterances in zip(df["dialogue_acts"], df["utterance"]):
        seq = list(acts["intent"])
        intents.update(seq)
        last_intent[seq[-1] if seq else "<none>"] += 1
        for intent, text in zip(seq, utterances):
            if not str(text).strip():
                empty_with_intent[intent] += 1
    print("\nintent counts over all turns:", dict(intents.most_common()))
    print("intent of the LAST turn:", dict(last_intent.most_common()))
    print("intent of turns with EMPTY utterance:", dict(empty_with_intent.most_common()))
    print("turns per dialogue:", df["utterance"].map(len).describe().round(2).to_dict())
    terminal, text_after_action = Counter(), 0
    for acts in df["dialogue_acts"]:
        seq = list(acts["intent"])
        if not any(seq):
            continue
        present = set(seq)
        terminal[next((a for a in ("accept", "reject", "quit", "offer") if a in present), "none")] += 1
        is_action = [a in ("offer", "accept", "reject", "quit") for a in seq]
        if True in is_action and not all(is_action[is_action.index(True):]):
            text_after_action += 1
    labelled = sum(terminal.values())
    print(f"dialogues WITH acts: {labelled:,} | strongest action act present: {dict(terminal)}")
    print(f"accept rate among dialogues with acts: {terminal['accept'] / labelled:.3f}")
    print(f"dialogues with text turns after the first action turn: {text_after_action:,}")
    print("categories:", dict(Counter(i["Category"][0] for i in df["items"]).most_common()))
    print("role of first speaker:", Counter(
        info["Role"][turns[0]] if len(turns) else "<empty>"
        for info, turns in zip(df["agent_info"], df["agent_turn"])
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
