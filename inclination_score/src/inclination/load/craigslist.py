"""CraigslistBargains loader: buyer = customer, seller = agent, outcome = an accept act exists.

Structured action turns (offer / accept / reject / quit) are used only to derive the outcome and
are then dropped. Dialogues without dialogue acts have an unknown outcome and are excluded.
"""
from pathlib import Path

import pandas as pd

DATASET = "craigslist"


def read_raw(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Returns {hf_split: frame}; the split name only serves to build stable conversation ids."""
    return {p.stem.split("-")[-1]: pd.read_parquet(p) for p in sorted(raw_dir.rglob("*.parquet"))}


def standardise(frames: dict[str, pd.DataFrame], src: dict) -> list[dict]:
    action_intents = set(src["action_intents"])
    terminal_intents = set(src["terminal_intents"])
    records = []
    for hf_split, df in frames.items():
        for index, row in enumerate(df.itertuples(index=False)):
            intents = list(row.dialogue_acts["intent"])
            if not any(intents):
                continue  # no dialogue acts: outcome cannot be derived
            roles = list(row.agent_info["Role"])
            turns = []
            for speaker, intent, text in zip(row.agent_turn, intents, row.utterance):
                if intent in action_intents:
                    if intent in terminal_intents:
                        break
                    continue
                role = "customer" if roles[speaker] == src["customer_role"] else "agent"
                turns.append((role, text))
            records.append({
                "dataset": DATASET,
                "conversation_id": f"{DATASET}-{hf_split}-{index}",
                "outcome": int(src["positive_intent"] in intents),
                "title": str(row.items["Title"][0]),
                "meta": {
                    "category": str(row.items["Category"][0]),
                    "listing_price": float(row.items["Price"][0]),
                },
                "is_synthetic": False,
                "turns": turns,
            })
    return records


def load(raw_dir: Path, cfg: dict) -> list[dict]:
    return standardise(read_raw(raw_dir), cfg["data"]["sources"][DATASET])
