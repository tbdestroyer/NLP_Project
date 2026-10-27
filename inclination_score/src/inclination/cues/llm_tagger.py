"""LLM cue tagger: one Gemini call per conversation, same output format as the rule tagger.

Run a smoke test with: python -m inclination.cues.llm_tagger --smoke 5
"""
import argparse
import sys

from pydantic import BaseModel

from inclination import llm
from inclination.config import load_config
from inclination.cues.taxonomy import AGENT_MOVE_NAMES, CUSTOMER_CUE_NAMES, DESCRIPTIONS, names_for_role

JOB = "cue_tagging"


class TurnCues(BaseModel):
    turn: int
    role: str        # echoed back so the model commits to the speaker before choosing names
    cues: list[str]


class ConversationCues(BaseModel):
    turns: list[TurnCues]


def _definitions(names: list[str]) -> str:
    return "\n".join(f"- {name}: {DESCRIPTIONS[name]}" for name in names)


def build_prompt(roles: list[str], texts: list[str]) -> str:
    transcript = "\n".join(
        f"[{turn}] {role.upper()}: {text}" for turn, (role, text) in enumerate(zip(roles, texts))
    )
    return (
        "You label turns of a conversation in which an AGENT tries to get a CUSTOMER to say yes "
        "(buy an item, accept a deal, donate, or sign up).\n\n"
        "For each CUSTOMER turn, list every customer cue that applies:\n"
        f"{_definitions(CUSTOMER_CUE_NAMES)}\n\n"
        "For each AGENT turn, list every agent move that applies:\n"
        f"{_definitions(AGENT_MOVE_NAMES)}\n\n"
        "Rules:\n"
        "- Use only the names above, spelled exactly. Customer cues only on CUSTOMER turns, "
        "agent moves only on AGENT turns.\n"
        "- Label AGENT turns as carefully as CUSTOMER turns. Most agent turns contain at least one "
        "agent move, for example states_price when a price or amount is given, asks_question when "
        "the agent asks something, builds_rapport for a greeting or thanks.\n"
        "- A turn can have several names. Use an empty list only when truly nothing applies.\n"
        "- Judge each turn by its meaning in context, not by keywords.\n"
        "- Return one entry for every turn number, in order, with its role "
        "(\"customer\" or \"agent\").\n\n"
        f"Conversation:\n{transcript}"
    )


def clean(parsed: ConversationCues | None, roles: list[str]) -> list[list[str]] | None:
    """Per-turn cue lists with unknown names and wrong-role names dropped; None if unusable."""
    if parsed is None:
        return None
    result = [[] for _ in roles]
    for entry in parsed.turns:
        if 0 <= entry.turn < len(roles):
            allowed = names_for_role(roles[entry.turn])
            result[entry.turn] = [name for name in allowed if name in set(entry.cues)]
    return result


def tag_conversation(client: llm.GeminiClient, roles: list[str], texts: list[str]):
    """Returns (cleaned per-turn cue lists or None, raw response text)."""
    cfg = load_config()["llm"]
    parsed, raw = client.generate(build_prompt(roles, texts), cfg["tagging_model"], ConversationCues, JOB)
    return clean(parsed, roles), raw


def _smoke(n: int) -> None:
    import pandas as pd

    from inclination.config import path

    cfg = load_config()
    leads = pd.read_parquet(path("outputs") / "leads.parquet")
    leads = leads[(leads["split"] == "test") & (leads["model"] == "v1")].sort_values("conversation_id")
    turns = pd.read_parquet(path("outputs") / "turns.parquet")
    per_dataset = -(-n // len(cfg["data"]["datasets"]))
    ids = [cid for d in cfg["data"]["datasets"]
           for cid in leads.loc[leads["dataset"] == d, "conversation_id"].head(per_dataset)][:n]
    client = llm.GeminiClient(max_calls=n)
    for conversation_id in ids:
        rows = turns[turns["conversation_id"] == conversation_id].sort_values("turn")
        roles, texts = rows["role"].tolist(), rows["text"].tolist()
        cleaned, raw = tag_conversation(client, roles, texts)
        print(f"\n=== {conversation_id} ({len(roles)} turns) ===")
        for turn, (role, text) in enumerate(zip(roles, texts)):
            print(f"[{turn}] {role}: {text[:110]}")
        print("RAW JSON:", raw)
        print("AFTER VALIDATION:", "unusable response" if cleaned is None else cleaned)
    print("\n" + client.summary())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", type=int, required=True, help="number of conversations to tag")
    if not llm.available():
        sys.exit("GEMINI_API_KEY is not set (put it in .env).")
    _smoke(parser.parse_args().smoke)
