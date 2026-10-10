"""Gemini jobs: cue tagging, v2 rubric scores and turning-point rationales.

Run: python -m inclination.llm_jobs tag | score | rationales | all   [--limit N]

Each job writes a file in outputs/ that `export` picks up. Jobs are resumable: answers are
cached, so a second run only pays for what is missing.
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd
from pydantic import BaseModel

from inclination import annotation, llm
from inclination.config import load_config, path
from inclination.cues import llm_tagger
from inclination.scoring import llm_rubric

RATIONALE_JOB = "rationale"
CUES_FILE, SCORES_FILE, RATIONALES_FILE = "cues_llm.parquet", "v2_scores.parquet", "rationales_llm.parquet"


class Rationale(BaseModel):
    rationale: str


def _conversations(turns: pd.DataFrame, ids) -> dict:
    wanted = turns[turns["conversation_id"].isin(set(ids))].sort_values(["conversation_id", "turn"])
    return {cid: rows for cid, rows in wanted.groupby("conversation_id", sort=False)}


def tagging_sample(leads: pd.DataFrame, cfg: dict) -> dict[str, list[str]]:
    """`llm.tag_sample_size` conversations per dataset: test split first, then val, then train."""
    rng = np.random.default_rng(cfg["seed"])
    rank = {"test": 0, "val": 1, "train": 2}
    sample = {}
    for dataset in cfg["data"]["datasets"]:
        pool = leads[leads["dataset"] == dataset].drop_duplicates("conversation_id")
        pool = pool.sort_values("conversation_id")
        order = np.lexsort((rng.random(len(pool)), pool["split"].map(rank).to_numpy()))
        sample[dataset] = pool["conversation_id"].to_numpy()[order][: cfg["llm"]["tag_sample_size"]].tolist()
    return sample


def scoring_sample(leads: pd.DataFrame, cfg: dict) -> dict[str, list[str]]:
    """`llm.score_sample_size` test conversations per dataset. The annotated conversations come
    first, so v2 can also be compared with the labels."""
    rng = np.random.default_rng(cfg["seed"])
    annotated = annotation.ensure_sample(leads.drop_duplicates("conversation_id"), cfg)
    sample = {}
    for dataset in cfg["data"]["datasets"]:
        pool = leads[(leads["dataset"] == dataset) & (leads["split"] == "test")]
        pool = sorted(set(pool["conversation_id"]) - set(annotated[dataset]))
        extra = rng.permutation(pool).tolist()
        sample[dataset] = (annotated[dataset] + extra)[: cfg["llm"]["score_sample_size"]]
    return sample


def run_tagging(client, turns, leads, cfg, limit=None) -> None:
    rows = []
    try:
        for dataset, ids in tagging_sample(leads, cfg).items():
            conversations = _conversations(turns, ids[:limit])
            for done, (conversation_id, part) in enumerate(conversations.items(), start=1):
                roles = part["role"].tolist()
                cues, _ = llm_tagger.tag_conversation(client, roles, part["text"].tolist())
                if cues is not None:
                    rows += [(dataset, conversation_id, turn, role, tags, "llm")
                             for turn, (role, tags) in enumerate(zip(roles, cues))]
                if done % 50 == 0:
                    print(f"  tagging {dataset}: {done}/{len(conversations)}", flush=True)
    except llm.LLMStopped as stop:
        print(f"  tagging stopped early: {stop}", flush=True)
    frame = pd.DataFrame(rows, columns=["dataset", "conversation_id", "turn", "role", "cues", "tagger"])
    frame.to_parquet(path("outputs") / CUES_FILE, index=False)
    print(f"  wrote {CUES_FILE}: {frame['conversation_id'].nunique()} conversations, {len(frame):,} turns")


def run_scoring(client, turns, leads, cfg, limit=None) -> None:
    rows = []
    try:
        for dataset, ids in scoring_sample(leads, cfg).items():
            conversations = _conversations(turns, ids[:limit])
            for done, conversation_id in enumerate(ids[:limit], start=1):
                part = conversations[conversation_id]
                results = llm_rubric.score_conversation(
                    client, dataset, part["role"].tolist(), part["text"].tolist()
                )
                rows += [(dataset, conversation_id, turn, score, reason)
                         for turn, (score, reason) in enumerate(results)]
                if done % 5 == 0:
                    print(f"  v2 scoring {dataset}: {done}/{len(ids[:limit])}", flush=True)
    except llm.LLMStopped as stop:  # a conversation cut short is left out; its calls stay cached
        print(f"  v2 scoring stopped early: {stop}", flush=True)
    frame = pd.DataFrame(rows, columns=["dataset", "conversation_id", "turn", "score", "reason"])
    frame.to_parquet(path("outputs") / SCORES_FILE, index=False)
    print(f"  wrote {SCORES_FILE}: {frame['conversation_id'].nunique()} conversations, {len(frame):,} turns")


def rationale_prompt(roles, texts, turn: int, direction: str, delta: float, max_words: int) -> str:
    transcript = "\n".join(f"[{k}] {roles[k].upper()}: {texts[k]}" for k in range(turn + 1))
    verb = "rose" if direction == "rise" else "dropped"
    return (
        "In this conversation an AGENT tries to get a CUSTOMER to say yes (buy, accept a deal, "
        f"donate or sign up). A model's estimate that the customer will say yes {verb} by "
        f"{abs(delta) * 100:.0f} points at turn {turn}.\n\n"
        f"Write one sentence of at most {max_words} words that explains what in turn {turn}, and "
        "in the agent turn just before it if relevant, most plausibly caused this. Start with "
        f"\"Interest {verb}\". Refer to what was said. Do not mention scores, points or models.\n\n"
        f"Conversation up to turn {turn}:\n{transcript}"
    )


def run_rationales(client, turns, leads, cfg, limit=None) -> None:
    """Rationales for method A turning points of v1 and v3 in the v2 sample conversations."""
    ids = [cid for dataset_ids in scoring_sample(leads, cfg).values() for cid in dataset_ids[:limit]]
    points = pd.read_parquet(path("outputs") / "turning_points.parquet")
    points = points[points["conversation_id"].isin(set(ids)) & (points["method"] == "delta")
                    & points["model"].isin(["v1", "v3"])]
    conversations = _conversations(turns, ids)
    rows = []
    try:
        for done, point in enumerate(points.itertuples(), start=1):
            part = conversations[point.conversation_id]
            prompt = rationale_prompt(part["role"].tolist(), part["text"].tolist(), int(point.turn),
                                      point.direction, point.delta, cfg["llm"]["rationale_max_words"])
            parsed, _ = client.generate(prompt, cfg["llm"]["model"], Rationale, RATIONALE_JOB)
            if parsed and parsed.rationale.strip():
                rows.append((point.model, point.conversation_id, int(point.turn), point.method,
                             parsed.rationale.strip()))
            if done % 25 == 0:
                print(f"  rationales: {done}/{len(points)}", flush=True)
    except llm.LLMStopped as stop:
        print(f"  rationales stopped early: {stop}", flush=True)
    frame = pd.DataFrame(rows, columns=["model", "conversation_id", "turn", "method", "rationale_llm"])
    frame.to_parquet(path("outputs") / RATIONALES_FILE, index=False)
    print(f"  wrote {RATIONALES_FILE}: {len(frame)} of {len(points)} turning points")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", choices=["tag", "score", "rationales", "all"])
    parser.add_argument("--limit", type=int, default=None, help="conversations per dataset (for a trial run)")
    args = parser.parse_args()
    if not llm.available():
        print("GEMINI_API_KEY is not set: skipping the LLM jobs (rule-based versions stay in use).")
        return
    cfg = load_config()
    turns = pd.read_parquet(path("outputs") / "turns.parquet")
    leads = pd.read_parquet(path("outputs") / "leads.parquet")
    jobs = {"tag": run_tagging, "score": run_scoring, "rationales": run_rationales}
    for name in (jobs if args.job == "all" else [args.job]):
        print(f"LLM job: {name}", flush=True)
        client = llm.GeminiClient(cfg)  # each job gets its own call budget
        jobs[name](client, turns, leads, cfg, args.limit)
        print(client.summary(), flush=True)
    summary = {"finished": True}
    (path("outputs") / "llm_jobs_done.json").write_text(json.dumps(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
