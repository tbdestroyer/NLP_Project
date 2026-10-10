"""Writes every file the dashboard reads into outputs/. Run: python -m inclination.export"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd

from inclination import annotation, explain, insights
from inclination.config import ROOT, load_config, path
from inclination.cues import rules
from inclination.cues.taxonomy import AGENT_MOVE_NAMES, CUSTOMER_CUE_NAMES
from inclination.data import load_processed
from inclination.scoring.cue_model import _runs
from inclination.train import model_path, prepare
from inclination.turning_points import delta_turning_points, pelt_turning_points

N_CUSTOMER = len(CUSTOMER_CUE_NAMES)


def trend_slope(scores: np.ndarray, window: int) -> float:
    """Least-squares slope of the last `window` scores (0 for a single turn)."""
    tail = scores[-window:]
    if len(tail) < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(tail)), tail, 1)[0])


def lead_status(final: float, trend: float, cfg: dict) -> str:
    if final >= cfg["hot_final_score"]:
        return "Hot"
    if trend > cfg["rising_trend"]:
        return "Rising"
    if trend < cfg["cooling_trend"]:
        return "Cooling"
    return "Stalled"


def status_reason(status: str, last_point) -> str:
    if last_point is None:
        return "no sharp changes"
    agrees = (status == "Cooling" and last_point.direction == "drop") or (
        status in ("Hot", "Rising") and last_point.direction == "rise"
    )
    if agrees:
        return f"{status.lower()} after {last_point.trigger}"
    return f"last sharp {last_point.direction} after {last_point.trigger}"


class V1Backend:
    """v1 cue model: scores every conversation; explanations use its coefficients."""
    name = "v1"

    def __init__(self, dataset: str, cfg: dict):
        self.model = joblib.load(model_path("v1", dataset))
        self.top_k = cfg["model"]["v1"]["top_features"]

    @staticmethod
    def available(dataset: str) -> bool:
        return model_path("v1", dataset).exists()

    def select(self, part: pd.DataFrame) -> np.ndarray:
        return np.ones(len(part), dtype=bool)

    def score(self, part: pd.DataFrame, cue_matrix: np.ndarray) -> np.ndarray:
        self.per_turn = self.model.per_turn(part["role"], part["text"], cue_matrix)
        self.X = self.model.features(self.per_turn, part["group"].to_numpy())
        return self.model.predict(self.X)

    def score_variants(self, rows: np.ndarray, groups: np.ndarray) -> np.ndarray:
        return self.model.score_last(self.per_turn.take(rows), groups)

    def final_raw(self, part: pd.DataFrame, cue_matrix: np.ndarray) -> np.ndarray:
        """Uncalibrated final-turn score of every conversation in `part` (any dataset)."""
        per_turn = self.model.per_turn(part["role"], part["text"], cue_matrix)
        X = self.model.features(per_turn, part["group"].to_numpy())
        last = (part["position"] == part["size"] - 1).to_numpy()
        return self.model.predict_raw(X[last])

    def top_features(self, rows: np.ndarray) -> list[list[dict]]:
        return self.model.top_contributions(self.X, rows, self.top_k)


class V3Backend:
    """v3 transformer: scores only the splits in config `model.v3.score_splits`."""
    name = "v3"

    def __init__(self, dataset: str, cfg: dict):
        from inclination.scoring.transformer import TransformerScorer  # needs torch

        self.dataset = dataset
        self.scorer = TransformerScorer.load(dataset, cfg)
        self.splits = cfg["model"]["v3"]["score_splits"]

    @staticmethod
    def available(dataset: str) -> bool:
        folder = ROOT / load_config()["model"]["dir"] / f"v3_{dataset}"
        return (folder / "meta.json").exists() and (folder / "calibrator.joblib").exists()

    def select(self, part: pd.DataFrame) -> np.ndarray:
        return part["split"].isin(self.splits).to_numpy()

    def score(self, part: pd.DataFrame, cue_matrix: np.ndarray) -> np.ndarray:
        self.turn_ids = self.scorer.encode_turns(part["role"], part["text"])
        sequences = self.scorer.prefix_sequences(self.turn_ids, part["group"].to_numpy())
        return self.scorer.predict(sequences, progress=f"{self.dataset} v3 scores")

    def score_variants(self, rows: np.ndarray, groups: np.ndarray) -> np.ndarray:
        starts, sizes, _ = _runs(groups)
        sequences = [
            self.scorer.pack([self.turn_ids[r] for r in rows[start : start + size]])
            for start, size in zip(starts, sizes)
        ]
        return self.scorer.predict(sequences, progress=f"{self.dataset} v3 faithfulness")

    def top_features(self, rows: np.ndarray) -> list[list[dict]]:
        return [[] for _ in rows]  # no per-feature attribution for the transformer

    def final_raw(self, part: pd.DataFrame, cue_matrix: np.ndarray) -> np.ndarray:
        """Uncalibrated final-turn score of every conversation in `part` (any dataset)."""
        turn_ids = self.scorer.encode_turns(part["role"], part["text"])
        starts, sizes, _ = _runs(part["group"].to_numpy())
        sequences = [self.scorer.pack(turn_ids[s : s + n]) for s, n in zip(starts, sizes)]
        return self.scorer.predict_raw(sequences)


class V2Backend:
    """v2 Gemini rubric: reads the scores written by `inclination.llm_jobs score`. No LLM is
    called here. Scores are uncalibrated and cover a small test sample only."""
    name = "v2"
    supports_removal = False  # faithfulness would need a new LLM call per turning point

    def __init__(self, dataset: str, cfg: dict):
        scores = pd.read_parquet(path("outputs") / V2_SCORES_FILE)
        self.scores = scores[scores["dataset"] == dataset]

    @staticmethod
    def available(dataset: str) -> bool:
        target = path("outputs") / V2_SCORES_FILE
        return target.exists() and (pd.read_parquet(target, columns=["dataset"])["dataset"] == dataset).any()

    def select(self, part: pd.DataFrame) -> np.ndarray:
        return part["conversation_id"].isin(set(self.scores["conversation_id"])).to_numpy()

    def score(self, part: pd.DataFrame, cue_matrix: np.ndarray) -> np.ndarray:
        merged = part[["conversation_id", "turn"]].merge(
            self.scores, on=["conversation_id", "turn"], how="left", validate="one_to_one"
        )
        self.reasons = merged["reason"].fillna("").to_numpy()
        return merged["score"].to_numpy()

    def top_features(self, rows: np.ndarray) -> list[list[dict]]:
        return [[] for _ in rows]


V1Backend.supports_removal = V3Backend.supports_removal = True
BACKENDS = {"v1": V1Backend, "v2": V2Backend, "v3": V3Backend}
V2_SCORES_FILE, LLM_CUES_FILE, LLM_RATIONALES_FILE = "v2_scores.parquet", "cues_llm.parquet", "rationales_llm.parquet"
MODEL_KEYED_FILES = ["scores.parquet", "turning_points.parquet", "leads.parquet", "insights.parquet",
                     "insight_examples.parquet", "cross_scores.parquet"]


def removal_effect(rows, backend, score, sizes, position) -> np.ndarray:
    """Score change at the evaluation point when each of `rows` is removed and re-scored."""
    if len(rows) == 0:
        return np.zeros(0)
    variant_rows, variant_groups, evaluation_rows = explain.removal_variants(rows, None, sizes, position)
    return score[evaluation_rows] - backend.score_variants(variant_rows, variant_groups)


def explain_points(points: pd.DataFrame, part: pd.DataFrame, tags, backend, cfg: dict,
                   with_faithfulness: bool = True):
    """Adds cues, rationale, top features and faithfulness to one dataset's turning points.
    `points` is indexed by row number in `part`. Faithfulness (and its random-turn baseline) is
    computed for method A only; pass with_faithfulness=False to leave it empty."""
    rows = points.index.to_numpy()
    roles = part["role"].to_numpy()
    position = part["position"].to_numpy()
    previous_agent = explain.previous_agent_rows(roles, position)

    rationales, triggers = [], []
    for row, direction in zip(rows, points["direction"]):
        previous = previous_agent[row]
        sentence, trigger = explain.template_rationale(
            direction, roles[row], tags[row], tags[previous] if previous >= 0 else []
        )
        rationales.append(sentence)
        triggers.append(trigger)

    sizes = part["size"].to_numpy()
    score = part["score"].to_numpy()
    faithfulness = np.full(len(rows), np.nan)
    faithfulness_random = np.full(len(rows), np.nan)
    top = [[] for _ in rows]
    if with_faithfulness and len(rows):
        faithfulness = removal_effect(rows, backend, score, sizes, position)
        top = backend.top_features(rows)
        # Baseline: remove one random other turn (never turn 0) of the same conversation.
        rng = np.random.default_rng(cfg["seed"])
        has_other = sizes[rows] > 2
        offsets = rng.integers(1, np.maximum(sizes[rows] - 1, 2))  # 1 .. size-2
        other_position = np.where(offsets >= position[rows], offsets + 1, offsets)
        random_rows = (rows - position[rows] + other_position)[has_other]
        faithfulness_random[has_other] = removal_effect(random_rows, backend, score, sizes, position)

    points = points.copy()
    points["cues"] = [list(tags[row]) for row in rows]
    points["rationale"] = rationales
    points["faithfulness"] = faithfulness
    points["faithfulness_random"] = faithfulness_random
    points["weak"] = np.abs(faithfulness) < cfg["turning"]["weak_faithfulness_threshold"]
    points["trigger"] = triggers
    points["role"] = roles[rows]
    points["prev_agent_turn"] = [
        int(part["turn"].iat[previous_agent[row]]) if previous_agent[row] >= 0 else -1 for row in rows
    ]
    points["top_features"] = [json.dumps(features) for features in top]
    return points


def build_leads(part: pd.DataFrame, points: pd.DataFrame, conversations: pd.DataFrame, cfg: dict):
    status_cfg = cfg["status"]
    last_point = {p.conversation_id: p for p in points.itertuples(index=False)}  # last wins
    n_points = points.groupby("conversation_id").size().to_dict()
    rows = []
    for conversation_id, scores in part.groupby("conversation_id", sort=False)["score"]:
        values = scores.to_numpy()
        trend = trend_slope(values, status_cfg["trend_window"])
        status = lead_status(values[-1], trend, status_cfg)
        rows.append({
            "conversation_id": conversation_id,
            "final_score": values[-1],
            "max_score": values.max(),
            "trend": trend,
            "status": status,
            "status_reason": status_reason(status, last_point.get(conversation_id)),
            "n_turning_points": n_points.get(conversation_id, 0),
        })
    return pd.DataFrame(rows).merge(conversations, on="conversation_id")


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(BACKENDS), default=list(BACKENDS),
                        help="models to export; a model without trained files is skipped")
    parser.add_argument("--append", action="store_true",
                        help="keep the rows of models not listed in --models instead of dropping them")
    args = parser.parse_args()
    models, append = args.models, args.append
    turns, conversations = load_processed()
    out = path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    all_scores, all_cues, all_points, all_leads, all_effects, all_examples = [], [], [], [], [], []
    test_parts = {}

    for dataset in cfg["data"]["datasets"]:
        part = prepare(turns, conversations, dataset)
        groups = part["group"].to_numpy()
        _, sizes, position = _runs(groups)
        part["position"], part["size"] = position, np.repeat(sizes, sizes)

        tags = rules.tag_turns(part["role"], part["text"])
        cue_matrix = rules.tags_to_matrix(tags)
        all_cues.append(part[["dataset", "conversation_id", "turn", "role"]].assign(
            cues=tags, tagger="rules"
        ))

        for name in models:
            if not BACKENDS[name].available(dataset):
                print(f"{dataset} {name}: no trained model, skipped")
                continue
            backend = BACKENDS[name](dataset, cfg)
            keep = backend.select(part)
            sub = part[keep].reset_index(drop=True)
            sub_tags = [tags[row] for row in np.flatnonzero(keep)]
            sub_matrix = cue_matrix[keep]
            sub["score"] = backend.score(sub, sub_matrix)
            sub["model"] = name
            all_scores.append(sub[["dataset", "conversation_id", "turn", "model", "score"]])

            turning = cfg["turning"]
            points = delta_turning_points(sub, turning["delta_threshold"])
            points = explain_points(points, sub, sub_tags, backend, cfg,
                                    with_faithfulness=backend.supports_removal)
            change_points = pelt_turning_points(sub, turning["pelt_penalty"], turning["pelt_min_size"])
            change_points = explain_points(change_points, sub, sub_tags, backend, cfg,
                                           with_faithfulness=False)
            for frame in (points, change_points):
                # v2 already has an LLM-written reason for every turn; others are merged later.
                reasons = getattr(backend, "reasons", None)
                frame["rationale_llm"] = None if reasons is None else reasons[frame.index.to_numpy()]
            all_points += [points, change_points]

            leads = build_leads(sub, points, conversations, cfg)
            all_leads.append(leads.assign(model=name))

            sub["in_scope"] = sub["split"] == "test"
            effects, examples = insights.move_effects(
                sub, sub_matrix[:, N_CUSTOMER:], cfg["insights"], cfg["seed"]
            )
            for frame, bucket in ((effects, all_effects), (examples, all_examples)):
                frame.insert(0, "model", name)
                frame.insert(0, "dataset", dataset)
                bucket.append(frame)
            weak = points["weak"].mean() if len(points) else 0.0
            print(f"{dataset} {name}: {len(sub):,} scores, {len(points):,} turning points "
                  f"({weak:.0%} weak), {len(change_points):,} change points, "
                  f"{len(effects)} insight rows", flush=True)
            del backend
        test = part[part["split"] == "test"].reset_index(drop=True)
        test_parts[dataset] = (test, cue_matrix[(part["split"] == "test").to_numpy()])

    # Cross-domain: each trained model scores the final turn of every dataset's test split.
    cross = []
    for name in models:
        for trained_on in cfg["data"]["datasets"]:
            if not hasattr(BACKENDS[name], "final_raw") or not BACKENDS[name].available(trained_on):
                continue
            backend = BACKENDS[name](trained_on, cfg)
            for tested_on, (test, test_matrix) in test_parts.items():
                last = test.groupby("conversation_id", sort=False).tail(1)
                cross.append(pd.DataFrame({
                    "model": name, "train_dataset": trained_on, "test_dataset": tested_on,
                    "conversation_id": last["conversation_id"].to_numpy(),
                    "score": backend.final_raw(test, test_matrix),
                }))
            print(f"cross-domain {name} trained on {trained_on}: done", flush=True)
            del backend

    def combine(frames: list) -> pd.DataFrame:
        frames = [frame for frame in frames if len(frame)]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # Gemini cue tags (if the tagging job has run) sit next to the rule tags, marked by `tagger`.
    if (out / LLM_CUES_FILE).exists():
        all_cues.append(pd.read_parquet(out / LLM_CUES_FILE))
    files = {
        "turns.parquet": turns,
        "scores.parquet": combine(all_scores),
        "cues.parquet": combine(all_cues),
        "turning_points.parquet": combine(all_points),
        "leads.parquet": combine(all_leads),
        "insights.parquet": combine(all_effects),
        "insight_examples.parquet": combine(all_examples),
        "cross_scores.parquet": combine(cross),
    }
    for name, frame in files.items():
        if append and name in MODEL_KEYED_FILES and (out / name).exists():
            # Keep the other models' rows; replace only the rows of the models exported now.
            kept = pd.read_parquet(out / name)
            frame = combine([kept[~kept["model"].isin(models)], frame])
        if frame.empty:
            continue
        if name == "turning_points.parquet" and (out / LLM_RATIONALES_FILE).exists():
            # Gemini-written rationales for v1 / v3 turning points, where the job produced one.
            written = pd.read_parquet(out / LLM_RATIONALES_FILE).rename(columns={"rationale_llm": "written"})
            frame = frame.merge(written, on=["model", "conversation_id", "turn", "method"], how="left")
            frame["rationale_llm"] = frame["written"].fillna(frame["rationale_llm"])
            frame = frame.drop(columns="written")
        frame.to_parquet(out / name, index=False)
        print(f"  wrote {name}: {len(frame):,} rows")
    assert set(AGENT_MOVE_NAMES) >= set(files["insights.parquet"].get("move", []))
    sample = annotation.ensure_sample(conversations, cfg)
    print(f"  annotation sample: {sum(map(len, sample.values()))} conversations "
          f"({annotation.sample_path().name})")


if __name__ == "__main__":
    main()
