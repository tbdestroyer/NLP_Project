"""Evaluation on the test split. Reads outputs/ and annotation/labels/, writes
outputs/metrics.json, outputs/figures/*.png and eval/report.md. Run: python eval/evaluate.py
"""
import json
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    roc_auc_score,
)

from inclination import annotation
from inclination.config import ROOT, load_config, path
from inclination.cues.taxonomy import ALL_NAMES, CUSTOMER_CUE_NAMES

INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
DIRECTION_OF_LABEL = {"more": "up", "same": "same", "less": "down"}
METHOD_LABELS = {"delta": "A: score change", "pelt": "B: change point"}


# ---- outcome prediction ---------------------------------------------------------------------

def reliability_bins(labels: np.ndarray, scores: np.ndarray, n_bins: int) -> list[dict]:
    bins = np.minimum((scores * n_bins).astype(int), n_bins - 1)
    return [
        {"mean_score": float(scores[bins == b].mean()), "observed_rate": float(labels[bins == b].mean()),
         "n": int((bins == b).sum())}
        for b in range(n_bins) if (bins == b).any()
    ]


def expected_calibration_error(bins: list[dict]) -> float:
    total = sum(b["n"] for b in bins)
    return float(sum(b["n"] / total * abs(b["observed_rate"] - b["mean_score"]) for b in bins))


def auc_with_ci(labels: np.ndarray, scores: np.ndarray, n_samples: int, level: float, rng):
    auc = roc_auc_score(labels, scores)
    samples = []
    while len(samples) < n_samples:
        index = rng.integers(0, len(labels), len(labels))
        if labels[index].min() != labels[index].max():
            samples.append(roc_auc_score(labels[index], scores[index]))
    low, high = np.quantile(samples, [(1 - level) / 2, (1 + level) / 2])
    return float(auc), float(low), float(high)


def scores_at_fraction(scores: pd.DataFrame, fraction: float) -> pd.Series:
    """Score at the turn that closes the first `fraction` of each conversation."""
    index = np.maximum(np.ceil(fraction * scores["n_turns"]).astype(int) - 1, 0)
    return scores[scores["turn"] == index].set_index("conversation_id")["score"]


def evaluate_outcome(scores: pd.DataFrame, leads: pd.DataFrame, cfg: dict, rng) -> dict:
    """`scores`: test-split turn scores with n_turns; `leads`: matching test-split leads."""
    eval_cfg = cfg["eval"]
    labels = leads["outcome"].to_numpy()
    final = leads["final_score"].to_numpy()
    base_rate = float(labels.mean())
    auc, low, high = auc_with_ci(
        labels, final, eval_cfg["bootstrap_samples"], eval_cfg["ci_level"], rng
    )
    outcome = leads.set_index("conversation_id")["outcome"]
    early = {}
    for fraction in eval_cfg["early_fractions"]:
        at = scores_at_fraction(scores, fraction)
        early[f"{fraction:.2f}"] = float(roc_auc_score(outcome.loc[at.index], at))
    ranked = leads.sort_values("final_score", ascending=False)["outcome"].to_numpy()
    bins = reliability_bins(labels, final, eval_cfg["calibration_bins"])
    return {
        "n_conversations": int(len(leads)),
        "base_rate": base_rate,
        "roc_auc": auc,
        "roc_auc_ci": [low, high],
        "pr_auc": float(average_precision_score(labels, final)),
        "brier": float(brier_score_loss(labels, final)),
        "ece": expected_calibration_error(bins),
        "reliability": bins,
        "baseline_majority": {
            "accuracy": max(base_rate, 1 - base_rate),
            "roc_auc": 0.5,
            "pr_auc": base_rate,
            "brier": base_rate * (1 - base_rate),  # always predicting the base rate
        },
        "early_auc": early,
        "precision_at": {str(k): float(ranked[:k].mean()) for k in eval_cfg["precision_at"]},
    }


# ---- turning points, faithfulness, cross-domain ---------------------------------------------

def evaluate_faithfulness(points: pd.DataFrame) -> list[dict]:
    """Method A turning points of the test split: |score change| when the trigger turn is
    removed, against removing a random other turn of the same conversation."""
    rows = []
    paired = points[(points["method"] == "delta") & points["faithfulness_random"].notna()]
    for (dataset, model), group in paired.groupby(["dataset", "model"], sort=False):
        trigger, random = group["faithfulness"].abs(), group["faithfulness_random"].abs()
        rows.append({
            "dataset": dataset, "model": model, "n": int(len(group)),
            "trigger_removed": float(trigger.mean()), "random_removed": float(random.mean()),
            "ratio": float(trigger.mean() / random.mean()) if random.mean() > 0 else None,
            "trigger_larger_share": float((trigger > random).mean()),
            "weak_share": float(group["weak"].mean()),
        })
    return rows


def count_turning_points(points: pd.DataFrame, leads: pd.DataFrame) -> list[dict]:
    rows = []
    n_conversations = leads.groupby(["dataset", "model"]).size()
    for (dataset, model, method), group in points.groupby(["dataset", "model", "method"], sort=False):
        total = int(n_conversations[(dataset, model)])
        rows.append({
            "dataset": dataset, "model": model, "method": method, "points": int(len(group)),
            "per_conversation": len(group) / total,
            "conversations_with_one": group["conversation_id"].nunique() / total,
            "rise_share": float((group["direction"] == "rise").mean()),
        })
    return rows


def evaluate_cross_domain(cross: pd.DataFrame, outcomes: pd.Series) -> list[dict]:
    cross = cross.assign(outcome=cross["conversation_id"].map(outcomes))
    return [
        {"model": model, "train_dataset": train, "test_dataset": test,
         "roc_auc": float(roc_auc_score(group["outcome"], group["score"])), "n": int(len(group))}
        for (model, train, test), group in cross.groupby(["model", "train_dataset", "test_dataset"], sort=False)
    ]


def evaluate_taggers(cues: pd.DataFrame) -> dict:
    """Rule tagger against the Gemini tagger, per cue, on the turns both have tagged."""
    by_tagger = {tagger: frame.set_index(["conversation_id", "turn"]) for tagger, frame in cues.groupby("tagger")}
    if not {"rules", "llm"} <= set(by_tagger):
        return {"n_turns": 0, "cues": []}
    both = by_tagger["rules"].join(by_tagger["llm"]["cues"].rename("llm_cues"), how="inner")
    rows = []
    for name in ALL_NAMES:
        role = "customer" if name in CUSTOMER_CUE_NAMES else "agent"
        part = both[both["role"] == role]
        rule = part["cues"].map(lambda tags: name in tags).to_numpy()
        model = part["llm_cues"].map(lambda tags: name in tags).to_numpy()
        kappa = cohen_kappa_score(rule, model) if rule.any() or model.any() else None
        rows.append({
            "cue": name, "role": role, "n_turns": int(len(part)),
            "rules": int(rule.sum()), "llm": int(model.sum()), "both": int((rule & model).sum()),
            "agreement": float((rule == model).mean()),
            "kappa": None if kappa is None or np.isnan(kappa) else float(kappa),
        })
    return {"n_turns": int(len(both)), "conversations": int(both.reset_index()["conversation_id"].nunique()),
            "cues": rows}


# ---- agreement with annotators --------------------------------------------------------------

def annotator_kind(name: str) -> str:
    return "LLM" if name.endswith("_llm") else "human"


def label_frame(records: dict) -> pd.DataFrame:
    rows = [
        {"conversation_id": cid, "dataset": record["dataset"], **label}
        for cid, record in records.items() for label in record["labels"]
    ]
    return pd.DataFrame(rows)


def model_directions(scores: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    """Direction of the score at each turn >= 1: up / same / down."""
    scores = scores.sort_values(["conversation_id", "turn"]).copy()
    change = scores.groupby("conversation_id")["score"].diff()
    scores["predicted"] = np.where(change > epsilon, "up", np.where(change < -epsilon, "down", "same"))
    return scores.loc[scores["turn"] > 0, ["conversation_id", "turn", "predicted"]]


def evaluate_directions(labels: pd.DataFrame, scores: pd.DataFrame, epsilon: float) -> list[dict]:
    """Customer turns after turn 0 (turn 0 has no previous score to compare with)."""
    customer = labels[(labels["role"] == "customer") & labels["direction"].notna()]
    merged = customer.merge(model_directions(scores, epsilon), on=["conversation_id", "turn"])
    merged["human"] = merged["direction"].map(DIRECTION_OF_LABEL)
    rows = []
    for dataset, group in [("all", merged), *merged.groupby("dataset", sort=False)]:
        if group.empty:
            continue
        rows.append({
            "dataset": dataset, "n_turns": int(len(group)),
            "accuracy": float((group["human"] == group["predicted"]).mean()),
            "always_same_accuracy": float((group["human"] == "same").mean()),
            "kappa": float(cohen_kappa_score(group["human"], group["predicted"])),
            "labelled": group["human"].value_counts().to_dict(),
            "predicted": group["predicted"].value_counts().to_dict(),
        })
    return rows


def match_turning_points(labelled: pd.DataFrame, predicted: pd.DataFrame, tolerance: int) -> dict:
    """Precision and recall of predicted turning points against labelled ones, +/- tolerance."""
    truth = labelled.groupby("conversation_id")["turn"].agg(list).to_dict()
    guess = predicted.groupby("conversation_id")["turn"].agg(list).to_dict()
    near = lambda turn, others: any(abs(turn - other) <= tolerance for other in others)
    hit_predictions = sum(near(t, truth.get(cid, [])) for cid, turns in guess.items() for t in turns)
    hit_labels = sum(near(t, guess.get(cid, [])) for cid, turns in truth.items() for t in turns)
    n_predicted, n_labelled = sum(map(len, guess.values())), sum(map(len, truth.values()))
    precision = hit_predictions / n_predicted if n_predicted else None
    recall = hit_labels / n_labelled if n_labelled else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    return {"n_predicted": n_predicted, "n_labelled": n_labelled,
            "precision": precision, "recall": recall, "f1": f1}


def evaluate_labels(scores: pd.DataFrame, points: pd.DataFrame, cfg: dict) -> dict:
    eval_cfg = cfg["eval"]
    annotators = {name: label_frame(annotation.load_labels(name)) for name in annotation.list_annotators()}
    annotators = {name: frame for name, frame in annotators.items() if len(frame)}
    result = {"annotators": [], "directions": [], "turning_points": [], "between_annotators": []}
    for name, labels in annotators.items():
        kind = annotator_kind(name)
        customer = labels[labels["role"] == "customer"]
        result["annotators"].append({
            "annotator": name, "kind": kind, "conversations": int(labels["conversation_id"].nunique()),
            "customer_turns": int(len(customer)), "turning_points": int(labels["turning_point"].sum()),
            "directions": customer["direction"].value_counts().to_dict(),
        })
        labelled_ids = set(labels["conversation_id"])
        labelled_points = labels[labels["turning_point"]]
        for model, model_scores in scores.groupby("model", sort=False):
            model_scores = model_scores[model_scores["conversation_id"].isin(labelled_ids)]
            # Only conversations this model scored count (v2 covers part of the labelled sample).
            scored_ids = set(model_scores["conversation_id"])
            labelled_points = labels[labels["turning_point"] & labels["conversation_id"].isin(scored_ids)]
            for row in evaluate_directions(labels, model_scores, eval_cfg["direction_epsilon"]):
                result["directions"].append({"annotator": name, "kind": kind, "model": model, **row})
            for method in METHOD_LABELS:
                predicted = points[(points["model"] == model) & (points["method"] == method)
                                   & points["conversation_id"].isin(scored_ids)]
                for dataset in ["all", *labels["dataset"].unique()]:
                    keep = (lambda f: f) if dataset == "all" else (lambda f: f[f["dataset"] == dataset])
                    result["turning_points"].append({
                        "annotator": name, "kind": kind, "model": model, "method": method,
                        "dataset": dataset,
                        **match_turning_points(keep(labelled_points), keep(predicted),
                                               eval_cfg["turning_point_tolerance"]),
                    })
    for first, second in combinations(annotators, 2):
        both = annotators[first].merge(annotators[second], on=["conversation_id", "turn"], suffixes=("_a", "_b"))
        customer = both[both["direction_a"].notna() & both["direction_b"].notna()]
        if customer.empty:
            continue
        result["between_annotators"].append({
            "annotators": [first, second], "kinds": [annotator_kind(first), annotator_kind(second)],
            "conversations": int(both["conversation_id"].nunique()),
            "direction_kappa": float(cohen_kappa_score(customer["direction_a"], customer["direction_b"])),
            "direction_agreement": float((customer["direction_a"] == customer["direction_b"]).mean()),
            "turning_point_kappa": float(cohen_kappa_score(both["turning_point_a"], both["turning_point_b"])),
        })
    return result


# ---- case studies ---------------------------------------------------------------------------

def pick_case_studies(outcome_metrics: dict, leads: pd.DataFrame, points: pd.DataFrame, k: int,
                      min_conversations: int) -> list[dict]:
    """Per dataset, for its best model: confident correct predictions, confident mistakes and
    the largest turning points. Models scored on fewer than `min_conversations` are not
    considered, since their AUC is too uncertain to call them best."""
    cases = []
    for dataset, by_model in outcome_metrics.items():
        eligible = [m for m in by_model if by_model[m]["n_conversations"] >= min_conversations]
        model = max(eligible or by_model, key=lambda m: by_model[m]["roc_auc"])
        mine = leads[(leads["dataset"] == dataset) & (leads["model"] == model)].copy()
        mine["confidence"] = np.where(mine["outcome"] == 1, mine["final_score"], 1 - mine["final_score"])
        ordered = mine.sort_values(["confidence", "conversation_id"])
        picks = [("confident and correct", ordered.tail(k).iloc[::-1]), ("confident mistake", ordered.head(k))]
        for kind, rows in picks:
            for row in rows.itertuples():
                outcome = "converted" if row.outcome == 1 else "did not convert"
                cases.append({
                    "dataset": dataset, "model": model, "kind": kind,
                    "conversation_id": row.conversation_id,
                    "description": f"\"{row.title}\": final score {row.final_score:.0%}, {outcome} "
                                   f"({row.n_turns} turns, {row.n_turning_points} turning points).",
                })
        delta = points[(points["dataset"] == dataset) & (points["model"] == model)
                       & (points["method"] == "delta")
                       & points["conversation_id"].isin(set(mine["conversation_id"]))]
        delta = delta.assign(size=delta["delta"].abs()).sort_values("size", ascending=False)
        delta = delta.drop_duplicates("conversation_id").head(k)
        titles = mine.set_index("conversation_id")["title"]
        for row in delta.itertuples():
            cases.append({
                "dataset": dataset, "model": model, "kind": "largest turning point",
                "conversation_id": row.conversation_id,
                "description": f"\"{titles[row.conversation_id]}\": {row.delta * 100:+.0f} points at "
                               f"turn {row.turn}. {row.rationale}",
            })
    return cases


# ---- figures --------------------------------------------------------------------------------

def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_title(title, loc="left", fontsize=11.5, color=INK, pad=12)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)


def plot_early_prediction(outcome_metrics: dict, model: str, cfg: dict, target) -> None:
    viz = cfg["viz"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURFACE)
    for dataset, by_model in outcome_metrics.items():
        if model not in by_model:
            continue
        early = by_model[model]["early_auc"]
        x, y = [float(f) * 100 for f in early], list(early.values())
        ax.plot(x, y, color=viz["dataset_colors"][dataset], linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=viz["dataset_labels"][dataset])
        ax.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.axhline(0.5, color=AXIS, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate("chance", (100, 0.5), xytext=(8, -11), textcoords="offset points",
                ha="right", fontsize=8.5, color=MUTED)
    ax.set_xticks([float(f) * 100 for f in cfg["eval"]["early_fractions"]])
    ax.set_xlim(5, 108)
    ax.set_ylim(0.4, 1.0)
    _style(ax, f"How early can the {model} score rank buyers above non-buyers?",
           "Share of the conversation seen (%)", "ROC AUC on the test split")
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(target, dpi=viz["figure_dpi"], facecolor=SURFACE)
    plt.close(fig)


def plot_reliability(outcome_metrics: dict, model: str, cfg: dict, target) -> None:
    viz = cfg["viz"]
    fig, ax = plt.subplots(figsize=(5.6, 5.2), facecolor=SURFACE)
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle=(0, (4, 3)), label="perfect calibration")
    min_n = cfg["eval"]["reliability_min_bin"]
    for dataset, by_model in outcome_metrics.items():
        if model not in by_model:
            continue
        bins = [b for b in by_model[model]["reliability"] if b["n"] >= min_n]
        ax.plot([b["mean_score"] for b in bins], [b["observed_rate"] for b in bins],
                color=viz["dataset_colors"][dataset], linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=f"{viz['dataset_labels'][dataset]} (ECE {by_model[model]['ece']:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal")
    _style(ax, f"Reliability of the {model} final-turn score (test split)",
           "Mean predicted score in the bin", "Share that converted")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(target, dpi=viz["figure_dpi"], facecolor=SURFACE)
    plt.close(fig)


# ---- report ---------------------------------------------------------------------------------

def _fmt(value, spec: str = ".3f") -> str:
    return "n/a" if value is None else format(value, spec)


def write_report(metrics: dict, figures: dict, cfg: dict, target) -> None:
    labels = cfg["viz"]["dataset_labels"]
    name = lambda dataset: labels.get(dataset, "All datasets" if dataset == "all" else dataset)
    outcome = metrics["outcome"]
    fractions = [f"{f:.2f}" for f in cfg["eval"]["early_fractions"]]
    ks = [str(k) for k in cfg["eval"]["precision_at"]]
    v2_sizes = sorted({by_model["v2"]["n_conversations"] for by_model in outcome.values() if "v2" in by_model})
    if v2_sizes:
        sizes = " to ".join(str(n) for n in dict.fromkeys([v2_sizes[0], v2_sizes[-1]]))
        v2_note = [
            f"- **v2**: Gemini (`{cfg['llm']['model']}`) scoring the transcript so far against a fixed",
            f"  rubric, one call per turn. Run on only **{sizes} test conversations per dataset**, and",
            "  **not calibrated** (there is no validation sample to fit on). Its intervals are wide and",
            "  its rows are not directly comparable with v1 and v3, which are scored on the full test split.",
        ]
    else:
        v2_note = ["- **v2** (LLM rubric) was not run."]
    out = [
        "# Evaluation report",
        "",
        "Generated by `make eval`. Unless stated otherwise, numbers are on the **test** split and",
        "use the final-turn score, calibrated on the validation split.",
        "",
        "- **v1**: logistic regression on rule-based cue counts and TF-IDF.",
        "- **v3**: fine-tuned `distilroberta-base` (the fallback for ModernBERT-base on a 4 GB GPU).",
        *v2_note,
        "",
        "## 1. Outcome prediction",
        "",
        "| dataset | model | n | base rate | ROC AUC (95% CI) | PR AUC | Brier | Brier, base-rate baseline | ECE |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for dataset, by_model in outcome.items():
        for model, m in by_model.items():
            out.append(
                f"| {name(dataset)} | {model} | {m['n_conversations']:,} | {m['base_rate']:.1%} "
                f"| {m['roc_auc']:.3f} ({m['roc_auc_ci'][0]:.3f}–{m['roc_auc_ci'][1]:.3f}) "
                f"| {m['pr_auc']:.3f} | {m['brier']:.3f} | {m['baseline_majority']['brier']:.3f} "
                f"| {m['ece']:.3f} |"
            )
    out += [
        "",
        "The majority-class baseline has ROC AUC 0.5 and PR AUC equal to the base rate.",
        "",
        "## 2. Early prediction",
        "",
        "ROC AUC using only the score at the given share of each conversation's turns.",
        "",
        "| dataset | model | " + " | ".join(f"{float(f):.0%}" for f in fractions) + " |",
        "|---|---|" + "---|" * len(fractions),
    ]
    for dataset, by_model in outcome.items():
        for model, m in by_model.items():
            out.append(f"| {name(dataset)} | {model} | "
                       + " | ".join(f"{m['early_auc'][f]:.3f}" for f in fractions) + " |")
    for model, figure in figures["early"].items():
        out += ["", f"![Early prediction, {model}]({figure})"]
    out += [
        "",
        "## 3. Lead ranking",
        "",
        "Share of the top-ranked test conversations that converted.",
        "",
        "| dataset | model | base rate | " + " | ".join(f"precision@{k}" for k in ks) + " |",
        "|---|---|---|" + "---|" * len(ks),
    ]
    for dataset, by_model in outcome.items():
        for model, m in by_model.items():
            out.append(f"| {name(dataset)} | {model} | {m['base_rate']:.1%} | "
                       + " | ".join(f"{m['precision_at'][k]:.0%}" for k in ks) + " |")
    out += [
        "",
        "## 4. Calibration",
        "",
        "Expected calibration error (ECE) is in the table of section 1. In the diagrams, a point",
        "on the dashed line means the predicted score matches the share that really converted.",
        f"Bins with fewer than {cfg['eval']['reliability_min_bin']} conversations are not drawn "
        "(they still count in the ECE).",
    ]
    for model, figure in figures["reliability"].items():
        out += ["", f"![Reliability, {model}]({figure})"]

    agreement = metrics["labels"]
    out += ["", "## 5. Agreement with annotators", ""]
    if not agreement["annotators"]:
        out += ["No label files exist in `annotation/labels/` yet, so this section is pending."]
    else:
        out += [
            "Labels cover 60 fixed test-split conversations (`annotation/sample.json`).",
            "",
            "| annotator | kind | conversations | customer turns | more / same / less | turning points |",
            "|---|---|---|---|---|---|",
        ]
        for a in agreement["annotators"]:
            d = a["directions"]
            out.append(f"| {a['annotator']} | **{a['kind']}** | {a['conversations']} | {a['customer_turns']} "
                       f"| {d.get('more', 0)} / {d.get('same', 0)} / {d.get('less', 0)} | {a['turning_points']} |")
        if all(a["kind"] == "LLM" for a in agreement["annotators"]):
            out += [
                "",
                "> **There are no human labels yet.** Every number in this section is agreement with",
                "> an **LLM annotator**, not with people. Treat it as indicative only.",
            ]
        out += [
            "",
            "### Direction of the score (up / same / down)",
            "",
            f"For each labelled customer turn after turn 0, the score's change from the previous turn",
            f"is called *same* when it is within ±{cfg['eval']['direction_epsilon']} and *up* or *down*",
            "otherwise, and compared with the label (more / same / less).",
            "",
            "| annotator | model | dataset | turns | accuracy | always-\"same\" accuracy | Cohen's kappa |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in agreement["directions"]:
            out.append(f"| {r['annotator']} ({r['kind']}) | {r['model']} | {name(r['dataset'])} | {r['n_turns']} "
                       f"| {r['accuracy']:.1%} | {r['always_same_accuracy']:.1%} | {r['kappa']:.3f} |")
        out += [
            "",
            "### Turning points",
            "",
            f"A predicted turning point counts as correct when a labelled one lies within "
            f"±{cfg['eval']['turning_point_tolerance']} turn.",
            "",
            "| annotator | model | method | dataset | predicted | labelled | precision | recall | F1 |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in agreement["turning_points"]:
            out.append(f"| {r['annotator']} ({r['kind']}) | {r['model']} | {METHOD_LABELS[r['method']]} "
                       f"| {name(r['dataset'])} | {r['n_predicted']} | {r['n_labelled']} "
                       f"| {_fmt(r['precision'], '.0%')} | {_fmt(r['recall'], '.0%')} | {_fmt(r['f1'], '.2f')} |")
        out += ["", "### Agreement between annotators", ""]
        if agreement["between_annotators"]:
            out += ["| annotators | conversations | direction agreement | direction kappa | turning-point kappa |",
                    "|---|---|---|---|---|"]
            for r in agreement["between_annotators"]:
                pair = " vs ".join(f"{a} ({k})" for a, k in zip(r["annotators"], r["kinds"]))
                out.append(f"| {pair} | {r['conversations']} | {r['direction_agreement']:.1%} "
                           f"| {r['direction_kappa']:.3f} | {r['turning_point_kappa']:.3f} |")
        else:
            out += ["Pending: Cohen's kappa between annotators needs at least two annotators, and only "
                    "one label file exists."]

    out += [
        "",
        "## 6. Turning points and explanation faithfulness",
        "",
        "Turning points found on the test split by the two methods.",
        "",
        "| dataset | model | method | points | per conversation | conversations with at least one | share that are rises |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in metrics["turning_points"]:
        out.append(f"| {name(r['dataset'])} | {r['model']} | {METHOD_LABELS[r['method']]} | {r['points']:,} "
                   f"| {r['per_conversation']:.2f} | {r['conversations_with_one']:.0%} | {r['rise_share']:.0%} |")
    out += [
        "",
        "**Faithfulness** (method A, test split): mean absolute score change when the trigger turn",
        "is removed and the conversation is re-scored, against removing one random other turn of",
        "the same conversation. A faithful explanation should move the score much more.",
        "",
        "| dataset | model | turning points | trigger removed | random turn removed | ratio | trigger effect is larger | flagged weak |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in metrics["faithfulness"]:
        out.append(f"| {name(r['dataset'])} | {r['model']} | {r['n']:,} | {r['trigger_removed']:.3f} "
                   f"| {r['random_removed']:.3f} | {_fmt(r['ratio'], '.1f')}x | {r['trigger_larger_share']:.0%} "
                   f"| {r['weak_share']:.0%} |")

    out += [
        "",
        "## 7. Cross-domain transfer",
        "",
        "ROC AUC of the final-turn score on each dataset's test split, for a model trained on the",
        "dataset in the row. Scores are uncalibrated here, which does not affect AUC. The diagonal",
        "is the in-domain result; it can differ slightly from section 1 because calibration",
        "creates ties between scores.",
    ]
    cross = pd.DataFrame(metrics["cross_domain"])
    for model, group in cross.groupby("model", sort=False):
        table = group.pivot(index="train_dataset", columns="test_dataset", values="roc_auc")
        order = [d for d in cfg["data"]["datasets"] if d in table.index]
        out += ["", f"**{model}**", "",
                "| trained on \\ tested on | " + " | ".join(name(d) for d in order) + " |",
                "|---|" + "---|" * len(order)]
        for train in order:
            cells = [f"**{table.at[train, test]:.3f}**" if train == test else f"{table.at[train, test]:.3f}"
                     for test in order]
            out.append(f"| {name(train)} | " + " | ".join(cells) + " |")

    out += [
        "",
        "## 8. Case studies",
        "",
        "Picked automatically from the test split, using each dataset's best model by ROC AUC.",
        "Open one in the dashboard with `http://localhost:8501/Conversation_View?conversation=<id>`.",
    ]
    cases = pd.DataFrame(metrics["case_studies"])
    for dataset, group in cases.groupby("dataset", sort=False):
        out += ["", f"### {name(dataset)} ({group['model'].iloc[0]})"]
        for kind, rows in group.groupby("kind", sort=False):
            out += ["", f"**{kind.capitalize()}**", ""]
            out += [f"- `{row.conversation_id}` — {row.description}" for row in rows.itertuples()]

    taggers = metrics["tagger_agreement"]
    out += ["", "## 9. Rule tagger against the Gemini tagger", ""]
    if not taggers["cues"]:
        out += ["The Gemini cue tagger has not been run, so there is nothing to compare."]
    else:
        out += [
            f"Both taggers were run on the same {taggers['conversations']:,} conversations "
            f"({taggers['n_turns']:,} turns), using `{cfg['llm']['tagging_model']}`. Neither is ground "
            "truth: this shows where they differ, not which one is right. Customer cues are compared "
            "on customer turns and agent moves on agent turns.",
            "",
            "| cue | on | tagged by rules | tagged by Gemini | tagged by both | agreement | Cohen's kappa |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in taggers["cues"]:
            out.append(f"| {r['cue']} | {r['role']} turns | {r['rules']:,} | {r['llm']:,} | {r['both']:,} "
                       f"| {r['agreement']:.1%} | {_fmt(r['kappa'])} |")
    out.append("")
    target.write_text("\n".join(out), encoding="utf-8")


def main() -> None:
    cfg = load_config()
    out = path("outputs")
    all_leads = pd.read_parquet(out / "leads.parquet")
    leads = all_leads[all_leads["split"] == "test"]
    test_ids = set(leads["conversation_id"])
    scores = pd.read_parquet(out / "scores.parquet")
    scores = scores[scores["conversation_id"].isin(test_ids)]
    scores_n = scores.merge(leads[["conversation_id", "model", "n_turns"]], on=["conversation_id", "model"])
    points = pd.read_parquet(out / "turning_points.parquet")
    points = points[points["conversation_id"].isin(test_ids)]

    outcome = {}
    for dataset in cfg["data"]["datasets"]:
        for model in sorted(leads.loc[leads["dataset"] == dataset, "model"].unique()):
            # A fresh generator per result, so adding a model never changes another's interval.
            rng = np.random.default_rng(cfg["seed"])
            keep_leads = leads[(leads["dataset"] == dataset) & (leads["model"] == model)]
            if keep_leads["outcome"].nunique() < 2:
                continue  # AUC is undefined when every conversation has the same outcome
            keep_scores = scores_n[(scores_n["dataset"] == dataset) & (scores_n["model"] == model)]
            outcome.setdefault(dataset, {})[model] = evaluate_outcome(keep_scores, keep_leads, cfg, rng)

    outcomes = all_leads.drop_duplicates("conversation_id").set_index("conversation_id")["outcome"]
    metrics = {
        "outcome": outcome,
        "turning_points": count_turning_points(points, leads),
        "faithfulness": evaluate_faithfulness(points),
        "cross_domain": evaluate_cross_domain(pd.read_parquet(out / "cross_scores.parquet"), outcomes),
        "labels": evaluate_labels(scores, points, cfg),
        "case_studies": pick_case_studies(outcome, leads, points, cfg["eval"]["case_studies_per_kind"],
                                          cfg["eval"]["case_study_min_conversations"]),
        "tagger_agreement": evaluate_taggers(pd.read_parquet(out / "cues.parquet")),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    figures_dir = path("figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    figures = {"early": {}, "reliability": {}}
    for model in sorted({m for by_model in outcome.values() for m in by_model}):
        for kind, plot in (("early", plot_early_prediction), ("reliability", plot_reliability)):
            name = "early_prediction" if kind == "early" else "reliability"
            target = figures_dir / f"{name}_{model}.png"
            plot(outcome, model, cfg, target)
            figures[kind][model] = "../" + target.relative_to(ROOT).as_posix()
    report = ROOT / "eval" / "report.md"
    write_report(metrics, figures, cfg, report)

    for dataset, by_model in outcome.items():
        for model, m in by_model.items():
            early = " ".join(f"{float(f):.0%}={v:.3f}" for f, v in m["early_auc"].items())
            print(f"{dataset:<11}{model}  AUC {m['roc_auc']:.3f} [{m['roc_auc_ci'][0]:.3f}, "
                  f"{m['roc_auc_ci'][1]:.3f}]  PR {m['pr_auc']:.3f}  Brier {m['brier']:.3f} "
                  f"(base {m['baseline_majority']['brier']:.3f})  ECE {m['ece']:.3f}  "
                  f"P@10 {m['precision_at']['10']:.0%}  P@20 {m['precision_at']['20']:.0%}  "
                  f"base {m['base_rate']:.1%}")
            print(f"{'':<11}early AUC: {early}")
    n_figures = sum(map(len, figures.values()))
    print(f"Wrote {out / 'metrics.json'}, {report} and {n_figures} figures")


if __name__ == "__main__":
    main()
