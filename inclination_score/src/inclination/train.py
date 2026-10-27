"""Trains a scorer per dataset. Run: python -m inclination.train [--model v1|v3]"""
import argparse

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

from inclination.config import ROOT, load_config, path
from inclination.cues import rules
from inclination.data import group_codes, load_processed
from inclination.scoring.cue_model import CueModel


def model_path(model: str, dataset: str):
    return ROOT / load_config()["model"]["dir"] / f"{model}_{dataset}.joblib"


def prepare(turns: pd.DataFrame, conversations: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Turns of one dataset with the conversation's outcome and split on every row."""
    part = turns[turns["dataset"] == dataset].merge(
        conversations[["conversation_id", "outcome", "split"]], on="conversation_id"
    )
    part = part.sort_values(["conversation_id", "turn"]).reset_index(drop=True)
    part["group"] = group_codes(part["conversation_id"])
    return part


def train_v3(datasets: list[str], max_train_conversations: int | None) -> None:
    """Fine-tunes the transformer scorer per dataset. Slow: see the README for runtimes."""
    from inclination.scoring import transformer  # imported here so v1 needs no torch

    cfg = load_config()
    if max_train_conversations:
        cfg["model"]["v3"]["max_train_conversations"] = max_train_conversations
        cfg["model"]["v3"]["max_val_conversations"] = max_train_conversations
    turns, conversations = load_processed()
    for dataset in datasets:
        print(f"{dataset}: training v3", flush=True)
        scorer, meta = transformer.train(prepare(turns, conversations, dataset), dataset, cfg)
        scorer.save(transformer.model_dir(dataset), meta)
        print(f"{dataset}: best validation AUC {meta['best_val_auc']:.4f} in {meta['minutes']} min, "
              f"saved to {transformer.model_dir(dataset)}", flush=True)
        del scorer


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["v1", "v3"], default="v1")
    parser.add_argument("--datasets", nargs="+", default=cfg["data"]["datasets"])
    parser.add_argument("--max-train-conversations", type=int, default=None,
                        help="v3 only: override the config, e.g. for a quick smoke test")
    args = parser.parse_args()
    if args.model == "v3":
        train_v3(args.datasets, args.max_train_conversations)
        return
    turns, conversations = load_processed()
    coefficients = []
    for dataset in cfg["data"]["datasets"]:
        part = prepare(turns, conversations, dataset)
        cue_matrix = rules.tags_to_matrix(rules.tag_turns(part["role"], part["text"]))
        model = CueModel(cfg)
        per_turn = model.train(
            part["role"], part["text"], cue_matrix, part["group"], part["outcome"], part["split"]
        )
        part["score"] = model.predict(model.features(per_turn, part["group"].to_numpy()))

        final = part.groupby("conversation_id").tail(1)
        report = {
            split: roc_auc_score(rows["outcome"], rows["score"])
            for split, rows in final.groupby("split")
        }
        print(f"{dataset}: calibration={model.calibrator.method} | final-turn AUC "
              + " ".join(f"{s}={report[s]:.3f}" for s in ("train", "val", "test")))

        target = model_path(model.name, dataset)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, target)
        coefficients.append(pd.DataFrame({
            "dataset": dataset, "model": model.name,
            "feature": model.feature_names, "coefficient": model.clf.coef_[0],
        }))

    out = path("outputs") / "v1_coefficients.parquet"
    pd.concat(coefficients, ignore_index=True).to_parquet(out, index=False)
    print(f"Saved models to {target.parent} and coefficients to {out}")


if __name__ == "__main__":
    main()
