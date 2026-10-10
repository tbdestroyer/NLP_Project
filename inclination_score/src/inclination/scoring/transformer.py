"""v3 scorer: fine-tuned transformer on `[CUSTOMER] text [AGENT] text ...` prefixes.

A prefix is built from the token ids of turns 0..k only, keeping the most recent turns when it
does not fit, so a score at turn k cannot depend on later turns.
"""
import json
import time
from pathlib import Path

from inclination.config import ROOT, load_config  # first: sets HF_HOME before the HF import

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from inclination.scoring.calibrate import fit_calibrator

NAME = "v3"


def model_dir(dataset: str) -> Path:
    return ROOT / load_config()["model"]["dir"] / f"{NAME}_{dataset}"


def turn_string(role: str, text: str) -> str:
    return f"[{role.upper()}] {text}"


def sample_prefix_ends(n_turns: int, max_prefixes: int) -> np.ndarray:
    """Turn indices of at most `max_prefixes` prefixes, spread evenly and including the last."""
    if n_turns <= max_prefixes:
        return np.arange(n_turns)
    return np.unique(np.linspace(0, n_turns - 1, max_prefixes).round().astype(int))


def token_batches(order, lengths, max_items: int, max_tokens: int) -> list[list[int]]:
    """Splits `order` into batches of at most `max_items` whose padded size stays within
    `max_tokens`, so long inputs get smaller batches."""
    batches, batch, width = [], [], 0
    for i in order:
        new_width = max(width, lengths[i])
        if batch and (len(batch) == max_items or (len(batch) + 1) * new_width > max_tokens):
            batches.append(batch)
            batch, new_width = [], lengths[i]
        batch.append(int(i))
        width = new_width
    if batch:
        batches.append(batch)
    return batches


class TransformerScorer:
    name = NAME

    def __init__(self, model, tokenizer, cfg: dict, calibrator=None, device: str | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = cfg["max_length"]
        self.batch_size = cfg["batch_size"]
        self.infer_batch_size = cfg["infer_batch_size"]
        self.infer_max_batch_tokens = cfg["infer_max_batch_tokens"]
        self.amp = cfg["amp"]
        self.calibrator = calibrator
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    # ---- encoding -------------------------------------------------------------------------

    def encode_turns(self, roles, texts) -> list[list[int]]:
        """Token ids of each turn on its own, without special tokens."""
        strings = [turn_string(role, text) for role, text in zip(roles, texts)]
        if not strings:
            return []
        return self.tokenizer(strings, add_special_tokens=False)["input_ids"]

    def pack(self, turn_ids: list[list[int]]) -> list[int]:
        """[CLS] + the most recent whole turns that fit + [SEP]. If even the last turn is too
        long, its most recent tokens are kept."""
        budget = self.max_length - 2
        kept, used = [], 0
        for ids in reversed(turn_ids):
            if used + len(ids) > budget:
                if not kept:
                    kept.append(ids[-budget:])
                break
            kept.append(ids)
            used += len(ids)
        body = [token for ids in reversed(kept) for token in ids]
        return [self.tokenizer.cls_token_id, *body, self.tokenizer.sep_token_id]

    def _batch(self, sequences: list[list[int]]):
        width = max(len(s) for s in sequences)
        ids = torch.full((len(sequences), width), self.tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(sequences), width), dtype=torch.long)
        for i, sequence in enumerate(sequences):
            ids[i, : len(sequence)] = torch.tensor(sequence)
            mask[i, : len(sequence)] = 1
        return ids.to(self.device), mask.to(self.device)

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.float16, enabled=self.amp and self.device == "cuda")

    # ---- scoring --------------------------------------------------------------------------

    @torch.no_grad()
    def predict_raw(self, sequences: list[list[int]], progress: str = "") -> np.ndarray:
        """P(outcome = 1) before calibration. Batches are formed by length to limit padding."""
        self.model.eval()
        lengths = [len(s) for s in sequences]
        batches = token_batches(
            np.argsort(lengths)[::-1], lengths, self.infer_batch_size, self.infer_max_batch_tokens
        )
        raw = np.zeros(len(sequences))
        started, scored = time.time(), 0
        for done, index in enumerate(batches):
            ids, mask = self._batch([sequences[i] for i in index])
            with self._autocast():
                logits = self.model(input_ids=ids, attention_mask=mask).logits
            raw[index] = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
            scored += len(index)
            if progress and done and done % 200 == 0:
                rate = scored / (time.time() - started)
                print(f"    {progress}: {scored:,}/{len(sequences):,} ({rate:.0f}/s)", flush=True)
        return raw

    def predict(self, sequences: list[list[int]], progress: str = "") -> np.ndarray:
        if not sequences:
            return np.zeros(0)
        return self.calibrator.transform(self.predict_raw(sequences, progress))

    def prefix_sequences(self, turn_ids: list[list[int]], groups: np.ndarray) -> list[list[int]]:
        """One packed sequence per turn: the prefix ending at that turn. `groups` marks the
        contiguous run (conversation) of each turn."""
        sequences, start = [], 0
        for row in range(len(turn_ids)):
            if row > 0 and groups[row] != groups[row - 1]:
                start = row
            sequences.append(self.pack(turn_ids[start : row + 1]))
        return sequences

    def score(self, conversation_turns) -> list[float]:
        """Standard scorer interface: [(role, text), ...] -> one calibrated score per turn."""
        turn_ids = self.encode_turns(*zip(*conversation_turns)) if conversation_turns else []
        groups = np.zeros(len(turn_ids), dtype=np.int64)
        return self.predict(self.prefix_sequences(turn_ids, groups)).tolist()

    # ---- persistence ----------------------------------------------------------------------

    def save(self, folder: Path, meta: dict) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(folder)
        self.tokenizer.save_pretrained(folder)
        joblib.dump(self.calibrator, folder / "calibrator.joblib")
        (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, dataset: str, cfg: dict) -> "TransformerScorer":
        folder = model_dir(dataset)
        model = AutoModelForSequenceClassification.from_pretrained(folder)
        tokenizer = AutoTokenizer.from_pretrained(folder)
        return cls(model, tokenizer, cfg["model"]["v3"], joblib.load(folder / "calibrator.joblib"))


def _training_samples(part: pd.DataFrame, conversation_ids, turn_ids, scorer, max_prefixes: int):
    """Packed sequences, labels and weights for the sampled prefixes of some conversations."""
    sequences, labels, weights = [], [], []
    wanted = part[part["conversation_id"].isin(set(conversation_ids))]
    for _, rows in wanted.groupby("conversation_id", sort=False):
        index = rows.index.to_numpy()
        ends = sample_prefix_ends(len(index), max_prefixes)
        for end in ends:
            sequences.append(scorer.pack([turn_ids[i] for i in index[: end + 1]]))
            labels.append(int(rows["outcome"].iat[0]))
            weights.append(1.0 / len(ends))
    return sequences, np.array(labels), np.array(weights)


def train(part: pd.DataFrame, dataset: str, cfg: dict) -> tuple[TransformerScorer, dict]:
    """`part`: one dataset's turns (sorted, default index) with outcome and split per row."""
    v3 = cfg["model"]["v3"]
    rng = np.random.default_rng(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = v3["name"] if device == "cuda" else v3["cpu_fallback"]
    if device == "cpu":
        print(f"  WARNING: no GPU found; falling back to {name} on a subsample")

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)
    scorer = TransformerScorer(model, tokenizer, v3, device=device)

    conversations = part.drop_duplicates("conversation_id")[["conversation_id", "split", "outcome"]]
    def pick(split: str, limit: int):
        ids = conversations.loc[conversations["split"] == split, "conversation_id"].to_numpy()
        return ids if len(ids) <= limit else rng.choice(ids, size=limit, replace=False)
    train_limit = v3["max_train_conversations"] if device == "cuda" else v3["cpu_max_train_conversations"]
    train_ids, val_ids = pick("train", train_limit), pick("val", v3["max_val_conversations"])

    used = part["conversation_id"].isin(set(train_ids) | set(val_ids)).to_numpy()
    turn_ids = [None] * len(part)
    for row, ids in zip(np.flatnonzero(used), scorer.encode_turns(part.loc[used, "role"], part.loc[used, "text"])):
        turn_ids[row] = ids

    max_prefixes = cfg["model"]["max_prefixes_per_conversation"]
    sequences, labels, weights = _training_samples(part, train_ids, turn_ids, scorer, max_prefixes)
    val_sequences, val_labels, _ = _training_samples(part, val_ids, turn_ids, scorer, max_prefixes)

    counts = np.bincount(labels, minlength=2)
    class_weights = torch.tensor(len(labels) / (2 * counts), dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=v3["learning_rate"], weight_decay=v3["weight_decay"])
    lengths = [len(s) for s in sequences]
    epoch_batches = [
        token_batches(rng.permutation(len(sequences)), lengths, v3["batch_size"], v3["max_batch_tokens"])
        for _ in range(v3["epochs"])
    ]
    steps_per_epoch = len(epoch_batches[0])
    total_steps = sum(map(len, epoch_batches))
    schedule = get_linear_schedule_with_warmup(
        optimizer, int(v3["warmup_fraction"] * total_steps), total_steps
    )
    grad_scaler = torch.amp.GradScaler("cuda", enabled=v3["amp"] and device == "cuda")
    print(f"  {name} on {device}: {len(sequences):,} train prefixes from {len(train_ids):,} "
          f"conversations, about {steps_per_epoch:,} steps/epoch, up to {v3['epochs']} epochs",
          flush=True)

    best_auc, best_state, history, stale = -1.0, None, [], 0
    started, step = time.time(), 0
    for epoch in range(v3["epochs"]):
        model.train()
        for index in epoch_batches[epoch]:
            ids, mask = scorer._batch([sequences[i] for i in index])
            target = torch.tensor(labels[index], device=device)
            weight = torch.tensor(weights[index], dtype=torch.float, device=device)
            with scorer._autocast():
                logits = model(input_ids=ids, attention_mask=mask).logits
            loss = (loss_fn(logits.float(), target) * weight).sum() / weight.sum()
            optimizer.zero_grad()
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), v3["max_grad_norm"])
            grad_scaler.step(optimizer)
            grad_scaler.update()
            schedule.step()
            step += 1
            if step % v3["log_every"] == 0:
                elapsed = time.time() - started
                remaining = elapsed / step * (total_steps - step)
                print(f"    epoch {epoch + 1} step {step:,}/{total_steps:,} loss {loss.item():.3f} "
                      f"| {elapsed / 60:.1f} min elapsed, ~{remaining / 60:.0f} min left", flush=True)
        auc = float(roc_auc_score(val_labels, scorer.predict_raw(val_sequences)))
        history.append(auc)
        print(f"  epoch {epoch + 1}: validation AUC over sampled prefixes = {auc:.4f}", flush=True)
        if auc > best_auc:
            best_auc, stale = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= v3["patience"]:
                print("  early stopping", flush=True)
                break
    model.load_state_dict(best_state)

    # Calibrate on every prefix of the validation conversations used above.
    val_rows = np.flatnonzero(part["conversation_id"].isin(set(val_ids)).to_numpy())
    groups = part["group"].to_numpy()[val_rows]
    val_prefixes = scorer.prefix_sequences([turn_ids[r] for r in val_rows], groups)
    _, sizes = np.unique(groups, return_counts=True)
    scorer.calibrator = fit_calibrator(
        scorer.predict_raw(val_prefixes), part["outcome"].to_numpy()[val_rows],
        np.repeat(1.0 / sizes, sizes), len(val_ids), cfg["model"]["calibration"],
    )
    meta = {
        "base_model": name, "device": device, "dataset": dataset,
        "train_conversations": int(len(train_ids)), "train_prefixes": int(len(sequences)),
        "val_conversations": int(len(val_ids)), "val_auc_by_epoch": history,
        "best_val_auc": best_auc, "calibration": scorer.calibrator.method,
        "max_length": v3["max_length"], "minutes": round((time.time() - started) / 60, 1),
    }
    return scorer, meta
