"""Turning points: turns where the score changes sharply."""
import numpy as np
import pandas as pd

COLUMNS = ["dataset", "conversation_id", "turn", "model", "method", "direction", "delta"]


def score_deltas(scores: pd.DataFrame) -> np.ndarray:
    """score_k - score_{k-1} per row (NaN on turn 0). `scores` is sorted by conversation, turn."""
    delta = scores["score"].diff().to_numpy(copy=True)
    delta[scores["turn"].to_numpy() == 0] = np.nan
    return delta


def delta_turning_points(scores: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Method A: |score_k - score_{k-1}| >= threshold.

    `scores` needs dataset, conversation_id, turn, model, score for ONE model, sorted by
    conversation and turn. The returned frame keeps the index of `scores`.
    """
    delta = score_deltas(scores)
    hit = np.abs(delta) >= threshold  # NaN compares False
    points = scores.loc[hit, ["dataset", "conversation_id", "turn", "model"]].copy()
    points["method"] = "delta"
    points["delta"] = delta[hit]
    points["direction"] = np.where(points["delta"] > 0, "rise", "drop")
    return points[COLUMNS]


def pelt_turning_points(scores: pd.DataFrame, penalty: float, min_size: int) -> pd.DataFrame:
    """Method B: change points of the score curve found by PELT (least-squares cost).

    A turning point is the first turn of a new segment; its delta is the difference between the
    mean score of the new segment and of the one before. Same input and output as method A.
    """
    import ruptures

    values = scores["score"].to_numpy()
    turn = scores["turn"].to_numpy()
    starts = np.flatnonzero(turn == 0)
    ends = np.append(starts[1:], len(values))
    rows, deltas = [], []
    for start, end in zip(starts, ends):
        y = values[start:end]
        n = len(y)
        # Splitting can lower the cost by at most n/4 * range^2, so most curves are skipped.
        if n < 2 * min_size or n / 4 * (y.max() - y.min()) ** 2 <= penalty:
            continue
        breaks = ruptures.Pelt(model="l2", min_size=min_size, jump=1).fit(y.reshape(-1, 1)).predict(pen=penalty)
        bounds = [0, *breaks]  # the last break is n
        for i in range(1, len(bounds) - 1):
            before, after = y[bounds[i - 1] : bounds[i]], y[bounds[i] : bounds[i + 1]]
            rows.append(start + bounds[i])
            deltas.append(after.mean() - before.mean())
    points = scores.iloc[rows][["dataset", "conversation_id", "turn", "model"]].copy()
    points["method"] = "pelt"
    points["delta"] = np.array(deltas, dtype=float)
    points["direction"] = np.where(points["delta"] > 0, "rise", "drop")
    return points[COLUMNS]
