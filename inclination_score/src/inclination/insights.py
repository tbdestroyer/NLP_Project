"""Script insights: how the customer's score moves on the customer turn after each agent move.

For an agent turn j and the next customer turn c, the effect is score[c] - score[j]: the change
that happens once the customer has replied. These are associations, not causes.
"""
import numpy as np
import pandas as pd

from inclination.cues.taxonomy import AGENT_MOVE_NAMES

BOOTSTRAP_CHUNK = 100


def next_customer_rows(roles: np.ndarray, position: np.ndarray) -> np.ndarray:
    """For each row, the row of the first customer turn after it in the same conversation, or -1."""
    result = np.full(len(roles), -1, dtype=np.int64)
    next_customer = -1
    for row in range(len(roles) - 1, -1, -1):
        result[row] = next_customer
        if roles[row] == "customer":
            next_customer = row
        if position[row] == 0:
            next_customer = -1
    return result


def bootstrap_ci(values: np.ndarray, n_samples: int, level: float, rng) -> tuple[float, float]:
    means = []
    for _ in range(0, n_samples, BOOTSTRAP_CHUNK):
        index = rng.integers(0, len(values), size=(BOOTSTRAP_CHUNK, len(values)))
        means.append(values[index].mean(axis=1))
    low, high = np.quantile(np.concatenate(means)[:n_samples], [(1 - level) / 2, (1 + level) / 2])
    return float(low), float(high)


def move_effects(frame: pd.DataFrame, moves: np.ndarray, cfg: dict, seed: int):
    """`frame`: one dataset's rows (sorted) with role, score, position, size, in_scope.
    `moves`: rows x agent-move 0/1 matrix. Only rows with in_scope (the test split) are counted.

    Returns (effects, examples). `effects` has one row per move and position bucket.
    """
    roles = frame["role"].to_numpy()
    position = frame["position"].to_numpy()
    score = frame["score"].to_numpy()
    following = next_customer_rows(roles, position)
    usable = (roles == "agent") & (following >= 0) & frame["in_scope"].to_numpy()
    delta = np.where(usable, score[np.maximum(following, 0)] - score, np.nan)
    buckets = cfg["position_buckets"]
    bucket = np.minimum((position * len(buckets)) // frame["size"].to_numpy(), len(buckets) - 1)
    rng = np.random.default_rng(seed)

    effects, examples = [], []
    for column, move in enumerate(AGENT_MOVE_NAMES):
        has_move = usable & (moves[:, column] > 0)
        for label in ["all", *buckets]:
            rows = np.flatnonzero(has_move if label == "all" else has_move & (bucket == buckets.index(label)))
            if len(rows) < cfg["min_count"]:
                continue
            values = delta[rows]
            low, high = bootstrap_ci(values, cfg["bootstrap_samples"], cfg["ci_level"], rng)
            effects.append({
                "move": move, "position": label, "mean_delta": float(values.mean()),
                "ci_low": low, "ci_high": high, "count": len(rows),
                "share_rise": float((values > 0).mean()), "share_drop": float((values < 0).mean()),
            })
            # Examples that show the typical direction, strongest first.
            sign = 1 if values.mean() >= 0 else -1
            for row in rows[np.argsort(-sign * values)[: cfg["n_examples"]]]:
                examples.append({
                    "move": move, "position": label,
                    "conversation_id": frame["conversation_id"].iat[row],
                    "agent_turn": int(frame["turn"].iat[row]),
                    "customer_turn": int(frame["turn"].iat[following[row]]),
                    "delta": float(delta[row]),
                })
    return pd.DataFrame(effects), pd.DataFrame(examples)
