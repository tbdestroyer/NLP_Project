"""v1 scorer: logistic regression on cue counts and TF-IDF of the customer's text so far.

Every feature of the prefix ending at turn k is a sum over turns 0..k, so a score can never
depend on later turns.
"""
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from inclination.cues import rules
from inclination.cues.taxonomy import AGENT_MOVE_NAMES, CUSTOMER_CUE_NAMES
from inclination.scoring.calibrate import fit_calibrator

N_CUSTOMER = len(CUSTOMER_CUE_NAMES)


@dataclass
class PerTurn:
    """Per-turn quantities; prefix features are sums of these rows."""
    customer_cues: sparse.csr_matrix   # turns x customer cues
    agent_moves: sparse.csr_matrix     # turns x agent moves
    counts: sparse.csr_matrix          # turns x vocabulary (customer turns only)
    customer_words: np.ndarray         # words in the turn if it is a customer turn, else 0
    is_customer: np.ndarray

    def take(self, rows) -> "PerTurn":
        return PerTurn(
            self.customer_cues[rows], self.agent_moves[rows], self.counts[rows],
            self.customer_words[rows], self.is_customer[rows],
        )


def _runs(groups: np.ndarray):
    """Start, size and within-run position for contiguous runs of equal group ids."""
    n = len(groups)
    starts = np.concatenate([[0], np.flatnonzero(groups[1:] != groups[:-1]) + 1])
    sizes = np.diff(np.concatenate([starts, [n]]))
    position = np.arange(n) - np.repeat(starts, sizes)
    return starts, sizes, position


def _band(position: np.ndarray, max_offset: int) -> sparse.csr_matrix:
    """Row i sums rows i-d for d = 0..max_offset, never crossing the start of its run."""
    n = len(position)
    rows, cols = [], []
    for offset in range(max_offset + 1):
        index = np.flatnonzero(position >= offset)
        if len(index) == 0:
            break
        rows.append(index)
        cols.append(index - offset)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n))


def prefix_operators(groups: np.ndarray, window: int):
    """Operators giving one output row per turn: (cumulative, last-`window`-turns, turn index)."""
    _, _, position = _runs(groups)
    return _band(position, int(position.max())), _band(position, window - 1), position


def last_operators(groups: np.ndarray, window: int):
    """Operators giving one output row per run: the features of its final prefix only."""
    starts, sizes, position = _runs(groups)
    n, n_runs = len(groups), len(starts)
    run = np.repeat(np.arange(n_runs), sizes)
    ones = np.ones(n, dtype=np.float32)
    total = sparse.csr_matrix((ones, (run, np.arange(n))), shape=(n_runs, n))
    recent = np.flatnonzero(np.repeat(sizes, sizes) - 1 - position < window)
    tail = sparse.csr_matrix((ones[recent], (run[recent], recent)), shape=(n_runs, n))
    return total, tail, sizes - 1


class CueModel:
    name = "v1"

    def __init__(self, cfg: dict):
        self.window = cfg["cues"]["recent_window"]
        self.calibration_cfg = cfg["model"]["calibration"]
        v1 = cfg["model"]["v1"]
        self.vectorizer = CountVectorizer(
            max_features=cfg["cues"]["tfidf_max_features"], min_df=cfg["cues"]["tfidf_min_df"]
        )
        self.tfidf = TfidfTransformer(sublinear_tf=True)
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(
            C=v1["C"], max_iter=v1["max_iter"], class_weight=v1["class_weight"]
        )
        self.calibrator = None

    # ---- features -------------------------------------------------------------------------

    def per_turn(self, roles, texts, cue_matrix: np.ndarray) -> PerTurn:
        is_customer = np.asarray(roles) == "customer"
        customer_text = [text if mine else "" for text, mine in zip(texts, is_customer)]
        words = np.array([len(text.split()) for text in customer_text], dtype=np.float32)
        return PerTurn(
            sparse.csr_matrix(cue_matrix[:, :N_CUSTOMER]),
            sparse.csr_matrix(cue_matrix[:, N_CUSTOMER:]),
            self.vectorizer.transform(customer_text).astype(np.float32),
            words,
            is_customer.astype(np.float32),
        )

    def _aggregate(self, per_turn: PerTurn, cumulative, recent, turn_index):
        n_customer_turns = cumulative @ per_turn.is_customer
        average_words = (cumulative @ per_turn.customer_words) / np.maximum(n_customer_turns, 1)
        dense = np.hstack([
            (cumulative @ per_turn.customer_cues).toarray(),
            (recent @ per_turn.customer_cues).toarray(),
            (cumulative @ per_turn.agent_moves).toarray(),
            np.asarray(turn_index, dtype=np.float32)[:, None],
            average_words[:, None],
        ])
        return dense, cumulative @ per_turn.counts

    def _stack(self, dense, counts) -> sparse.csr_matrix:
        return sparse.hstack(
            [sparse.csr_matrix(self.scaler.transform(dense)), self.tfidf.transform(counts)]
        ).tocsr()

    def features(self, per_turn: PerTurn, groups: np.ndarray) -> sparse.csr_matrix:
        """One feature row per turn: the prefix ending at that turn."""
        return self._stack(*self._aggregate(per_turn, *prefix_operators(groups, self.window)))

    @property
    def feature_names(self) -> list[str]:
        return (
            [f"cum:{name}" for name in CUSTOMER_CUE_NAMES]
            + [f"recent:{name}" for name in CUSTOMER_CUE_NAMES]
            + [f"agent:{name}" for name in AGENT_MOVE_NAMES]
            + ["turn_index", "avg_customer_words"]
            + [f"word:{word}" for word in self.vectorizer.get_feature_names_out()]
        )

    # ---- training and scoring -------------------------------------------------------------

    def train(self, roles, texts, cue_matrix, groups, labels, splits) -> PerTurn:
        """All arguments have one entry per turn; `labels` is the conversation outcome."""
        roles, groups, labels, splits = map(np.asarray, (roles, groups, labels, splits))
        texts = list(texts)
        train, val = splits == "train", splits == "val"
        self.vectorizer.fit(
            [text for text, role, use in zip(texts, roles, train) if use and role == "customer"]
        )
        per_turn = self.per_turn(roles, texts, cue_matrix)

        dense, counts = self._aggregate(
            per_turn.take(train), *prefix_operators(groups[train], self.window)
        )
        self.scaler.fit(dense)
        self.tfidf.fit(counts)
        self.clf.fit(
            self._stack(dense, counts), labels[train], sample_weight=prefix_weights(groups[train])
        )

        raw_val = self.predict_raw(self.features(per_turn.take(val), groups[val]))
        self.calibrator = fit_calibrator(
            raw_val, labels[val], prefix_weights(groups[val]),
            len(np.unique(groups[val])), self.calibration_cfg,
        )
        return per_turn

    def predict_raw(self, X) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]

    def predict(self, X) -> np.ndarray:
        return self.calibrator.transform(self.predict_raw(X))

    def score_last(self, per_turn: PerTurn, groups: np.ndarray) -> np.ndarray:
        """Calibrated score of the final prefix of each run in `groups`."""
        return self.predict(
            self._stack(*self._aggregate(per_turn, *last_operators(groups, self.window)))
        )

    def score(self, conversation_turns) -> list[float]:
        """Standard scorer interface: [(role, text), ...] -> one calibrated score per turn."""
        roles = [role for role, _ in conversation_turns]
        texts = [text for _, text in conversation_turns]
        cue_matrix = rules.tags_to_matrix(rules.tag_turns(roles, texts))
        per_turn = self.per_turn(roles, texts, cue_matrix)
        return self.predict(self.features(per_turn, np.zeros(len(roles), dtype=np.int64))).tolist()

    # ---- explanation support --------------------------------------------------------------

    def top_contributions(self, X, rows: np.ndarray, k: int) -> list[list[dict]]:
        """For each row r: the k features with the largest |coefficient x (x_r - x_{r-1})|."""
        names = self.feature_names
        change = (X[rows] - X[rows - 1]).multiply(self.clf.coef_[0]).tocsr()
        result = []
        for i in range(change.shape[0]):
            start, end = change.indptr[i], change.indptr[i + 1]
            values, columns = change.data[start:end], change.indices[start:end]
            order = np.argsort(-np.abs(values))[:k]
            result.append([
                {"feature": names[columns[j]], "contribution": round(float(values[j]), 4)}
                for j in order if values[j] != 0
            ])
        return result


def prefix_weights(groups: np.ndarray) -> np.ndarray:
    """1 / number of prefixes in the conversation, so long conversations do not dominate."""
    _, sizes, _ = _runs(groups)
    return np.repeat(1.0 / sizes, sizes)
