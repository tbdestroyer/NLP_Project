"""Score calibration fitted on the validation split: isotonic, or Platt when validation is small."""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


class Calibrator:
    def __init__(self, method: str, clip: float):
        self.method = method
        self.clip = clip
        self.model = None

    def fit(self, raw: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> "Calibrator":
        if self.method == "isotonic":
            self.model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self.model.fit(raw, labels, sample_weight=weights)
        else:
            self.model = LogisticRegression(C=1e6)
            self.model.fit(_logit(raw).reshape(-1, 1), labels, sample_weight=weights)
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=float)
        if self.method == "isotonic":
            calibrated = self.model.predict(raw)
        else:
            calibrated = self.model.predict_proba(_logit(raw).reshape(-1, 1))[:, 1]
        return np.clip(calibrated, self.clip, 1 - self.clip)


def fit_calibrator(raw, labels, weights, n_val_conversations: int, cfg: dict) -> Calibrator:
    """`cfg` is config `model.calibration`."""
    method = cfg["method"]
    if n_val_conversations < cfg["min_val_conversations"]:
        method = "platt"
    return Calibrator(method, cfg["clip"]).fit(np.asarray(raw), np.asarray(labels), np.asarray(weights))
