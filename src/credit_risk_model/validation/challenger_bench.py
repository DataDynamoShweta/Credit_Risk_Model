"""Common holdout benchmark for champion and challenger probabilities."""
from .calibration import calibration
from .discrimination import discrimination


def benchmark(y_true, predictions: dict[str, object]):
    rows = []
    for name, probability in predictions.items():
        d = discrimination(y_true, probability)
        c = calibration(y_true, probability)
        rows.append({"model": name, **d, "brier": c["brier"], "log_loss": c["log_loss"]})
    return sorted(rows, key=lambda row: row["auroc"], reverse=True)
