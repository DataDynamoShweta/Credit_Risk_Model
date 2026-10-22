"""Probability-to-score scaling using points-to-double-odds (PDO)."""
import numpy as np

def probability_to_score(probability, *, base_score: int = 600,
                         base_odds_good_to_bad: float = 50.0, pdo: int = 20):
    """Map PD to score. Higher score means lower PD.

    The base score corresponds to the supplied good:bad odds. Probabilities are clipped
    to avoid infinite logits. Validate the score convention with business stakeholders.
    """
    if base_odds_good_to_bad <= 0 or pdo <= 0:
        raise ValueError("base odds and PDO must be positive")
    p = np.clip(np.asarray(probability, dtype=float), 1e-8, 1 - 1e-8)
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds_good_to_bad)
    score = offset - factor * np.log(p / (1 - p))
    result = np.rint(score).astype(int)
    return int(result) if result.ndim == 0 else result
