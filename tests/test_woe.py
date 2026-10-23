"""WoE bin ordering and monotonicity (G4.1)."""
import numpy as np
import pandas as pd

from credit_risk_model.features.woe_binning import calculate_woe_iv


def _bins_lower_edges(result):
    return [float(b.split(",")[0].strip("([")) for b in result.table["bin"]]


def test_strictly_increasing_feature_is_monotonic_and_numerically_ordered():
    # Values 0..299 so that text sorting would put "(99.9, ...]" after "(199.8, ...]".
    x = pd.Series(np.arange(300, dtype=float))
    y = pd.Series((x >= 150).astype(int))
    result = calculate_woe_iv(x, y, bins=3)
    edges = _bins_lower_edges(result)
    assert edges == sorted(edges)
    assert result.monotonic is True


def test_decreasing_relationship_is_monotonic():
    x = pd.Series(np.arange(1000, dtype=float))
    rng = np.random.default_rng(1)
    y = pd.Series((rng.random(1000) < np.linspace(0.6, 0.02, 1000)).astype(int))
    result = calculate_woe_iv(x, y, bins=5)
    rates = result.table["bad_rate"].to_numpy()
    assert np.all(np.diff(rates) <= 0)
    assert result.monotonic is True


def test_non_monotonic_relationship_is_reported():
    x = pd.Series(np.repeat([1.0, 2.0, 3.0, 4.0], 100))
    y = pd.Series(np.concatenate([np.r_[np.ones(10), np.zeros(90)],
                                  np.r_[np.ones(50), np.zeros(50)],
                                  np.r_[np.ones(10), np.zeros(90)],
                                  np.r_[np.ones(50), np.zeros(50)]]).astype(int))
    assert calculate_woe_iv(x, y, bins=4).monotonic is False


def test_categorical_feature_has_no_monotonicity_claim():
    x = pd.Series(["b", "a", "c", "a", "b", "c"] * 20)
    y = pd.Series([0, 1, 0, 1, 1, 0] * 20)
    result = calculate_woe_iv(x, y)
    assert result.kind == "categorical"
    assert result.monotonic is None
    assert list(result.table["bin"]) == ["a", "b", "c"]


def test_low_cardinality_numeric_falls_back_to_ordered_values():
    x = pd.Series([10, 2, 2, 10, 1, 1, 2, 10] * 10)
    y = pd.Series([1, 0, 1, 1, 0, 0, 0, 1] * 10)
    result = calculate_woe_iv(x, y, bins=5)
    assert result.iv >= 0
    edges = _bins_lower_edges(result)
    assert edges == sorted(edges)


def test_single_value_feature():
    result = calculate_woe_iv(pd.Series([5.0] * 10), pd.Series([0, 1] * 5))
    assert len(result.table) == 1 and result.monotonic is True
