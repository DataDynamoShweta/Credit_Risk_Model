import numpy as np
import pandas as pd
import pytest

from credit_risk_model.validation.stability import csi_by_feature, psi


def test_psi_same_distribution_near_zero():
    x = pd.Series([1,2,3,4,5,6,7,8,9,10]*10)
    assert psi(x, x) < 1e-8

def test_csi_reports_shared_features():
    result = csi_by_feature(pd.DataFrame({"a":[1,2,3]}), pd.DataFrame({"a":[2,3,4],"b":[1,2,3]}))
    assert "a" in result and "b" not in result


def test_binary_feature_drift_is_detected():
    # 15% -> 60% ones: quantile edges collapse to one bin, so this used to return 0.0.
    ref = pd.Series([1] * 15 + [0] * 85)
    cur = pd.Series([1] * 60 + [0] * 40)
    assert psi(ref, cur) > 0.25


def test_low_cardinality_numeric_drift_is_detected():
    ref = pd.Series(np.repeat([0, 1, 2, 3], [70, 10, 10, 10]))
    cur = pd.Series(np.repeat([0, 1, 2, 3], [25, 25, 25, 25]))
    assert psi(ref, cur) > 0.25


def test_categorical_drift_is_detected():
    ref = pd.Series(["RENT"] * 50 + ["OWN"] * 50)
    cur = pd.Series(["RENT"] * 90 + ["OWN"] * 10)
    assert psi(ref, cur) > 0.25


def test_unseen_category_counts_as_drift():
    ref = pd.Series(["A"] * 100)
    cur = pd.Series(["A"] * 50 + ["NEW"] * 50)
    assert np.isfinite(psi(ref, cur)) and psi(ref, cur) > 0.25


def test_stable_continuous_feature_is_low():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=5000), rng.normal(size=5000)) < 0.02


def test_csi_restricted_to_requested_features():
    ref = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
    cur = pd.DataFrame({"a": [1, 2, 3], "target": [1, 1, 1]})
    assert list(csi_by_feature(ref, cur, features=["a"])) == ["a"]
    with pytest.raises(ValueError, match="missing"):
        csi_by_feature(ref, cur, features=["a", "b"])


def test_empty_sample_rejected():
    with pytest.raises(ValueError):
        psi(pd.Series([], dtype=float), pd.Series([1.0]))
