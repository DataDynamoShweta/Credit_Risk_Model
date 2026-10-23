"""Champion calibration, metadata and feature-contract regressions (G3, G4.5)."""
import numpy as np
import pandas as pd
import pytest

from credit_risk_model.data import synthetic
from credit_risk_model.models.champion import ChampionModel
from credit_risk_model.models.contract import FeatureContractError
from credit_risk_model.validation.suite import split_dataset
from tests.helpers import FEATURES_OK


@pytest.fixture(scope="module")
def splits():
    frame = synthetic.generate(8000, seed=7)
    return split_dataset(frame, "target", synthetic.DATE_COLUMN, seed=7)


@pytest.fixture(scope="module")
def model(splits):
    return ChampionModel.fit(splits.train, exclude=(synthetic.DATE_COLUMN,))


def test_default_training_is_calibrated_in_the_large(model, splits):
    p = model.predict_proba(splits.test)
    observed = splits.test["target"].mean()
    assert 0.03 < observed < 0.08  # imbalanced, roughly 5% defaults
    assert abs(p.mean() - observed) < 0.25 * observed


def test_balanced_class_weight_inflates_pd(splits):
    weighted = ChampionModel.fit(splits.train, exclude=(synthetic.DATE_COLUMN,),
                                 class_weight="balanced")
    assert weighted.predict_proba(splits.test).mean() > 5 * splits.test["target"].mean()
    assert weighted.metadata["class_weight"] == "balanced"


def test_metadata_records_training_context(model):
    meta = model.metadata
    assert meta["class_weight"] is None
    assert meta["random_state"] == 42
    assert len(meta["training_data_sha256"]) == 64
    assert {"python", "scikit-learn", "pandas", "numpy"} <= set(meta["library_versions"])
    assert [f["name"] for f in meta["features"]] == list(FEATURES_OK)
    assert {"model_name", "model_version", "training_rows"} <= set(meta)


def test_higher_risk_features_give_higher_pd(model):
    low = model.predict_proba({**FEATURES_OK, "int_rate": 6.0})[0]
    high = model.predict_proba({**FEATURES_OK, "int_rate": 22.0})[0]
    assert high > low


@pytest.mark.parametrize("features,fragment", [
    ({}, "no features"),
    ({k: v for k, v in FEATURES_OK.items() if k != "dti"}, "missing required features: dti"),
    ({**FEATURES_OK, "extra": 1}, "unexpected features: extra"),
    ({**FEATURES_OK, "int_rate": "11.5"}, "int_rate"),
    ({**FEATURES_OK, "int_rate": float("nan")}, "int_rate"),
    ({**FEATURES_OK, "home_ownership": 1}, "home_ownership"),
])
def test_contract_rejects_bad_records(model, features, fragment):
    with pytest.raises(FeatureContractError, match=fragment):
        model.predict_proba(features)


def test_contract_errors_do_not_echo_values(model):
    with pytest.raises(FeatureContractError) as info:
        model.predict_proba({**FEATURES_OK, "int_rate": "SECRET-VALUE"})
    assert "SECRET-VALUE" not in str(info.value)


def test_frame_missing_column_rejected(model, splits):
    with pytest.raises(FeatureContractError, match="dti"):
        model.predict_proba(splits.test.drop(columns=["dti"]))


def test_unseen_category_is_scored(model):
    p = model.predict_proba({**FEATURES_OK, "home_ownership": "NEVER_SEEN"})
    assert 0 < p[0] < 1


def test_single_class_target_rejected():
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "target": [0, 0, 0]})
    with pytest.raises(ValueError, match="two classes"):
        ChampionModel.fit(frame)


def test_empty_dataset_rejected():
    with pytest.raises(ValueError):
        ChampionModel.fit(pd.DataFrame({"x": [], "target": []}))


def test_constant_feature_is_tolerated():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"x": rng.normal(size=200), "const": 1.0,
                          "target": rng.integers(0, 2, 200)})
    model = ChampionModel.fit(frame)
    assert model.predict_proba(frame).shape == (200,)
