"""Challenger algorithm selection and benchmark (G4.4)."""
import importlib.util

import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from credit_risk_model.data import synthetic
from credit_risk_model.models import predict_pd
from credit_risk_model.models.challenger import UnsupportedAlgorithmError, build_challenger
from credit_risk_model.validation.challenger_bench import benchmark

EXCLUDE = (synthetic.DATE_COLUMN,)


@pytest.fixture(scope="module")
def frame():
    return synthetic.generate(2000, seed=3)


@pytest.mark.parametrize("algorithm,cls", [("histgb", HistGradientBoostingClassifier),
                                           ("random_forest", RandomForestClassifier)])
def test_algorithm_builds_expected_model(frame, algorithm, cls):
    pipe = build_challenger(frame, algorithm=algorithm, exclude=EXCLUDE)
    assert isinstance(pipe.named_steps["model"], cls)
    assert pipe.named_steps["model"].class_weight is None
    p = predict_pd(pipe, frame)
    assert p.shape == (len(frame),) and ((p >= 0) & (p <= 1)).all()


def test_default_algorithm_is_histgb(frame):
    pipe = build_challenger(frame, exclude=EXCLUDE)
    assert isinstance(pipe.named_steps["model"], HistGradientBoostingClassifier)


def test_unsupported_algorithm_raises(frame):
    with pytest.raises(UnsupportedAlgorithmError, match="Unsupported"):
        build_challenger(frame, algorithm="magic", exclude=EXCLUDE)


@pytest.mark.parametrize("algorithm,module", [("xgboost", "xgboost"), ("lightgbm", "lightgbm")])
def test_optional_algorithms_fail_clearly_without_dependency(frame, algorithm, module):
    if importlib.util.find_spec(module):
        pipe = build_challenger(frame, algorithm=algorithm, exclude=EXCLUDE)
        assert predict_pd(pipe, frame).shape == (len(frame),)
    else:
        with pytest.raises(RuntimeError, match="challengers"):
            build_challenger(frame, algorithm=algorithm, exclude=EXCLUDE)


def test_benchmark_sorts_by_auroc():
    y = [0, 0, 1, 1, 0, 1]
    rows = benchmark(y, {"good": [.1, .2, .9, .8, .3, .7], "bad": [.9, .8, .1, .2, .7, .3]})
    assert [r["model"] for r in rows] == ["good", "bad"]
    assert {"auroc", "gini", "ks", "brier", "log_loss"} <= set(rows[0])
