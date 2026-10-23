"""Out-of-time evaluation and chronological split edge cases (G4.2)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from credit_risk_model.data import synthetic
from credit_risk_model.models.champion import ChampionModel
from credit_risk_model.validation.oot import evaluate_oot, oot_split
from credit_risk_model.validation.vintage import vintage_performance

DATE = synthetic.DATE_COLUMN


@pytest.fixture(scope="module")
def frame():
    return synthetic.generate(3000, seed=11)


def test_evaluate_oot_with_champion_model(frame):
    model = ChampionModel.fit(frame, exclude=(DATE,))
    result = evaluate_oot(model, frame, "target", DATE)
    assert result["status"] == "assessed"
    assert 0.5 < result["auroc"] <= 1


def test_evaluate_oot_with_raw_sklearn_model(frame):
    numeric = frame[["loan_amnt", "int_rate", "annual_inc", "dti", "emp_length", "target", DATE]]
    model = LogisticRegression(max_iter=1000).fit(numeric.drop(columns=["target", DATE]),
                                                   numeric["target"])
    result = evaluate_oot(model, numeric, "target", DATE)
    assert result["status"] == "assessed" and "auroc" in result


def test_single_class_window_is_not_assessed(frame):
    model = ChampionModel.fit(frame, exclude=(DATE,))
    window = frame.copy()
    window.loc[window.index[len(window) // 2:], "target"] = 0
    result = evaluate_oot(model, window, "target", DATE, test_fraction=0.3)
    assert result["status"] == "not_assessed"
    assert "one outcome class" in result["reason"]
    assert "brier" in result


def test_unsorted_dates_are_sorted():
    frame = pd.DataFrame({"dt": pd.to_datetime(["2020-03-01", "2020-01-01", "2020-02-01",
                                                "2020-05-01", "2020-04-01"]), "x": range(5)})
    train, test = oot_split(frame, "dt", test_fraction=0.4)
    assert train["dt"].max() < test["dt"].min()


def test_ties_never_straddle_the_cutoff():
    dates = pd.to_datetime(["2020-01-01"] * 4 + ["2020-02-01"] * 4 + ["2020-03-01"] * 2)
    frame = pd.DataFrame({"dt": dates, "x": range(10)})
    train, test = oot_split(frame, "dt", test_fraction=0.5)
    assert not set(train["dt"]) & set(test["dt"])


def test_missing_dates_are_rejected():
    frame = pd.DataFrame({"dt": ["2020-01-01", None, "2020-03-01"], "x": [1, 2, 3]})
    with pytest.raises(ValueError, match="missing or unparseable"):
        oot_split(frame, "dt")


def test_all_rows_on_one_date_rejected():
    frame = pd.DataFrame({"dt": ["2020-01-01"] * 5, "x": range(5)})
    with pytest.raises(ValueError, match="cut-off date"):
        oot_split(frame, "dt")


def test_vintage_performance_groups_by_quarter():
    dates = pd.date_range("2020-01-01", periods=360, freq="D")
    y = np.tile([0, 0, 0, 1], 90)
    rows = vintage_performance(dates, y, np.linspace(0.01, 0.9, 360))
    assert [r["vintage"] for r in rows] == ["2020Q1", "2020Q2", "2020Q3", "2020Q4"]
    assert all(r["status"] == "assessed" for r in rows)


def test_vintage_single_class_period_not_assessed():
    rows = vintage_performance(pd.to_datetime(["2020-01-05", "2020-02-05"]), [0, 0], [0.1, 0.2])
    assert rows[0]["status"] == "not_assessed"
