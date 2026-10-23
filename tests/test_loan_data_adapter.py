"""LendingClub adapter: explicit target, leakage deny-list, type parsing (G12)."""
import pandas as pd
import pytest

from credit_risk_model.data.loan_data_adapter import (
    POST_ORIGINATION_DENY_LIST,
    AdapterConfig,
    AdapterError,
    adapt,
)


def fixture_frame() -> pd.DataFrame:
    """Small frame with the column names and formats of loan_data_2015.csv."""
    statuses = ["Fully Paid", "Charged Off", "Current", "Default", "Late (31-120 days)",
                "In Grace Period", "Fully Paid", "Issued", "Fully Paid", "Charged Off"]
    n = len(statuses)
    return pd.DataFrame({
        "id": range(n), "member_id": range(100, 100 + n), "loan_status": statuses,
        "issue_d": ["Jan-15", "Feb-15", "Mar-15", "Apr-15", "May-15", "Jun-15", "Jul-15",
                    "Aug-15", "Sep-15", "Oct-15"],
        "loan_amnt": [10000] * n, "term": [" 36 months", " 60 months"] * 5,
        "int_rate": [12.29, 19.99] * 5, "installment": [300.0] * n,
        "emp_length": ["10+ years", "< 1 year", "3 years", None, "1 year"] * 2,
        "annual_inc": [60000.0] * n, "dti": [18.0] * n, "delinq_2yrs": [0] * n,
        "inq_last_6mths": [1] * n, "open_acc": [8] * n, "pub_rec": [0] * n,
        "revol_bal": [5000] * n, "revol_util": [55.5] * n, "total_acc": [20] * n,
        "grade": ["B", "D"] * 5, "home_ownership": ["RENT", "MORTGAGE"] * 5,
        "verification_status": ["Verified"] * n, "purpose": ["credit_card"] * n,
        "total_pymnt": [1.0] * n, "recoveries": [0.0] * n, "last_pymnt_amnt": [50.0] * n,
        "url": ["x"] * n, "desc": ["free text"] * n,
    })


def test_target_definition_and_exclusions():
    frame, report = adapt(fixture_frame())
    assert len(frame) == 6  # 3 Fully Paid + 2 Charged Off + 1 Default
    assert frame["target"].sum() == 3
    assert report["excluded_indeterminate"] == {"Current": 1, "Late (31-120 days)": 1,
                                                "In Grace Period": 1, "Issued": 1}


def test_no_post_origination_or_identifier_columns():
    frame, report = adapt(fixture_frame())
    assert not set(frame.columns) & POST_ORIGINATION_DENY_LIST
    assert not {"id", "member_id", "url", "desc"} & set(frame.columns)
    assert "total_pymnt" in report["dropped_deny_listed"]


def test_type_parsing_and_dates():
    frame, _ = adapt(fixture_frame())
    assert frame["term"].tolist()[:2] == [36.0, 60.0]
    assert frame.loc[0, "emp_length"] == 10 and frame.loc[1, "emp_length"] == 0
    assert frame["application_date"].min() == pd.Timestamp("2015-01-01")
    assert frame["grade"].dtype == object


def test_config_cannot_include_leakage_columns():
    with pytest.raises(AdapterError, match="post-origination"):
        adapt(fixture_frame(), AdapterConfig(numeric_features=("loan_amnt", "total_pymnt")))


def test_missing_columns_reported():
    with pytest.raises(AdapterError, match="missing required columns"):
        adapt(fixture_frame().drop(columns=["dti"]))


def test_unparseable_issue_date_rejected():
    raw = fixture_frame()
    raw.loc[0, "issue_d"] = "sometime"
    with pytest.raises(AdapterError, match="unparseable"):
        adapt(raw)


def test_output_trains_the_champion():
    from credit_risk_model.models.champion import ChampionModel
    raw = pd.concat([fixture_frame()] * 20, ignore_index=True)
    frame, _ = adapt(raw)
    model = ChampionModel.fit(frame, exclude=("application_date",))
    assert model.predict_proba(frame).shape == (len(frame),)
