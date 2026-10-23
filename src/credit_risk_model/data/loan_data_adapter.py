"""Adapter from the LendingClub-style ``loan_data_2015.csv`` to a model-ready frame.

Target event (explicit and auditable):

* bad  (1): ``loan_status`` in ``bad_statuses``  (default: Charged Off, Default)
* good (0): ``loan_status`` in ``good_statuses`` (default: Fully Paid)
* every other status (Current, Issued, Late, In Grace Period, ...) is indeterminate and
  excluded, and the excluded counts are reported.

Only allow-listed origination-time features are kept, and the allow-list is checked
against a deny-list of post-origination fields (payments, recoveries, last payment, etc.)
so leakage columns can never be included by configuration.

Column names and formats were checked against the copy of ``loan_data_2015.csv`` in this
repository (421,094 rows, 74 columns). Only ~26,000 of those loans have a final outcome;
keeping resolved loans only over-represents early pay-offs and early defaults, so results
on this data are not representative of the full 2015 book.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

POST_ORIGINATION_DENY_LIST = frozenset({
    "loan_status", "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries",
    "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
    "last_credit_pull_d", "pymnt_plan", "funded_amnt_inv",
})
IDENTIFIER_COLUMNS = frozenset({"id", "member_id", "url", "desc", "emp_title", "title", "zip_code"})

DEFAULT_NUMERIC = ("loan_amnt", "term", "int_rate", "installment", "emp_length", "annual_inc",
                   "dti", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
                   "revol_bal", "revol_util", "total_acc")
DEFAULT_CATEGORICAL = ("grade", "home_ownership", "verification_status", "purpose")


class AdapterError(ValueError):
    pass


@dataclass
class AdapterConfig:
    status_column: str = "loan_status"
    issue_date_column: str = "issue_d"
    issue_date_format: str = "%b-%y"   # e.g. "Sep-15"
    output_date_column: str = "application_date"
    target: str = "target"
    bad_statuses: frozenset = frozenset({"Charged Off", "Default"})
    good_statuses: frozenset = frozenset({"Fully Paid"})
    numeric_features: tuple = DEFAULT_NUMERIC
    categorical_features: tuple = DEFAULT_CATEGORICAL
    deny_list: frozenset = field(default=POST_ORIGINATION_DENY_LIST)

    def features(self) -> list[str]:
        return [*self.numeric_features, *self.categorical_features]

    def check(self) -> None:
        leaked = sorted(set(self.features()) & (self.deny_list | IDENTIFIER_COLUMNS))
        if leaked:
            raise AdapterError(f"post-origination or identifier columns in feature list: {leaked}")
        if self.bad_statuses & self.good_statuses:
            raise AdapterError("a status cannot be both good and bad")


def _to_number(series: pd.Series) -> pd.Series:
    """Parse values such as ' 36 months', '13.5%', '10+ years', '< 1 year'."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    text = series.astype("string").str.strip()
    text = text.str.replace("< 1", "0", regex=False)
    extracted = text.str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def adapt(raw: pd.DataFrame, config: AdapterConfig | None = None) -> tuple[pd.DataFrame, dict]:
    """Return (model-ready frame, report). The frame has features, date and target."""
    config = config or AdapterConfig()
    config.check()
    required = [config.status_column, config.issue_date_column, *config.features()]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise AdapterError(f"missing required columns: {missing}")

    status = raw[config.status_column].astype("string").str.strip()
    is_bad, is_good = status.isin(config.bad_statuses), status.isin(config.good_statuses)
    keep = is_bad | is_good
    excluded = status[~keep].value_counts(dropna=False)

    frame = pd.DataFrame(index=raw.index[keep])
    for column in config.numeric_features:
        frame[column] = _to_number(raw.loc[keep, column])
    for column in config.categorical_features:
        frame[column] = raw.loc[keep, column].astype("string").str.strip().fillna("MISSING")
        frame[column] = frame[column].astype(object)
    dates = pd.to_datetime(raw.loc[keep, config.issue_date_column],
                           format=config.issue_date_format, errors="coerce")
    if dates.isna().any():
        raise AdapterError(f"{int(dates.isna().sum())} rows have an unparseable "
                           f"{config.issue_date_column}")
    frame[config.output_date_column] = dates
    frame[config.target] = is_bad[keep].astype(int)
    if frame[config.target].nunique() != 2:
        raise AdapterError("adapted data does not contain both good and bad outcomes")
    report = {
        "input_rows": len(raw), "output_rows": len(frame),
        "bad_rows": int(frame[config.target].sum()),
        "default_rate": float(frame[config.target].mean()),
        "excluded_indeterminate": {str(k): int(v) for k, v in excluded.items()},
        "bad_statuses": sorted(config.bad_statuses),
        "good_statuses": sorted(config.good_statuses),
        "features": config.features(),
        "dropped_deny_listed": sorted(set(raw.columns) & config.deny_list),
        "date_range": [str(dates.min().date()), str(dates.max().date())],
    }
    return frame.reset_index(drop=True), report
