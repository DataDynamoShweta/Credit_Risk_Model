# Data

## Canonical synthetic data

`src/credit_risk_model/data/synthetic.py` is the only synthetic generator. Training,
validation, the API examples, the tests and the nightly monitoring workflow all use it
(`scripts/generate_synthetic.py` writes it to CSV).

| Column | Type | Notes |
|---|---|---|
| `application_date` | date | Spread over 36 months from 2018-01-01; enables OOT and vintage analysis |
| `loan_amnt`, `int_rate`, `annual_inc`, `dti`, `emp_length` | numeric | |
| `home_ownership` | categorical | `RENT`, `MORTGAGE`, `OWN` |
| `target` | 0/1 | Default flag, about 5% |

`--drift` shifts `int_rate`, `dti` and the `home_ownership` mix. The data is invented;
results on it say nothing about a real portfolio.

## Splits

`validation/suite.py::split_dataset` is shared by training and validation. With a date
column, the latest 20% of applications (cut on a date boundary, so no date is in both
samples) is the out-of-time sample, and the rest is split 75/25 stratified into train and
test. Training records the split parameters and a fingerprint of the data; validation
rebuilds the same split and scores only test and OOT rows.

## Real data: `loan_data_2015.csv`

Stored with Git LFS (175 MB, 421,094 rows, 74 columns). Run `git lfs pull` to fetch it.

```bash
python pipelines/train.py    --data loan_data_2015.csv --adapter lending_club
python pipelines/validate.py --data loan_data_2015.csv --adapter lending_club
```

`src/credit_risk_model/data/loan_data_adapter.py`:

- **Target:** bad = `Charged Off`, `Default`; good = `Fully Paid`; every other status is
  excluded and counted in the adapter report.
- **Date:** `issue_d` (format `Sep-15`) → `application_date`.
- **Features:** an allow-list of origination-time fields (loan amount, term, rate,
  instalment, employment length, income, DTI, credit-bureau counts, revolving balance and
  utilisation, grade, home ownership, verification status, purpose). It is checked against
  a deny-list of post-origination fields (`total_pymnt`, `recoveries`, `last_pymnt_amnt`,
  `out_prncp`, …) and identifiers, so leakage columns cannot be configured in.

Run on the copy in this repository (October 2026): 26,144 resolved loans kept
(3,160 bad, 12.1%), 2015-01 to 2015-12; held-out AUROC 0.73, calibration ratio 1.02,
out-of-time AUROC 0.68.

Caveats:

- Only about 6% of 2015 loans have a final outcome in this snapshot. Keeping resolved loans
  only over-represents early pay-offs and early defaults, and later vintages have had less
  time to default, so out-of-time results are biased. The results are not representative of
  the 2015 book.
- `int_rate` and `grade` are set by the lender's own pricing model at origination; they are
  legitimate origination-time inputs but make this model partly a re-statement of that one.
- No data dictionary, licence review or lineage check has been done.
