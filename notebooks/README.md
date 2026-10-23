# Original exploratory notebook

`Final.ipynb` is the original single-cell analysis that preceded this package. It is kept
unchanged for reference. **Do not reuse its approach or its results.** The issues below
were identified by reading the notebook's code; they are why the package was rebuilt.

| # | Issue | Where | Consequence | How the package handles it |
|---|---|---|---|---|
| 1 | Imputers are fitted on the **full dataset before the train/test split**, and `get_dummies` / `StandardScaler` are applied to all rows before splitting. | `preprocess_data`, `generate_scorecard` | Preprocessing leakage: test rows influence the transformations. | Preprocessing lives inside a scikit-learn `Pipeline` fitted on the training split only (`models/champion.py`). |
| 2 | Cross-validation runs on data that was **already scaled** with full-data statistics. | `cross_val_score(model, X_train, ...)` after `StandardScaler().fit_transform(X)` | CV scores are optimistic. | Validation uses a held-out test sample and an out-of-time sample (`validation/suite.py`). |
| 3 | Target is `loan_status == 'Charged Off'`; **every other status counts as good**, including `Current`, `Issued`, `Late` and `In Grace Period`. | `generate_scorecard` | Immature 2015 loans are treated as non-defaults; `Default` and late loans are labelled good. | Explicit event definition: bad = Charged Off / Default, good = Fully Paid, everything else excluded and counted (`data/loan_data_adapter.py`). |
| 4 | Only ID/date/text columns are dropped. **Post-origination fields** (`total_pymnt`, `recoveries`, `last_pymnt_amnt`, `out_prncp`, …) stay in the features. | `columns_to_drop` | Outcome leakage; AUROC would be unrealistically high. | Allow-list of origination-time features, checked against a post-origination deny-list. |
| 5 | `scale_score` computes `(p − 0.5) × 600 + 300`, so a **higher default probability gives a higher score**, and it is also applied to the AUROC. | `scale_score` | Score direction is the reverse of the PDO convention; "scaled AUC" has no meaning. | PDO scaling where a higher PD gives a lower score, with tests (`features/scorecard.py`). |
| 6 | **No stratified split** despite class imbalance. | `train_test_split(..., random_state=...)` | Default rate can differ between train and test. | Stratified train/test split (`split_dataset`). |
| 7 | Confusion matrix at the default **0.5 threshold** is reported as a scorecard metric. | `confusion_matrix(y_test, model.predict(X_test))` | With ~12% bad rate, 0.5 is arbitrary and nearly every loan is predicted good. | No decision threshold is applied; decisions need an approved policy. |
| 8 | Scores are produced for the **whole dataset**, including training rows, and mapped to `member_id`. | `all_predictions = model.predict_proba(X_scaled)` | In-sample scores look better than they are; personal identifiers mixed with scores. | Only held-out rows are scored for validation; identifiers are dropped by the adapter. |

Issue 4 is confirmed against the data: `loan_data_2015.csv` in this repository contains
`total_pymnt`, `recoveries`, `last_pymnt_amnt` and the other deny-listed columns, and the
notebook does not drop them.

To run the package on the same data instead:

```bash
python pipelines/train.py    --data loan_data_2015.csv --adapter lending_club
python pipelines/validate.py --data loan_data_2015.csv --adapter lending_club
```
