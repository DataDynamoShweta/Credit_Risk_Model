# Independent Model Validation Report

- **Model:** Logistic regression champion
- **Validation date (UTC):** 2026-10-22
- **Validation status:** Review required
- **Model purpose:** Probability-of-default estimation

## Executive summary
Automated results require interpretation by an independent validator.

## Quantitative results

### discrimination

- **auroc:** 0.7044403450426107

- **gini:** 0.4088806900852213

- **ks:** 0.36764705882352944

- **auroc_ci:** [0.6464812694309177, 0.76344523360284]

- **bootstrap_valid_samples:** 200


### calibration

- **brier:** 0.20788142544460253

- **log_loss:** 0.6066499864958742

- **hosmer_lemeshow_statistic:** 858.973840263207

- **reliability:** [{'count': 120, 'observed_rate': 0.008333333333333333, 'mean_pd': 0.18728140926709252}, {'count': 120, 'observed_rate': 0.025, 'mean_pd': 0.24602097129455}, {'count': 120, 'observed_rate': 0.041666666666666664, 'mean_pd': 0.2877645282156821}, {'count': 120, 'observed_rate': 0.03333333333333333, 'mean_pd': 0.3311193359192588}, {'count': 120, 'observed_rate': 0.025, 'mean_pd': 0.37724900762327335}, {'count': 120, 'observed_rate': 0.05, 'mean_pd': 0.43140191777451914}, {'count': 120, 'observed_rate': 0.03333333333333333, 'mean_pd': 0.4912374630480436}, {'count': 120, 'observed_rate': 0.075, 'mean_pd': 0.5509300973056318}, {'count': 120, 'observed_rate': 0.11666666666666667, 'mean_pd': 0.6257745732201356}, {'count': 120, 'observed_rate': 0.15833333333333333, 'mean_pd': 0.7617296519474067}]



## Findings and remediation

- No findings were supplied. Independent reviewers must confirm this is appropriate.


## Limitations, assumptions and use restrictions
Not approved for production use.

## Governance and regulatory considerations
This report is an evidence template, not a declaration of compliance with PRA SS1/23, IFRS 9 or regulatory capital requirements. Document model scope, materiality, validation independence, data lineage, conceptual soundness, implementation testing, outcomes analysis, approval, limitations and remediation in accordance with the institution's approved policy.