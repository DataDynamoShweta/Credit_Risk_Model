import numpy as np
from credit_risk_model.validation.discrimination import discrimination

def test_perfect_ranking_has_auc_one():
    result = discrimination([0,0,1,1], [0.1,0.2,0.8,0.9])
    assert result["auroc"] == 1.0
    assert result["gini"] == 1.0
    assert result["ks"] > 0.9

def test_bootstrap_ci_present():
    result = discrimination([0,1,0,1,0,1], [0.1,0.8,0.3,0.7,0.2,0.9], bootstrap=20)
    assert len(result["auroc_ci"]) == 2
