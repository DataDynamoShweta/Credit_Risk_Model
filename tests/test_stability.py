import pandas as pd
from credit_risk_model.validation.stability import psi, csi_by_feature

def test_psi_same_distribution_near_zero():
    x = pd.Series([1,2,3,4,5,6,7,8,9,10]*10)
    assert psi(x, x) < 1e-8

def test_csi_reports_shared_features():
    result = csi_by_feature(pd.DataFrame({"a":[1,2,3]}), pd.DataFrame({"a":[2,3,4],"b":[1,2,3]}))
    assert "a" in result and "b" not in result
