import pandas as pd

from credit_risk_model.features.scorecard import probability_to_score
from credit_risk_model.features.woe_binning import calculate_woe_iv
from credit_risk_model.validation.calibration import calibration
from credit_risk_model.validation.oot import oot_split


def test_woe_iv_nonnegative():
    x=pd.Series([1,2,3,4,5,6,7,8])
    y=pd.Series([0,0,0,0,1,1,1,1])
    result=calculate_woe_iv(x,y,bins=4)
    assert result.iv >= 0
    assert "woe" in result.table

def test_score_decreases_as_pd_increases():
    assert probability_to_score(.01) > probability_to_score(.2)

def test_calibration_returns_metrics():
    result=calibration([0,0,1,1],[.1,.2,.8,.9],groups=2)
    assert "brier" in result and "hosmer_lemeshow_statistic" in result

def test_oot_split_is_chronological():
    frame=pd.DataFrame({"dt":pd.date_range("2020-01-01", periods=10), "x":range(10)})
    train,test=oot_split(frame,"dt",test_fraction=.2)
    assert train.dt.max() < test.dt.min()
