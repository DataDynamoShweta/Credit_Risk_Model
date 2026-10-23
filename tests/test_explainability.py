import pytest

from credit_risk_model.explainability import shap_explainer


def test_availability_flag():
    assert isinstance(shap_explainer.is_available(), bool)


def test_raises_without_shap():
    if shap_explainer.is_available():
        pytest.skip("SHAP installed — cannot test the missing-dep path")
    with pytest.raises(RuntimeError, match="SHAP not installed"):
        shap_explainer.global_importance(None, None)
