import pytest

from credit_risk_model.data import synthetic
from credit_risk_model.explainability import shap_explainer
from credit_risk_model.models.champion import ChampionModel


def test_availability_flag():
    assert isinstance(shap_explainer.is_available(), bool)


def test_raises_without_shap(monkeypatch):
    monkeypatch.setattr(shap_explainer, "_HAS_SHAP", False)
    with pytest.raises(RuntimeError, match="SHAP not installed"):
        shap_explainer.global_importance(None, None)


@pytest.mark.skipif(not shap_explainer.is_available(), reason="shap not installed")
def test_shap_works_with_champion_wrapper():
    frame = synthetic.generate(1500, seed=5)
    model = ChampionModel.fit(frame, exclude=(synthetic.DATE_COLUMN,))
    X = frame[model.feature_names]
    importance = shap_explainer.global_importance(model, X)
    assert {row["feature"] for row in importance} == set(model.feature_names)
    assert importance[0]["feature"] == "int_rate"  # the strongest driver in the generator
    local = shap_explainer.local_explanation(model, X, X.iloc[[0]])
    assert {c["direction"] for c in local} <= {"increases_pd", "decreases_pd"}
