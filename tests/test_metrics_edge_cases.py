"""Scorecard, discrimination, calibration, loader, alert and report-builder edge cases."""
import numpy as np
import pytest

from credit_risk_model.data.loader import load_csv
from credit_risk_model.features.scorecard import probability_to_score
from credit_risk_model.governance.report_builder import fmt, render_report
from credit_risk_model.monitoring.alerts import build_alerts
from credit_risk_model.monitoring.drift import alert_level, drift_summary, worst
from credit_risk_model.validation.calibration import calibration
from credit_risk_model.validation.discrimination import discrimination


def test_score_strictly_decreases_with_pd():
    scores = probability_to_score(np.linspace(0.001, 0.999, 50))
    assert np.all(np.diff(scores) <= 0) and scores[0] > scores[-1]


def test_score_handles_pd_bounds():
    assert np.isfinite(probability_to_score(0.0)) and np.isfinite(probability_to_score(1.0))
    assert probability_to_score(0.0) > probability_to_score(1.0)


def test_score_is_reproducible_and_anchored():
    # At base odds 50:1 good:bad (PD = 1/51) the score equals the base score.
    assert probability_to_score(1 / 51) == 600
    assert probability_to_score(0.05) == probability_to_score(0.05)


def test_score_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        probability_to_score(0.1, pdo=0)


def test_discrimination_single_class_rejected():
    with pytest.raises(ValueError, match="both target classes"):
        discrimination([0, 0, 0], [0.1, 0.2, 0.3])


def test_calibration_in_the_large_reported():
    result = calibration([0, 0, 0, 1], [0.25, 0.25, 0.25, 0.25], groups=2)
    assert result["mean_predicted_pd"] == pytest.approx(0.25)
    assert result["observed_default_rate"] == pytest.approx(0.25)
    assert result["calibration_ratio"] == pytest.approx(1.0)


def test_calibration_single_class_has_no_ratio():
    result = calibration([0, 0, 0], [0.1, 0.2, 0.3])
    assert result["calibration_ratio"] is None and result["brier"] > 0


def test_loader_rejects_duplicate_headers(tmp_path):
    path = tmp_path / "dup.csv"
    path.write_text("a,a,b\n1,2,3\n")
    with pytest.raises(ValueError, match="duplicate column"):
        load_csv(path)


def test_loader_rejects_empty_and_missing(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_csv(empty)
    header_only = tmp_path / "header.csv"
    header_only.write_text("a,b\n")
    with pytest.raises(ValueError, match="empty"):
        load_csv(header_only)
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "nope.csv")


def test_loader_checks_target(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="Target column"):
        load_csv(path, target="target")
    assert list(load_csv(path).columns) == ["a", "b"]


def test_alert_vocabulary_is_consistent():
    assert [alert_level(v, 0.1, 0.25) for v in (0.05, 0.1, 0.3)] == ["ok", "investigate", "escalate"]
    assert worst(["ok", "escalate", "investigate"]) == "escalate"
    import pandas as pd
    ref = pd.DataFrame({"a": [0] * 80 + [1] * 20, "b": range(100)})
    cur = pd.DataFrame({"a": [0] * 40 + [1] * 60, "b": range(100)})
    summary = drift_summary(ref, cur, features=["a", "b"])
    alerts = build_alerts(summary, model_name="m", run_id="r1")
    assert summary["features"]["a"]["level"] == "escalate"
    assert alerts == [{"level": "escalate", "model": "m", "feature": "a",
                       "psi": summary["features"]["a"]["psi"], "run_id": "r1"}]


def test_report_builder_formats_values(tmp_path):
    template = tmp_path / "t.md.j2"
    template.write_text("{{ x | fmt }} {{ y | pct }} {{ z | check }} {{ n | fmt }}")
    assert render_report({"x": 0.12345, "y": 0.05, "z": True, "n": None}, template) == \
        "0.123 5.0% ✅ pass —"
    assert fmt(1234) == "1,234"


def test_report_builder_requires_template(tmp_path):
    with pytest.raises(FileNotFoundError):
        render_report({}, tmp_path / "missing.j2")
