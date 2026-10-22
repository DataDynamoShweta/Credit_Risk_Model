"""Drift reporting; Evidently integration is optional and version-dependent."""
from ..validation.stability import csi_by_feature


def drift_summary(reference, current, *, thresholds=None):
    thresholds = thresholds or {"warning": .1, "critical": .25}
    features = csi_by_feature(reference, current)
    results = {}
    for name, value in features.items():
        status = "normal"
        if value >= thresholds["critical"]:
            status = "critical"
        elif value >= thresholds["warning"]:
            status = "warning"
        results[name] = {"psi": value, "status": status}
    return {"features": results, "alert_features": [k for k,v in results.items() if v["status"] != "normal"]}

def evidently_report(reference, current):
    """Return None if Evidently is not installed; callers can use native PSI regardless."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        return None
    report = Report([DataDriftPreset()])
    return report.run(reference_data=reference, current_data=current)
