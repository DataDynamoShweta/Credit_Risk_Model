"""Feature drift summaries using one alert vocabulary: ok / investigate / escalate."""
from ..validation.stability import csi_by_feature

LEVELS = ("ok", "investigate", "escalate")
LEVEL_ORDER = {level: i for i, level in enumerate(LEVELS)}


def alert_level(value: float | None, investigate: float, escalate: float) -> str:
    if value is None:
        return "ok"
    if value >= escalate:
        return "escalate"
    if value >= investigate:
        return "investigate"
    return "ok"


def worst(levels) -> str:
    return max(levels, key=lambda v: LEVEL_ORDER[v], default="ok")


def drift_summary(reference, current, *, features: list[str] | None = None,
                  thresholds: dict | None = None):
    """PSI per feature with an alert level. ``features`` should be the model inputs only."""
    thresholds = thresholds or {"investigate": .10, "escalate": .25}
    values = csi_by_feature(reference, current, features=features)
    results = {name: {"psi": value,
                      "level": alert_level(value, thresholds["investigate"], thresholds["escalate"])}
               for name, value in values.items()}
    return {"features": results,
            "alert_features": [k for k, v in results.items() if v["level"] != "ok"]}


def evidently_report(reference, current):
    """Optional Evidently (>=0.7) drift report; returns None if Evidently is not installed."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        return None
    report = Report([DataDriftPreset()])
    return report.run(reference_data=reference, current_data=current)
