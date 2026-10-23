"""Monitoring outputs: summary JSON, per-feature CSV and a Markdown report."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .drift import alert_level, worst

RECOMMENDATIONS = {
    "escalate": "Escalate to model validation. Consider model review or suspension.",
    "investigate": "Investigate the flagged features and performance. Schedule a review.",
    "ok": "No action required. Continue routine monitoring.",
}


def write_summary(
    out_dir: Path, report_path: Path, *, model_name: str, model_version: str,
    reference_meta: dict[str, Any], current_meta: dict[str, Any],
    population_psi: float, feature_metrics: list[dict[str, Any]],
    thresholds: dict[str, float], performance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write monitoring_summary.json and psi_csi_metrics.csv to ``out_dir`` and the
    Markdown report to ``report_path``. Returns the summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pop_level = alert_level(population_psi, thresholds["psi_investigate"],
                            thresholds["psi_escalate"])
    perf_level = (performance or {}).get("level", "ok")
    overall = worst([pop_level, perf_level] + [f["level"] for f in feature_metrics])
    summary = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_name": model_name, "model_version": model_version,
        "reference": reference_meta, "current": current_meta,
        "population_psi": round(population_psi, 4),
        "population_level": pop_level,
        "performance": performance,
        "overall_level": overall,
        "features_exceeding_thresholds": [f["feature"] for f in feature_metrics
                                          if f["level"] != "ok"],
        "recommended_action": RECOMMENDATIONS[overall],
        "thresholds": thresholds,
    }
    (out_dir / "monitoring_summary.json").write_text(json.dumps(summary, indent=2),
                                                     encoding="utf-8")
    with (out_dir / "psi_csi_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["feature", "csi", "level"])
        writer.writeheader()
        for row in feature_metrics:
            writer.writerow({"feature": row["feature"], "csi": round(row["csi"], 4),
                             "level": row["level"]})
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_markdown(summary, feature_metrics), encoding="utf-8")
    return summary


def _render_markdown(summary: dict, features: list[dict]) -> str:
    lines = [
        f"# Monitoring Report — {summary['model_name']} {summary['model_version']}", "",
        f"- **Run timestamp:** {summary['run_timestamp_utc']}",
        f"- **Overall level:** `{summary['overall_level']}`",
        (f"- **Population PSI (predicted PD):** {summary['population_psi']} "
         f"(`{summary['population_level']}`)"),
        f"- **Recommended action:** {summary['recommended_action']}", "",
        "## Feature stability (CSI on model inputs)", "",
        "| Feature | CSI | Level |",
        "|---|---:|---|",
    ]
    lines += [f"| `{f['feature']}` | {f['csi']:.4f} | {f['level']} |" for f in features]
    perf = summary.get("performance")
    lines += ["", "## Performance", ""]
    if perf and perf.get("status") == "assessed":
        lines += [
            "| Metric | Reference | Current |", "|---|---:|---:|",
            f"| AUROC | {perf['reference_auroc']:.4f} | {perf['current_auroc']:.4f} |",
            f"| Mean predicted PD | {perf['reference_mean_pd']:.4f} | {perf['current_mean_pd']:.4f} |",
            (f"| Observed default rate | {perf['reference_default_rate']:.4f} | "
            f"{perf['current_default_rate']:.4f} |"),
            "", f"AUROC drop: {perf['auroc_drop']:.4f} (`{perf['level']}`)",
        ]
    else:
        reason = (perf or {}).get("reason", "no labels in the current sample")
        lines.append(f"Not assessed: {reason}.")
    return "\n".join(lines) + "\n"
