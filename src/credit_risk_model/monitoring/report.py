"""Renders monitoring summary JSON, CSV, and Markdown report."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ALERT_ORDER = {"ok": 0, "investigate": 1, "escalate": 2}


def alert_level(value: float, investigate: float, alert: float) -> str:
    if value >= alert:
        return "escalate"
    if value >= investigate:
        return "investigate"
    return "ok"


def write_summary(
    out_dir: Path, *, model_name: str, model_version: str,
    reference_meta: dict[str, Any], current_meta: dict[str, Any],
    population_psi: float, feature_metrics: list[dict[str, Any]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pop_level = alert_level(
        population_psi, thresholds.get("psi_investigate", 0.10),
        thresholds.get("psi_alert", 0.25))
    breaches = [f for f in feature_metrics
                if f["csi_level"] in ("investigate", "escalate")]
    overall = max([pop_level] + [f["csi_level"] for f in feature_metrics],
                  key=lambda v: ALERT_ORDER[v])
    summary = {
        "run_timestamp_utc": datetime.now(timezone.utc)
            .isoformat(timespec="seconds"),
        "model_name": model_name, "model_version": model_version,
        "reference": reference_meta, "current": current_meta,
        "population_psi": round(population_psi, 4),
        "population_alert_level": pop_level,
        "overall_alert_level": overall,
        "features_exceeding_thresholds": [f["feature"] for f in breaches],
        "recommended_action": _recommend(overall),
        "thresholds": thresholds,
    }
    (out_dir / "monitoring_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "psi_csi_metrics.csv").open(
        "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh,
            fieldnames=["feature", "psi", "csi", "csi_level"])
        w.writeheader()
        for row in feature_metrics:
            w.writerow({"feature": row["feature"],
                        "psi": round(row["psi"], 4),
                        "csi": round(row["csi"], 4),
                        "csi_level": row["csi_level"]})
    md = _render_markdown(summary, feature_metrics)
    md_path = out_dir.parent.parent / "reports" / "monitoring_report.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")
    return summary


def _recommend(level: str) -> str:
    if level == "escalate":
        return ("Escalate to model validation. "
                "Consider model review or retirement.")
    if level == "investigate":
        return "Investigate top CSI contributors. Schedule a refresh review."
    return "No action required. Continue routine monitoring."


def _render_markdown(summary: dict, features: list[dict]) -> str:
    lines = [
        (f"# Monitoring Report — {summary['model_name']} "
         f"v{summary['model_version']}"), "",
        f"**Run timestamp:** {summary['run_timestamp_utc']}",
        f"**Overall alert:** `{summary['overall_alert_level']}`",
        (f"**Population PSI:** {summary['population_psi']} "
         f"(`{summary['population_alert_level']}`)"), "",
        f"**Recommended action:** {summary['recommended_action']}", "",
        "## Feature stability", "",
        "| Feature | PSI | CSI | Level |",
        "|---------|-----|-----|-------|",
    ]
    for f in features:
        lines.append(f"| `{f['feature']}` | {f['psi']:.4f} | "
                     f"{f['csi']:.4f} | {f['csi_level']} |")
    return "\n".join(lines) + "\n"
