"""Simple alert policy. Connect to email/Slack/incident tooling outside this library."""
def build_alerts(summary: dict, *, model_name: str, run_id: str | None = None):
    return [{"severity": summary["features"][name]["status"], "model": model_name,
             "feature": name, "psi": summary["features"][name]["psi"], "run_id": run_id}
            for name in summary.get("alert_features", [])]
