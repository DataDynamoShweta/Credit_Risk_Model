"""Alert records built from drift results. Delivery (email, Slack, tickets) is out of scope."""
def build_alerts(summary: dict, *, model_name: str, run_id: str | None = None):
    """One alert per feature whose level is investigate or escalate."""
    return [{"level": summary["features"][name]["level"], "model": model_name,
             "feature": name, "psi": summary["features"][name]["psi"], "run_id": run_id}
            for name in summary.get("alert_features", [])]
