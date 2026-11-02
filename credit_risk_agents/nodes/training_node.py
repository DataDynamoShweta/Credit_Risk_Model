# credit_risk_agents/nodes/training_node.py
from __future__ import annotations

import datetime
from typing import Any, Dict, List
from credit_risk_agents.state import CreditRiskState
from credit_risk_agents.tools.pipeline_runner import PipelineRunner


def _now() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def training_node(state: CreditRiskState) -> CreditRiskState:
    run_id = state.get("run_id", "default")
    runner = PipelineRunner(run_id=run_id)

    events: List[Dict[str, Any]] = [
        {
            "time": _now(),
            "type": "info",
            "stage": "training",
            "agent": "Modeling",
            "msg": f"Fitting champion scorecard: {state.get('champion', 'Logistic Regression')}",
        },
        {
            "time": _now(),
            "type": "info",
            "stage": "training",
            "agent": "Challenger",
            "msg": f"Fitting challenger model: {state.get('challenger', 'HistGradientBoosting')}",
        },
    ]

    result = runner.run_training(
        dataset_path=state.get("dataset_path", "synthetic"),
        target_column=state.get("target_column", "target"),
        random_seed=state.get("random_seed", 42),
        challenger=state.get("challenger", "histgb"),
        oot_pct=state.get("oot_pct", 20.0),
    )

    if not result.ok:
        err_msg = f"Training failed (exit code {result.exit_code}): {result.stderr[-400:] if result.stderr else 'Unknown error'}"
        events.append({
            "time": _now(),
            "type": "error",
            "stage": "training",
            "agent": "Modeling",
            "msg": err_msg,
        })
        return {
            "current_stage": "failed",
            "error_message": err_msg,
            "events": events,
        }

    events.append({
        "time": _now(),
        "type": "ok",
        "stage": "training",
        "agent": "Modeling",
        "msg": "Champion and challenger models successfully fitted and serialized.",
    })

    artifacts = dict(state.get("artifact_paths") or {})
    artifacts.update(result.artifact_paths)

    return {
        "current_stage": "validation",
        "artifact_paths": artifacts,
        "stages_completed": ["training"],
        "events": events,
    }

