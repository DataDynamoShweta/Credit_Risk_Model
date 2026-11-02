# credit_risk_agents/nodes/validation_node.py
from __future__ import annotations

import datetime
import json
from typing import Any, Dict, List
from credit_risk_agents.state import CreditRiskState
from credit_risk_agents.tools.llm_client import RiskCommentary, structured_chat
from credit_risk_agents.tools.pipeline_runner import PipelineRunner


def _now() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def validation_node(state: CreditRiskState) -> CreditRiskState:
    run_id = state.get("run_id", "default")
    runner = PipelineRunner(run_id=run_id)

    events: List[Dict[str, Any]] = [
        {
            "time": _now(),
            "type": "info",
            "stage": "validation",
            "agent": "Validation",
            "msg": "Executing independent validation suite: discrimination, calibration, stability (PSI/CSI)",
        }
    ]

    result = runner.run_validation(
        dataset_path=state.get("dataset_path", "synthetic"),
        target_column=state.get("target_column", "target"),
        random_seed=state.get("random_seed", 42),
    )

    if not result.ok or not result.metrics:
        err_msg = f"Validation failed (exit code {result.exit_code}): {result.stderr[-400:] if result.stderr else 'Metric extraction failed'}"
        events.append({
            "time": _now(),
            "type": "error",
            "stage": "validation",
            "agent": "Validation",
            "msg": err_msg,
        })
        return {
            "current_stage": "failed",
            "error_message": err_msg,
            "events": events,
        }

    metrics = result.metrics
    auroc = metrics.get("auroc", 0.0)
    ks = metrics.get("ks", 0.0)
    psi = metrics.get("psi", 0.0)
    brier = metrics.get("brier", 0.0)

    events.append({
        "time": _now(),
        "type": "ok",
        "stage": "validation",
        "agent": "Validation",
        "msg": f"Validation metrics: AUROC={auroc:.4f}, KS={ks:.4f}, Brier={brier:.4f}, PSI={psi:.4f}",
    })

    events.append({
        "time": _now(),
        "type": "info",
        "stage": "validation",
        "agent": "Model Risk Reviewer",
        "msg": "Generating model risk commentary using LLM (Azure OpenAI / local fallback)",
    })

    # Call LLM client (Azure OpenAI with structured output and graceful fallback)
    system_prompt = (
        "You are an executive model risk officer reviewing a credit risk scorecard. "
        "Review the quantitative validation metrics and return a structured assessment."
    )
    user_prompt = (
        f"Validation metrics for run {run_id}:\n"
        f"{json.dumps(metrics, indent=2)}\n"
        "Provide executive summary, list of concerns, and recommendation (PROCEED/HOLD/REJECT)."
    )

    commentary = structured_chat(
        system=system_prompt,
        user=user_prompt,
        schema=RiskCommentary,
        metrics=metrics,
    )

    events.append({
        "time": _now(),
        "type": "info" if commentary.recommendation == "PROCEED" else "warn",
        "stage": "validation",
        "agent": "Model Risk Reviewer",
        "msg": f"Risk assessment: {commentary.recommendation} · {commentary.summary[:140]}...",
    })

    artifacts = dict(state.get("artifact_paths") or {})
    artifacts.update(result.artifact_paths)

    return {
        "current_stage": "governance",
        "validation_metrics": metrics,
        "llm_risk_commentary": commentary.model_dump(),
        "artifact_paths": artifacts,
        "stages_completed": ["validation"],
        "events": events,
    }

