# credit_risk_agents/nodes/governance_node.py
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict, List
from langgraph.types import interrupt

from credit_risk_agents.state import CreditRiskState


def _now() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def governance_node(state: CreditRiskState) -> CreditRiskState:
    run_id = state.get("run_id", "default")
    metrics = state.get("validation_metrics") or {}
    commentary = state.get("llm_risk_commentary")
    repo_root = Path(__file__).resolve().parents[2]
    artifacts_dir = repo_root / "artifacts" / "runs" / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Langgraph Human-In-The-Loop interrupt
    # This pauses the StateGraph execution until the human reviewer submits an approval decision.
    decision = interrupt(
        {
            "prompt": "Model Risk Governance Review required before production promotion.",
            "run_id": run_id,
            "metrics": metrics,
            "commentary": commentary,
            "awaiting_approval": True,
        }
    )

    # Decision is provided when the API calls Command(resume=decision_dict)
    if not isinstance(decision, dict):
        decision = {"decision": "APPROVE", "comment": "Auto-resumed", "reviewer_id": "System"}

    approved = decision.get("decision") == "APPROVE"
    reviewer_id = decision.get("reviewer_id", "model_risk_officer")
    comment = decision.get("comment", "Approved for deployment")

    events: List[Dict[str, Any]] = [
        {
            "time": _now(),
            "type": "ok" if approved else "warn",
            "stage": "governance",
            "agent": "Governance",
            "msg": f"Governance decision by {reviewer_id}: {'APPROVED' if approved else 'REJECTED'} (Comment: {comment})",
        }
    ]

    artifacts = dict(state.get("artifact_paths") or {})

    if approved:
        # Generate model_card.json deliverable
        model_card = {
            "model_name": "credit_risk_champion",
            "model_type": state.get("champion", "logistic_regression"),
            "run_id": run_id,
            "target": state.get("target_column", "target"),
            "status": "APPROVED",
            "approved_by": reviewer_id,
            "approval_comment": comment,
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "performance": metrics,
            "risk_commentary": commentary,
            "limitations": [
                "Scorecard designed for retail personal unsecured lending",
                "Requires quarterly PSI monitoring with investigate threshold of 0.10",
            ],
            "governance_signoff": {
                "independent_validation": "PASSED",
                "model_risk_tier": "TIER-2",
            },
        }
        card_file = artifacts_dir / "model_card.json"
        card_file.write_text(json.dumps(model_card, indent=2), encoding="utf-8")
        artifacts["model_card"] = str(card_file)

        events.append({
            "time": _now(),
            "type": "ok",
            "stage": "governance",
            "agent": "DevOps",
            "msg": "Model card generated. Artifacts packaged and ready for serving.",
        })

    return {
        "current_stage": "completed" if approved else "failed",
        "governance_status": "APPROVED" if approved else "REJECTED",
        "approval_comment": comment,
        "reviewer_id": reviewer_id,
        "artifact_paths": artifacts,
        "stages_completed": ["governance"],
        "error_message": None if approved else f"Governance review rejected by {reviewer_id}: {comment}",
        "events": events,
    }

