# credit_risk_agents/graph.py
from __future__ import annotations

import datetime
from typing import Any, Dict, List, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from credit_risk_agents.state import CreditRiskState
from credit_risk_agents.nodes.intake_node import intake_node
from credit_risk_agents.nodes.training_node import training_node
from credit_risk_agents.nodes.validation_node import validation_node
from credit_risk_agents.nodes.governance_node import governance_node

AUROC_THRESHOLD = 0.70
KS_THRESHOLD = 0.30


def _now() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def route_after_intake(state: CreditRiskState) -> Literal["training", "halt"]:
    if state.get("current_stage") == "failed":
        return "halt"
    return "training"


def route_after_training(state: CreditRiskState) -> Literal["validation", "halt"]:
    if state.get("current_stage") == "failed":
        return "halt"
    return "validation"


def route_after_validation(
    state: CreditRiskState,
) -> Literal["governance", "halt"]:
    if state.get("current_stage") == "failed":
        return "halt"
    metrics = state.get("validation_metrics") or {}
    auroc = metrics.get("auroc", 0.0)
    ks = metrics.get("ks", 0.0)
    if auroc < AUROC_THRESHOLD or ks < KS_THRESHOLD:
        return "halt"
    return "governance"


def halt_node(state: CreditRiskState) -> CreditRiskState:
    metrics = state.get("validation_metrics") or {}
    auroc = metrics.get("auroc", 0.0)
    ks = metrics.get("ks", 0.0)
    err_msg = state.get("error_message") or (
        f"Validation gate failed policy threshold: "
        f"AUROC={auroc:.4f} (min {AUROC_THRESHOLD}), KS={ks:.4f} (min {KS_THRESHOLD})"
    )
    events: List[Dict[str, Any]] = [
        {
            "time": _now(),
            "type": "error",
            "stage": "validation_gate",
            "agent": "Model Risk Reviewer",
            "msg": err_msg,
        }
    ]
    return {
        "current_stage": "failed",
        "governance_status": "REJECTED",
        "error_message": err_msg,
        "stages_completed": ["validation_gate"],
        "events": events,
    }


def build_graph(checkpointer=None):
    builder = StateGraph(CreditRiskState)

    builder.add_node("intake", intake_node)
    builder.add_node("training", training_node)
    builder.add_node("validation", validation_node)
    builder.add_node("governance", governance_node)
    builder.add_node("halt", halt_node)

    builder.add_edge(START, "intake")
    builder.add_conditional_edges("intake", route_after_intake, {"training": "training", "halt": "halt"})
    builder.add_conditional_edges("training", route_after_training, {"validation": "validation", "halt": "halt"})
    builder.add_conditional_edges(
        "validation",
        route_after_validation,
        {"governance": "governance", "halt": "halt"},
    )

    builder.add_edge("governance", END)
    builder.add_edge("halt", END)

    saver = checkpointer if checkpointer is not None else MemorySaver()
    return builder.compile(checkpointer=saver)


# Global checkpointer and compiled graph instance
global_checkpointer = MemorySaver()
graph = build_graph(checkpointer=global_checkpointer)
