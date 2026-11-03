# api/routers/workflows.py
from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from langgraph.types import Command
from pydantic import BaseModel

from credit_risk_agents.graph import graph

logger = logging.getLogger(__name__)

router = APIRouter(tags=["agentic-workflows"])

# In-memory store for run-level events in case of fast polling before state commit
_run_events: dict[str, list[dict[str, Any]]] = {}
_run_lock = threading.Lock()


# ---------------------------------------------------------------- request models
class StartWorkflowRequest(BaseModel):
    dataset: str | None = "synthetic"
    dataset_path: str | None = None
    target: str | None = "target"
    target_column: str | None = None
    champion: str | None = "logistic_regression"
    challenger: str | None = "hist_gradient_boosting"
    oot_pct: float | None = 20.0
    seed: int | None = 42
    random_seed: int | None = None
    stages: list[str | dict] | None = None
    agents: list[str] | None = None


class StartWorkflowResponse(BaseModel):
    run_id: str
    status: str = "running"
    events_url: str
    artifacts_url: str


class ApprovalRequest(BaseModel):
    decision: Literal["APPROVE", "REJECT"] = "APPROVE"
    comment: str | None = "Model approved by Model Risk Officer"
    reviewer_id: str = "mro_officer_01"


# ---------------------------------------------------------------- background executor
def _execute_graph(run_id: str, initial_state: dict) -> None:
    config = {"configurable": {"thread_id": run_id}}
    try:
        graph.invoke(initial_state, config=config)
    except Exception as exc:
        logger.exception("Error executing graph for run %s", run_id)
        with _run_lock:
            if run_id not in _run_events:
                _run_events[run_id] = []
            _run_events[run_id].append({
                "time": "NOW",
                "type": "error",
                "stage": "failed",
                "agent": "System",
                "msg": f"Graph execution error: {exc}",
            })


# ---------------------------------------------------------------- endpoints
@router.post("/workflows/credit-risk/run", response_model=StartWorkflowResponse, status_code=202)
def run_credit_risk_workflow(payload: StartWorkflowRequest) -> StartWorkflowResponse:
    run_id = f"crw-{uuid.uuid4().hex[:10]}"

    # Resolve dataset path: if lendingclub, point to loan_data_2015.csv if available
    chosen_dataset = payload.dataset_path or payload.dataset or "synthetic"
    if str(chosen_dataset).lower() in ("lendingclub", "lending club", "lending_club"):
        repo_root = Path(__file__).resolve().parents[2]
        csv_candidate = repo_root / "loan_data_2015.csv"
        if csv_candidate.exists():
            chosen_dataset = str(csv_candidate)
        else:
            chosen_dataset = "synthetic"

    chosen_target = payload.target_column or payload.target or "target"
    if "loan_status" in chosen_target:
        chosen_target = "loan_status"

    chosen_seed = payload.random_seed if payload.random_seed is not None else (payload.seed or 42)

    initial_state = {
        "run_id": run_id,
        "dataset_path": chosen_dataset,
        "target_column": chosen_target,
        "random_seed": chosen_seed,
        "champion": payload.champion or "logistic_regression",
        "challenger": payload.challenger or "hist_gradient_boosting",
        "oot_pct": payload.oot_pct or 20.0,
        "stages": [s if isinstance(s, str) else s.get("name", "") for s in (payload.stages or [])],
        "agents": payload.agents or [],
        "current_stage": "intake",
        "stages_completed": [],
        "governance_status": "PENDING",
        "artifact_paths": {},
        "events": [],
    }

    with _run_lock:
        _run_events[run_id] = []

    # Run graph asynchronously in background thread
    t = threading.Thread(target=_execute_graph, args=(run_id, initial_state), daemon=True)
    t.start()

    return StartWorkflowResponse(
        run_id=run_id,
        status="running",
        events_url=f"/runs/{run_id}/events",
        artifacts_url=f"/runs/{run_id}/artifacts",
    )


@router.get("/runs/{run_id}/events")
def get_run_events(run_id: str, since_index: int = Query(0, ge=0)) -> dict:
    config = {"configurable": {"thread_id": run_id}}
    snapshot = graph.get_state(config)
    events: list[dict[str, Any]] = []

    if snapshot and snapshot.values:
        events = list(snapshot.values.get("events") or [])

    # Include any in-memory system events
    with _run_lock:
        mem_events = _run_events.get(run_id, [])
        for e in mem_events:
            if e not in events:
                events.append(e)

    sliced = events[since_index:]
    return {
        "run_id": run_id,
        "total_events": len(events),
        "since_index": since_index,
        "events": sliced,
    }


@router.get("/runs/{run_id}")
def get_run_status(run_id: str) -> dict:
    config = {"configurable": {"thread_id": run_id}}
    snapshot = graph.get_state(config)

    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    values = snapshot.values
    completed = values.get("stages_completed", [])
    current = values.get("current_stage", "intake")

    # Determine awaiting approval
    is_awaiting_approval = snapshot.next == ("governance",)
    status = current
    if is_awaiting_approval:
        status = "awaiting_approval"

    # Progress calculation
    if status == "completed" or status == "failed":
        pct = 100
    elif is_awaiting_approval:
        pct = 75
    elif current == "validation":
        pct = 60
    elif current == "training":
        pct = 35
    elif current == "intake":
        pct = 15
    else:
        pct = min(90, len(completed) * 20)

    return {
        "run_id": run_id,
        "status": status,
        "progress_pct": pct,
        "completed_stages": completed,
        "validation_metrics": values.get("validation_metrics"),
        "llm_risk_commentary": values.get("llm_risk_commentary"),
        "artifact_paths": values.get("artifact_paths", {}),
        "governance_status": values.get("governance_status"),
        "error_message": values.get("error_message"),
        "awaiting_approval": is_awaiting_approval,
        "next": snapshot.next,
    }


@router.post("/workflows/credit-risk/{run_id}/approve")
@router.post("/workflows/credit-risk/{run_id}/approval")
def approve_run(run_id: str, payload: ApprovalRequest) -> dict:
    config = {"configurable": {"thread_id": run_id}}
    snapshot = graph.get_state(config)

    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="Run not found")

    if snapshot.next != ("governance",):
        raise HTTPException(
            status_code=409,
            detail=f"Run {run_id} is not awaiting approval (next={snapshot.next})",
        )

    resume_payload = {
        "decision": payload.decision,
        "comment": payload.comment,
        "reviewer_id": payload.reviewer_id,
    }

    def _resume():
        try:
            graph.invoke(Command(resume=resume_payload), config=config)
        except Exception:
            logger.exception("Error resuming graph for %s", run_id)

    t = threading.Thread(target=_resume, daemon=True)
    t.start()
    t.join(timeout=10.0)

    final = graph.get_state(config).values
    return {
        "run_id": run_id,
        "status": final.get("current_stage"),
        "governance_status": final.get("governance_status"),
        "approval_comment": final.get("approval_comment"),
    }


@router.get("/runs/{run_id}/artifacts")
def list_run_artifacts(run_id: str) -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    run_dir = repo_root / "artifacts" / "runs" / run_id
    if not run_dir.exists():
        # Fallback to general artifacts dir
        run_dir = repo_root / "artifacts"

    items: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("*")):
        if path.is_file():
            items.append({
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "url": f"/runs/{run_id}/artifacts/{path.name}",
            })

    return {"run_id": run_id, "artifacts": items}


@router.get("/runs/{run_id}/artifacts/{name}")
def download_run_artifact(run_id: str, name: str):
    repo_root = Path(__file__).resolve().parents[2]
    # Check run-specific artifacts
    file_path = repo_root / "artifacts" / "runs" / run_id / name
    if not file_path.exists():
        # Check global artifacts or reports
        file_path = repo_root / "artifacts" / name
        if not file_path.exists():
            file_path = repo_root / "reports" / name

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Artifact {name} not found")

    media_type = "application/json"
    if name.endswith(".md"):
        media_type = "text/markdown"
    elif name.endswith(".joblib"):
        media_type = "application/octet-stream"

    return FileResponse(file_path, media_type=media_type, filename=name)

