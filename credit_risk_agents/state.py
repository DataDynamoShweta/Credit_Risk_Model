# credit_risk_agents/state.py
from __future__ import annotations

from typing import Annotated, Any, Literal

from typing_extensions import TypedDict

GovernanceStatus = Literal["PENDING", "APPROVED", "REJECTED"]
StageName = Literal["intake", "training", "validation", "governance", "completed", "failed"]


def merge_lists(a: list[Any] | None, b: list[Any] | None) -> list[Any]:
    res = list(a or [])
    if b:
        res.extend(b)
    return res


class CreditRiskState(TypedDict, total=False):
    # ---- identity / inputs -------------------------------------------------
    run_id: str
    dataset_path: str
    target_column: str
    random_seed: int
    stages: list[str]
    agents: list[str]
    champion: str
    challenger: str
    oot_pct: float

    # ---- control flow -----------------------------------------------------
    current_stage: StageName
    stages_completed: Annotated[list[str], merge_lists]

    # ---- evidence & deliverables ------------------------------------------
    input_hash: str | None  # SHA-256 of raw dataset
    validation_metrics: dict[str, float] | None  # AUROC, KS, Brier, PSI, etc.
    llm_risk_commentary: str | dict[str, Any] | None  # LLM output
    artifact_paths: dict[str, str]  # {"champion": "...", "metrics": "..."}

    # ---- live event stream ------------------------------------------------
    events: Annotated[list[dict[str, Any]], merge_lists]

    # ---- governance -------------------------------------------------------
    governance_status: GovernanceStatus  # default set in graph entry
    approval_comment: str | None
    reviewer_id: str | None

    # ---- runtime configuration --------------------------------------------
    llm_config: dict[str, Any] | None

    # ---- errors -----------------------------------------------------------
    error_message: str | None
