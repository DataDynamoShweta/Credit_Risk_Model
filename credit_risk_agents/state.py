# credit_risk_agents/state.py
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, Union
from typing_extensions import TypedDict

GovernanceStatus = Literal["PENDING", "APPROVED", "REJECTED"]
StageName = Literal[
    "intake", "training", "validation", "governance", "completed", "failed"
]


def merge_lists(a: Optional[List[Any]], b: Optional[List[Any]]) -> List[Any]:
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
    stages: List[str]
    agents: List[str]
    champion: str
    challenger: str
    oot_pct: float

    # ---- control flow -----------------------------------------------------
    current_stage: StageName
    stages_completed: Annotated[List[str], merge_lists]

    # ---- evidence & deliverables ------------------------------------------
    input_hash: Optional[str]                       # SHA-256 of raw dataset
    validation_metrics: Optional[Dict[str, float]]   # AUROC, KS, Brier, PSI, etc.
    llm_risk_commentary: Optional[Union[str, Dict[str, Any]]]  # LLM output
    artifact_paths: Dict[str, str]                  # {"champion": "...", "metrics": "..."}

    # ---- live event stream ------------------------------------------------
    events: Annotated[List[Dict[str, Any]], merge_lists]

    # ---- governance -------------------------------------------------------
    governance_status: GovernanceStatus             # default set in graph entry
    approval_comment: Optional[str]
    reviewer_id: Optional[str]

    # ---- errors -----------------------------------------------------------
    error_message: Optional[str]

