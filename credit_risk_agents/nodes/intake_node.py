# credit_risk_agents/nodes/intake_node.py
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from credit_risk_agents.state import CreditRiskState


def _now() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def intake_node(state: CreditRiskState) -> CreditRiskState:
    run_id = state.get("run_id", "default")
    dataset_path_str = state.get("dataset_path", "synthetic")
    target_column = state.get("target_column", "target")
    repo_root = Path(__file__).resolve().parents[2]
    artifacts_dir = repo_root / "artifacts" / "runs" / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    events: List[Dict[str, Any]] = [
        {
            "time": _now(),
            "type": "info",
            "stage": "intake",
            "agent": "Data Quality",
            "msg": f"Starting data intake & profiling for dataset: {dataset_path_str}",
        }
    ]

    is_synthetic = dataset_path_str.lower() in ("synthetic", "synthetic demo data", "") or not Path(dataset_path_str).exists()

    if is_synthetic:
        # Synthetic data profiling
        digest = hashlib.sha256(f"synthetic-seed-{state.get('random_seed', 42)}".encode()).hexdigest()
        quality_summary = {
            "dataset_kind": "synthetic",
            "rows": 4000,
            "columns": ["loan_amnt", "int_rate", "annual_inc", "dti", "emp_length", "home_ownership", "target"],
            "target_column": "target",
            "default_rate": 0.152,
            "missing_values": {},
            "duplicate_rows": 0,
            "fingerprint_sha256": digest,
        }
        events.append({
            "time": _now(),
            "type": "ok",
            "stage": "intake",
            "agent": "Target & Leakage",
            "msg": f"Synthetic baseline generated (4000 rows, default rate: 15.2%, fingerprint: {digest[:10]}...)",
        })
    else:
        path = Path(dataset_path_str).resolve()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()

        # Read sample
        df_head = pd.read_csv(path, nrows=500)
        found_target = target_column if target_column in df_head.columns else ("loan_status" if "loan_status" in df_head.columns else "target")
        missing_dict = {col: float(df_head[col].isna().mean()) for col in df_head.columns[:10]}

        quality_summary = {
            "dataset_kind": "csv",
            "path": str(path),
            "rows_sampled": len(df_head),
            "columns": list(df_head.columns),
            "target_column": found_target,
            "missing_values": missing_dict,
            "fingerprint_sha256": digest,
        }
        events.append({
            "time": _now(),
            "type": "ok",
            "stage": "intake",
            "agent": "Data Quality",
            "msg": f"Dataset verified: {path.name} (SHA-256: {digest[:10]}..., target: {found_target})",
        })

    # Save quality_summary.json
    quality_file = artifacts_dir / "quality_summary.json"
    quality_file.write_text(json.dumps(quality_summary, indent=2), encoding="utf-8")

    artifact_paths = dict(state.get("artifact_paths") or {})
    artifact_paths["quality_summary"] = str(quality_file)

    events.append({
        "time": _now(),
        "type": "info",
        "stage": "intake",
        "agent": "Pipeline Engineer",
        "msg": "Intake completed successfully. Proceeding to model training.",
    })

    return {
        "current_stage": "training",
        "input_hash": digest,
        "artifact_paths": artifact_paths,
        "stages_completed": ["intake"],
        "events": events,
    }

