"""Feature contract enforced at the model-serving boundary.

Missing features are never silently imputed and unexpected features are rejected by
default. Error messages name features only; they never echo submitted values.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import pandas as pd

NUMERIC, CATEGORICAL = "numeric", "categorical"


class FeatureContractError(ValueError):
    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("; ".join(issues))


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: str  # "numeric" or "categorical"

    def to_dict(self) -> dict:
        return asdict(self)


def infer_specs(frame: pd.DataFrame) -> list[FeatureSpec]:
    specs = []
    for column in frame.columns:
        is_numeric = (pd.api.types.is_numeric_dtype(frame[column])
                      or pd.api.types.is_bool_dtype(frame[column]))
        specs.append(FeatureSpec(column, NUMERIC if is_numeric else CATEGORICAL))
    return specs


def _short(name: object) -> str:
    text = str(name)
    return text if len(text) <= 64 else text[:61] + "..."


def validate_record(features: dict, specs: list[FeatureSpec], *,
                    allow_extra: bool = False) -> None:
    """Raise FeatureContractError if a single scoring record breaks the contract."""
    issues: list[str] = []
    if not features:
        raise FeatureContractError(["no features supplied"])
    expected = {s.name for s in specs}
    missing = [s.name for s in specs if s.name not in features]
    if missing:
        issues.append("missing required features: " + ", ".join(missing))
    if not allow_extra:
        unexpected = sorted(_short(k) for k in features if k not in expected)
        if unexpected:
            issues.append("unexpected features: " + ", ".join(unexpected[:20]))
    for spec in specs:
        if spec.name not in features:
            continue
        value = features[spec.name]
        if spec.kind == NUMERIC:
            ok = (isinstance(value, (int, float)) and not isinstance(value, bool)
                  and math.isfinite(value))
            if not ok:
                issues.append(f"feature '{spec.name}' must be a finite number")
        elif not isinstance(value, str) or not value.strip():
            issues.append(f"feature '{spec.name}' must be a non-empty string")
    if issues:
        raise FeatureContractError(issues)


def validate_frame(frame: pd.DataFrame, specs: list[FeatureSpec]) -> pd.DataFrame:
    """Return the contract columns of ``frame`` in order; raise if any are missing.

    Extra columns (for example a target or date column) are dropped, because batch
    scoring frames legitimately carry them.
    """
    missing = [s.name for s in specs if s.name not in frame.columns]
    if missing:
        raise FeatureContractError(["missing required features: " + ", ".join(missing)])
    if frame.empty:
        raise FeatureContractError(["no rows supplied"])
    return frame[[s.name for s in specs]]
