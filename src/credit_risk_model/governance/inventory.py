"""Model inventory record and risk-tier assignment."""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

@dataclass
class ModelInventoryRecord:
    model_id: str
    name: str
    owner: str
    purpose: str
    model_type: str
    status: str = "development"
    risk_tier: str = "high"
    version: str = "0.1.0"
    limitations: str = "Not approved for production use"
    last_validation_date: str | None = None

    def to_dict(self):
        result = asdict(self)
        result["record_updated_at"] = datetime.now(timezone.utc).isoformat()
        return result

def assign_risk_tier(impact: str, complexity: str, reliance: str = "high") -> str:
    """Transparent illustrative tiering heuristic; replace with approved policy."""
    score = sum(x.lower() in {"high", "material", "critical"} for x in [impact, complexity, reliance])
    return "high" if score >= 2 else ("medium" if score == 1 else "low")
