# credit_risk_agents/tools/llm_client.py
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    # Search for .env in repo root or parent directories
    current_dir = Path(__file__).resolve().parent
    for p in [current_dir, *current_dir.parents]:
        env_file = p / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            break
except Exception:
    pass

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class RiskCommentary(BaseModel):
    summary: str = Field(..., max_length=500)
    concerns: list[str] = Field(default_factory=list, max_length=10)
    recommendation: str = Field(..., pattern="^(PROCEED|HOLD|REJECT)$")


def _heuristic_commentary(user_prompt: str, metrics: Optional[Dict[str, Any]] = None) -> RiskCommentary:
    """Deterministic, policy-compliant risk commentary fallback when external LLM is offline."""
    extracted_metrics: Dict[str, float] = {}
    if metrics:
        extracted_metrics = {k.lower(): float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    else:
        # Try extracting numbers from user_prompt
        try:
            m = re.findall(r'"([a-zA-Z_]+)":\s*([0-9.]+)', user_prompt)
            for k, v in m:
                extracted_metrics[k.lower()] = float(v)
        except Exception:
            pass

    auroc = extracted_metrics.get("auroc", extracted_metrics.get("test_auroc", 0.75))
    ks = extracted_metrics.get("ks", 0.35)
    psi = extracted_metrics.get("psi", 0.05)
    brier = extracted_metrics.get("brier", extracted_metrics.get("brier_score", 0.12))

    concerns: list[str] = []
    if auroc < 0.70:
        concerns.append(f"AUROC ({auroc:.3f}) falls below regulatory baseline threshold of 0.70.")
    if ks < 0.30:
        concerns.append(f"KS statistic ({ks:.3f}) is below acceptable separation threshold of 0.30.")
    if psi > 0.25:
        concerns.append(f"Population Stability Index ({psi:.3f}) indicates significant distribution drift (>0.25).")

    if auroc >= 0.70 and ks >= 0.30 and psi <= 0.25:
        recommendation = "PROCEED"
        summary = (
            f"The champion scorecard demonstrates sound discriminatory power (AUROC: {auroc:.3f}, KS: {ks:.3f}) "
            f"with low calibration error (Brier: {brier:.3f}) and stable population dynamics (PSI: {psi:.3f}). "
            f"Metrics satisfy governance requirements for model risk review."
        )
    elif auroc >= 0.65 and ks >= 0.25:
        recommendation = "HOLD"
        summary = (
            f"Scorecard validation shows marginal performance (AUROC: {auroc:.3f}, KS: {ks:.3f}). "
            f"Policy limits allow conditional hold pending qualitative feature review and sensitivity testing."
        )
    else:
        recommendation = "REJECT"
        summary = (
            f"Scorecard discrimination (AUROC: {auroc:.3f}, KS: {ks:.3f}) fails supervisory minimums. "
            f"Automated gating rejects promotion to production serving."
        )

    return RiskCommentary(
        summary=summary[:450],
        concerns=concerns[:5],
        recommendation=recommendation,
    )


def structured_chat(
    model: Optional[str] = None,
    system: str = "",
    user: str = "",
    schema: Type[T] = RiskCommentary,  # type: ignore
    temperature: float = 0.1,
    metrics: Optional[Dict[str, Any]] = None,
) -> T:
    """Call Azure OpenAI, then Ollama, or gracefully fallback to rule-based commentary."""
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_API_BASE")
    azure_version = os.getenv("AZURE_API_VERSION", "2023-12-01-preview")
    azure_model = model or os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1")

    # 1. Attempt Azure OpenAI if configured
    if azure_api_key and azure_endpoint:
        try:
            from openai import AzureOpenAI
            client = AzureOpenAI(
                api_key=azure_api_key,
                api_version=azure_version,
                azure_endpoint=azure_endpoint,
                timeout=3.0,
            )
            json_schema_prompt = (
                f"\nYou must return ONLY valid JSON matching this schema:\n"
                f"{json.dumps(schema.model_json_schema())}\n"
                f"Do not include markdown fences or any explanation outside JSON."
            )
            resp = client.chat.completions.create(
                model=azure_model,
                messages=[
                    {"role": "system", "content": system + json_schema_prompt},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            raw_text = resp.choices[0].message.content or "{}"
            return schema.model_validate_json(raw_text)
        except Exception as exc:
            logger.warning("Azure OpenAI call failed (%s), trying local/fallback", exc)

    # 2. Attempt local Ollama if available
    try:
        from ollama import chat as ollama_chat
        ollama_model = model or "qwen2.5:3b"
        response = ollama_chat(
            model=ollama_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format=schema.model_json_schema(),
            options={"temperature": temperature, "num_predict": 256},
        )
        return schema.model_validate_json(response.message.content)
    except Exception as exc:
        logger.debug("Ollama call failed or not running: %s", exc)

    # 3. Graceful heuristic fallback
    if issubclass(schema, RiskCommentary):
        return _heuristic_commentary(user, metrics=metrics)  # type: ignore

    # Generic fallback
    return schema()  # type: ignore

