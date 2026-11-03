# credit_risk_agents/tools/llm_client.py
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv

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


def _heuristic_commentary(
    user_prompt: str, metrics: dict[str, Any] | None = None
) -> RiskCommentary:
    """Deterministic, policy-compliant risk commentary fallback when external LLM is offline."""
    extracted_metrics: dict[str, float] = {}
    if metrics:
        extracted_metrics = {
            k.lower(): float(v) for k, v in metrics.items() if isinstance(v, (int, float))
        }
    else:
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
        concerns.append(
            f"KS statistic ({ks:.3f}) is below acceptable separation threshold of 0.30."
        )
    if psi > 0.25:
        concerns.append(
            f"Population Stability Index ({psi:.3f}) indicates significant distribution drift (>0.25)."
        )

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


_HF_PIPELINE: Any = None
_HF_TRIED = False


def _get_hf_pipeline() -> Any:
    """Lazily load the local Hugging Face model; returns None if unavailable."""
    global _HF_PIPELINE, _HF_TRIED
    if _HF_TRIED:
        return _HF_PIPELINE
    _HF_TRIED = True
    root = Path(__file__).resolve().parents[2]
    model_dir = Path(os.getenv("HF_MODEL_DIR") or root / "models" / "Qwen2.5-3B-Instruct")
    if not model_dir.is_dir():
        logger.debug("Local LLM directory not found: %s", model_dir)
        return None
    from transformers import pipeline

    _HF_PIPELINE = pipeline("text-generation", model=str(model_dir), device_map="auto")
    return _HF_PIPELINE


def test_llm_connection(config: dict[str, Any]) -> dict[str, Any]:
    """Test connectivity to a user-configured LLM provider with a fast ping."""
    provider = (config.get("provider") or "auto").lower()
    api_key = config.get("api_key")
    endpoint = config.get("endpoint")
    model = config.get("model")
    api_version = config.get("api_version") or "2023-12-01-preview"

    if provider == "auto":
        env_key = os.getenv("AZURE_OPENAI_API_KEY")
        env_ep = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_API_BASE")
        if env_key and env_ep:
            provider = "azure"
            api_key = env_key
            endpoint = env_ep
            model = model or os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1")
        else:
            return {
                "ok": True,
                "provider": "auto",
                "message": "Auto mode active (System heuristic / local fallback ready)",
            }

    try:
        if provider == "azure":
            if not api_key or not endpoint:
                return {
                    "ok": False,
                    "provider": "azure",
                    "error": "Azure OpenAI requires both API Key and Endpoint URL.",
                }
            from openai import AzureOpenAI

            azure_model = model or os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1")
            client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint,
                timeout=4.0,
            )
            resp = client.chat.completions.create(
                model=azure_model,
                messages=[{"role": "user", "content": "Ping"}],
                max_tokens=5,
            )
            return {
                "ok": True,
                "provider": "azure",
                "model": azure_model,
                "message": f"Successfully connected to Azure OpenAI ({azure_model})",
            }
        elif provider in ("openai", "ollama", "custom"):
            from openai import OpenAI

            base_url = endpoint or ("http://localhost:11434/v1" if provider == "ollama" else None)
            key = api_key or ("ollama" if provider == "ollama" else "dummy-key")
            chat_model = model or (
                "gpt-4o-mini"
                if provider == "openai"
                else ("llama3.2" if provider == "ollama" else "default")
            )
            client = OpenAI(api_key=key, base_url=base_url, timeout=4.0)
            resp = client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Ping"}],
                max_tokens=5,
            )
            return {
                "ok": True,
                "provider": provider,
                "model": chat_model,
                "message": f"Successfully connected to {provider.upper()} ({chat_model})",
            }
        else:
            return {"ok": False, "provider": provider, "error": f"Unsupported provider: {provider}"}
    except Exception as exc:
        return {"ok": False, "provider": provider, "error": str(exc)}


def structured_chat(
    model: str | None = None,
    system: str = "",
    user: str = "",
    schema: type[T] = RiskCommentary,  # type: ignore
    temperature: float = 0.1,
    metrics: dict[str, Any] | None = None,
    llm_config: dict[str, Any] | None = None,
) -> T:
    """Call configured LLM (Azure OpenAI, OpenAI, Ollama, custom), local model, or rule fallback."""
    json_schema_prompt = (
        "\nYou must return ONLY valid JSON matching this schema:\n"
        + json.dumps(schema.model_json_schema())
        + "\nDo not include markdown fences or any explanation outside JSON."
    )
    full_system = system + json_schema_prompt

    # 1. If runtime llm_config provided from UI
    if llm_config and llm_config.get("provider") and llm_config.get("provider") != "auto":
        prov = llm_config["provider"].lower()
        try:
            if prov == "azure":
                key = llm_config.get("api_key") or os.getenv("AZURE_OPENAI_API_KEY")
                ep = (
                    llm_config.get("endpoint")
                    or os.getenv("AZURE_OPENAI_ENDPOINT")
                    or os.getenv("AZURE_API_BASE")
                )
                ver = llm_config.get("api_version") or os.getenv(
                    "AZURE_API_VERSION", "2023-12-01-preview"
                )
                m = llm_config.get("model") or model or os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1")
                if key and ep:
                    from openai import AzureOpenAI

                    client = AzureOpenAI(
                        api_key=key, api_version=ver, azure_endpoint=ep, timeout=5.0
                    )
                    resp = client.chat.completions.create(
                        model=m,
                        messages=[
                            {"role": "system", "content": full_system},
                            {"role": "user", "content": user},
                        ],
                        temperature=temperature,
                        response_format={"type": "json_object"},
                    )
                    raw_text = resp.choices[0].message.content or "{}"
                    return schema.model_validate_json(raw_text)

            elif prov in ("openai", "ollama", "custom"):
                from openai import OpenAI

                base_url = llm_config.get("endpoint") or (
                    "http://localhost:11434/v1" if prov == "ollama" else None
                )
                key = (
                    llm_config.get("api_key")
                    or os.getenv("OPENAI_API_KEY")
                    or ("ollama" if prov == "ollama" else "dummy-key")
                )
                m = (
                    llm_config.get("model")
                    or model
                    or (
                        "gpt-4o"
                        if prov == "openai"
                        else ("llama3.2" if prov == "ollama" else "default")
                    )
                )
                client = OpenAI(api_key=key, base_url=base_url, timeout=5.0)
                resp = client.chat.completions.create(
                    model=m,
                    messages=[
                        {"role": "system", "content": full_system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                raw_text = resp.choices[0].message.content or "{}"
                return schema.model_validate_json(raw_text)
        except Exception as exc:
            logger.warning(
                "Custom LLM provider '%s' failed (%s); falling back to default.", prov, exc
            )

    # 2. Attempt Azure OpenAI if configured via environment
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_API_BASE")
    azure_version = os.getenv("AZURE_API_VERSION", "2023-12-01-preview")
    azure_model = model or os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1")

    if azure_api_key and azure_endpoint:
        try:
            from openai import AzureOpenAI

            client = AzureOpenAI(
                api_key=azure_api_key,
                api_version=azure_version,
                azure_endpoint=azure_endpoint,
                timeout=3.0,
            )
            resp = client.chat.completions.create(
                model=azure_model,
                messages=[
                    {"role": "system", "content": full_system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            raw_text = resp.choices[0].message.content or "{}"
            return schema.model_validate_json(raw_text)
        except Exception as exc:
            logger.warning("Azure OpenAI call failed (%s), trying local/fallback", exc)

    # 3. Attempt local Hugging Face model if available
    try:
        pipe = _get_hf_pipeline()
        if pipe is not None:
            out = pipe(
                [
                    {"role": "system", "content": full_system},
                    {"role": "user", "content": user},
                ],
                max_new_tokens=256,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                return_full_text=False,
            )
            text = out[0]["generated_text"]
            if not isinstance(text, str):
                text = text[-1]["content"]
            match = re.search(r"\{.*\}", text, re.DOTALL)
            return schema.model_validate_json(match.group(0) if match else text)
    except Exception as exc:
        logger.debug("Local Hugging Face call failed or model not available: %s", exc)

    # 4. Graceful heuristic fallback
    if issubclass(schema, RiskCommentary):
        return _heuristic_commentary(user, metrics=metrics)  # type: ignore

    return schema()  # type: ignore
