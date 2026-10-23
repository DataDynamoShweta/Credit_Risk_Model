"""Render validation reports from the Jinja2 template.

``reports/templates/validation_report.md.j2`` is the single source of truth for the report
layout; there is no embedded fallback template. PDF export lives in ``pdf_export``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

TEMPLATE_NAME = "validation_report.md.j2"


def fmt(value, digits: int = 3) -> str:
    """Format numbers for Markdown tables; None becomes an em dash."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def pct(value, digits: int = 1) -> str:
    return "—" if value is None else f"{value * 100:.{digits}f}%"


def check_mark(value) -> str:
    return {True: "✅ pass", False: "❌ fail", None: "➖ not checked"}[value]


def default_template_path() -> Path:
    from credit_risk_model.config import get_settings
    return get_settings().reports_dir / "templates" / TEMPLATE_NAME


def render_report(context: dict, template_path: Path | None = None) -> str:
    path = Path(template_path) if template_path else default_template_path()
    if not path.exists():
        raise FileNotFoundError(f"Report template not found: {path}")
    env = Environment(loader=FileSystemLoader(str(path.parent)), undefined=StrictUndefined,
                      trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)
    env.filters.update(fmt=fmt, pct=pct, check=check_mark)
    context = {"validation_date": datetime.now(timezone.utc).date().isoformat(), **context}
    return env.get_template(path.name).render(**context)


def build_markdown(output_path, context: dict, template_path: Path | None = None) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(context, template_path), encoding="utf-8")
    return output
