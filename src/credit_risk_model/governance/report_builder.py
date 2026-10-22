"""Generate Markdown validation reports, with optional PDF conversion."""
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Template

DEFAULT_TEMPLATE = """# Independent Model Validation Report

- **Model:** {{ model_name }}
- **Validation date (UTC):** {{ validation_date }}
- **Validation status:** {{ status }}
- **Model purpose:** {{ purpose }}

## Executive summary
{{ executive_summary }}

## Quantitative results
{% for section, metrics in results.items() %}
### {{ section }}
{% for key, value in metrics.items() %}
- **{{ key }}:** {{ value }}
{% endfor %}
{% endfor %}

## Findings and remediation
{% for finding in findings %}
- **{{ finding.severity }} — {{ finding.title }}:** {{ finding.description }} (Owner: {{ finding.owner }}; Due: {{ finding.due_date }})
{% else %}
- No findings were supplied. Independent reviewers must confirm this is appropriate.
{% endfor %}

## Limitations, assumptions and use restrictions
{{ limitations }}

## Governance and regulatory considerations
This report is an evidence template, not a declaration of compliance with PRA SS1/23, IFRS 9 or regulatory capital requirements. Document model scope, materiality, validation independence, data lineage, conceptual soundness, implementation testing, outcomes analysis, approval, limitations and remediation in accordance with the institution's approved policy.
"""

def build_markdown(output_path, *, model_name, purpose, results, status="Review required",
                   executive_summary="Automated results require interpretation by an independent validator.",
                   findings=None, limitations="Not approved for production use.", template_path=None):
    template_text = Path(template_path).read_text(encoding="utf-8") if template_path else DEFAULT_TEMPLATE
    content = Template(template_text).render(
        model_name=model_name, purpose=purpose, results=results, status=status,
        executive_summary=executive_summary, findings=findings or [], limitations=limitations,
        validation_date=datetime.now(timezone.utc).date().isoformat())
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    return output

def markdown_to_pdf(markdown_path, pdf_path):
    """Optional PDF conversion through WeasyPrint; raises a clear error if unavailable."""
    try:
        import markdown
        from weasyprint import HTML
    except ImportError as exc:
        raise RuntimeError("Install the 'pdf' extra for PDF report generation") from exc
    html = markdown.markdown(Path(markdown_path).read_text(encoding="utf-8"), extensions=["tables"])
    HTML(string=html).write_pdf(str(pdf_path))
    return Path(pdf_path)
