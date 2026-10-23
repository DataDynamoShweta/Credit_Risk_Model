"""Optional Markdown -> PDF export via WeasyPrint."""
from __future__ import annotations

from pathlib import Path


def is_available() -> bool:
    try:
        import weasyprint  # noqa: F401
        return True
    except ImportError:
        return False


def export_pdf(md_path: Path, pdf_path: Path) -> Path:
    if not is_available():
        raise RuntimeError(
            "PDF export requires WeasyPrint. "
            "Install with `pip install -e '.[pdf]'`."
        )
    if not md_path.exists():
        raise FileNotFoundError(md_path)

    import markdown as md_lib
    from weasyprint import HTML

    html = md_lib.markdown(
        md_path.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code"],
    )
    style = ("body{font-family:Helvetica,Arial,sans-serif;"
             "max-width:900px;margin:2rem auto;line-height:1.5;}"
             "table{border-collapse:collapse;}"
             "th,td{border:1px solid #ccc;padding:6px 10px;}")
    doc = (f"<!doctype html><html><head><meta charset='utf-8'>"
           f"<style>{style}</style></head><body>{html}</body></html>")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=doc).write_pdf(str(pdf_path))
    return pdf_path
