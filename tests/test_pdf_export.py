from pathlib import Path
from unittest.mock import MagicMock, patch

from credit_risk_model.governance import pdf_export


def test_missing_dependency(tmp_path: Path):
    md = tmp_path / "r.md"; md.write_text("# x")
    with patch.object(pdf_export, "is_available", return_value=False):
        try:
            pdf_export.export_pdf(md, tmp_path / "r.pdf")
        except RuntimeError as e:
            assert "WeasyPrint" in str(e)


def test_invokes_pipeline(tmp_path: Path):
    md = tmp_path / "r.md"
    md.write_text("# Title\n\n| a | b |\n|---|---|\n| 1 | 2 |")
    fake_html = MagicMock()
    with patch.object(pdf_export, "is_available", return_value=True), \
         patch.dict("sys.modules", {
             "weasyprint": MagicMock(HTML=fake_html),
             "markdown": MagicMock(
                 markdown=lambda s, extensions: "<p>ok</p>")}):
        pdf_export.export_pdf(md, tmp_path / "r.pdf")
        fake_html.assert_called_once()
