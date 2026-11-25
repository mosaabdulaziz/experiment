import shutil
import subprocess

import pytest

from report_pdf import RendererUnavailable, chrome_path, html_to_pdf, svg_to_png

HAVE = chrome_path() is not None


def test_no_chrome_is_reported_not_crashed():
    with pytest.raises(RendererUnavailable):
        html_to_pdf("<p>x</p>", chrome="/nonexistent/chrome")


@pytest.mark.skipif(not HAVE, reason="no Chrome on this machine")
def test_html_prints_to_a_pdf():
    pdf = html_to_pdf('<!doctype html><html lang="ar" dir="rtl"><body><h1>تقرير</h1></body></html>')
    assert pdf.startswith(b"%PDF")


@pytest.mark.skipif(not HAVE, reason="no Chrome on this machine")
def test_svg_renders_to_png():
    png = svg_to_png('<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50"><rect width="50" height="50"/></svg>', 100, 50)
    assert png.startswith(b"\x89PNG")


@pytest.mark.skipif(not HAVE or shutil.which("pdftotext") is None, reason="no Chrome / pdftotext")
def test_a_printed_week_report_carries_its_appendix_rows(tmp_path):
    from report_html import render_html
    from test_report_html import M
    pdf = tmp_path / "w.pdf"
    pdf.write_bytes(html_to_pdf(render_html(M, "en", lambda a: None, lan=True, printing=True)))
    text = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True).stdout
    appendix = text[text.index("Appendix"):]
    assert "12:00" in appendix and "12:40" in appendix              # a haircut row and an alert row


def test_chrome_as_root_runs_without_its_sandbox(monkeypatch):
    import report_pdf
    seen = {}

    class FakePopen:
        def __init__(self, cmd, **kw):
            seen["cmd"], seen["kw"] = cmd, kw
            self.returncode = 0
        def communicate(self, timeout=None):
            return b"", b""
    monkeypatch.setattr(report_pdf.os, "geteuid", lambda: 0)
    monkeypatch.setattr(report_pdf.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(report_pdf, "chrome_path", lambda c=None: "/usr/bin/true")
    with pytest.raises(RendererUnavailable):                       # FakePopen writes nothing
        html_to_pdf("<p>x</p>")
    assert "--no-sandbox" in seen["cmd"] and seen["kw"].get("start_new_session") is True


def test_the_temp_folder_is_the_one_given_and_is_cleaned_up(tmp_path):
    """2026-10-25: under the site's data_dir, not /tmp (the root disk is full)."""
    seen = tmp_path / "seen.txt"
    fake = tmp_path / "chrome"
    fake.write_text('#!/bin/sh\nfor a in "$@"; do case "$a" in\n'
                    f'  --user-data-dir=*) echo "${{a#--user-data-dir=}}" > {seen} ;;\n'
                    '  --print-to-pdf=*) printf "%%PDF fake" > "${a#--print-to-pdf=}" ;;\n'
                    'esac; done\n')
    fake.chmod(0o755)
    assert html_to_pdf("<p>x</p>", chrome=str(fake), tmp_dir=tmp_path / ".tmp") == b"%PDF fake"
    assert seen.read_text().strip().startswith(str(tmp_path / ".tmp" / "barber-report-"))
    assert list((tmp_path / ".tmp").iterdir()) == []
