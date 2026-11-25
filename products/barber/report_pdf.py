"""Headless Chrome for the reports: the HTML report printed to PDF, a chart's SVG rendered to PNG (for Word). The
binary: the configured path, else google-chrome, chromium, chromium-browser on PATH. Each call uses a throwaway
profile directory and a timeout; any failure is RendererUnavailable (the endpoint answers 503)."""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

CANDIDATES = ("google-chrome", "chromium", "chromium-browser")


class RendererUnavailable(RuntimeError):
    pass


def chrome_path(configured: str | None = None) -> str | None:
    if configured:
        return configured if Path(configured).is_file() else None
    return next((p for p in map(shutil.which, CANDIDATES) if p), None)


def _run(chrome: str | None, args: list[str], page: str, out_name: str, timeout: float,
         tmp_dir: Path | None = None) -> bytes:
    exe = chrome_path(chrome)
    if exe is None:
        raise RendererUnavailable("no Chrome / Chromium found")
    if tmp_dir is not None:                             # 2026-10-25: the site's data_dir, not /tmp (root disk full)
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="barber-report-", dir=tmp_dir) as tmp:
        src, out = Path(tmp) / "page.html", Path(tmp) / out_name
        src.write_text(page, encoding="utf-8")
        root = ["--no-sandbox"] if hasattr(os, "geteuid") and os.geteuid() == 0 else []   # Chrome refuses root otherwise
        cmd = [exe, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check", *root,
               f"--user-data-dir={tmp}/profile", *[a.replace("{out}", str(out)) for a in args], src.as_uri()]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        except OSError as exc:
            raise RendererUnavailable(f"Chrome failed: {exc}") from None
        try:
            proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)       # its helper processes too
            proc.wait()
            raise RendererUnavailable(f"Chrome timed out after {timeout:.0f} s") from None
        if not out.is_file() or out.stat().st_size == 0:
            raise RendererUnavailable("Chrome wrote nothing")
        return out.read_bytes()


def html_to_pdf(html: str, chrome: str | None = None, timeout: float = 60, tmp_dir: Path | None = None) -> bytes:
    return _run(chrome, ["--no-pdf-header-footer", "--print-to-pdf={out}"], html, "report.pdf", timeout, tmp_dir)


def svg_to_png(svg: str, width: int, height: int, chrome: str | None = None, timeout: float = 30, scale: int = 2,
               tmp_dir: Path | None = None) -> bytes:
    """`scale`: device pixels per CSS pixel (2 = a sharp chart in Word at its printed size)."""
    page = f'<!doctype html><html><body style="margin:0;background:#fff">{svg}</body></html>'
    return _run(chrome, [f"--window-size={width},{height}", f"--force-device-scale-factor={scale}", "--hide-scrollbars",
                         "--screenshot={out}"], page, "chart.png", timeout, tmp_dir)
