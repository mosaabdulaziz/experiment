"""Saved reports (2026-10-25, user; spec docs/superpowers/specs/2026-10-25-barber-saved-reports-design.md): each
finished shop day's analysis and haircut photo reports, each finished week's (`week_start`, Saturday at Palm Hills)
and month's analysis report, and on a run on recordings each lap's day, saved under <data_dir>/reports/ when the
period ends. Nothing is sent anywhere, and nothing here deletes a saved report.

Layout: reports/day/<YYYY-MM-DD>/, reports/day/<footage day>/lap-<lap end YYYY-MM-DD_HHMM>/, reports/week/<first
day>/, reports/month/<YYYY-MM>/, each with the downloads' file names (<name>_<lang>.<ext>) and a report.json manifest
written last: a folder with a manifest is complete. The HTML (and Word) is built at once on the caller's thread --
the lap-end hook runs on the consumer's, before the lap's rows are cleared -- and the PDFs are printed later on the
saver's own thread, from print pages kept in the folder's .print/ (so a restart finishes them)."""
from __future__ import annotations

import json
import logging
import shutil
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

from report_data import period_bounds
from report_text import LANGS
from shopday import shop_day

log = logging.getLogger("barber.saved_reports")

FORMATS = ("pdf", "html", "docx")
PHOTO_FORMATS = ("pdf", "html")                 # the photo report has no Word form
WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")      # date.weekday() order
CATCH_UP_DAYS = 7                               # a service down at a day's end saves it when back, this far back
MANIFEST = "report.json"
PRINT_DIR = ".print"
PENDING = "pending.json"


@dataclass(frozen=True)
class SavedReports:
    langs: tuple[str, ...]
    formats: tuple[str, ...]
    week_start: int = 5                         # date.weekday(): 5 = Saturday


def parse_saved_reports(raw: dict | None) -> SavedReports | None:
    """A site yaml's `saved_reports: {langs: [en, ar], formats: [pdf, html], week_start: sat}`; none = off."""
    if not raw:
        return None
    langs = tuple(str(x) for x in raw.get("langs") or ("en",))
    formats = tuple(str(x) for x in raw.get("formats") or ("pdf",))
    week_start = str(raw.get("week_start") or "sat").lower()
    if not set(langs) <= set(LANGS):
        raise ValueError(f"saved_reports.langs must be among {LANGS}, not {list(langs)}")
    if not set(formats) <= set(FORMATS):
        raise ValueError(f"saved_reports.formats must be among {FORMATS}, not {list(formats)}")
    if week_start not in WEEKDAYS:
        raise ValueError(f"saved_reports.week_start must be one of {WEEKDAYS}, not {week_start!r}")
    return SavedReports(langs, formats, WEEKDAYS.index(week_start))


@dataclass(frozen=True)
class Period:
    kind: str                                   # day | week | month
    first: date                                 # its first shop day
    last: date                                  # its last shop day

    @property
    def folder(self) -> str:
        return f"month/{self.first:%Y-%m}" if self.kind == "month" else f"{self.kind}/{self.first.isoformat()}"


def day_of(d: date) -> Period:
    return Period("day", d, d)


def week_of(d: date, week_start: int) -> Period:
    first = d - timedelta(days=(d.weekday() - week_start) % 7)
    return Period("week", first, first + timedelta(days=6))


def month_of(d: date) -> Period:
    first = d.replace(day=1)
    return Period("month", first, (first + timedelta(days=32)).replace(day=1) - timedelta(days=1))


def due_periods(now: datetime, hour: int, week_start: int) -> list[Period]:
    """The finished periods a live site keeps saved at `now`: the last CATCH_UP_DAYS shop days, the latest finished
    week and the latest finished month. A shop day is finished once the next one has started (day_start_hour)."""
    today = shop_day(now, hour)
    days = [day_of(today - timedelta(days=n)) for n in range(1, CATCH_UP_DAYS + 1)]
    return days + [week_of(week_of(today, week_start).first - timedelta(days=1), week_start),
                   month_of(today.replace(day=1) - timedelta(days=1))]


def out_name(name: str, lang: str) -> str:
    stem, dot, ext = name.rpartition(".")
    return f"{stem}_{lang}.{ext}" if dot else f"{name}_{lang}"


def lap_folder(day: date, ended: datetime) -> str:
    return f"day/{day.isoformat()}/lap-{ended:%Y-%m-%d_%H%M}"


@dataclass(frozen=True)
class Pages:
    """One report in one language, built: the HTML to save, the HTML to print to PDF, and Word's bytes if asked."""
    screen: str
    printing: str
    docx: bytes | None = None


@dataclass(frozen=True)
class Hooks:
    """What the server gives the saver -- the download endpoints' own code (server.py)."""
    build: Callable[[Period, str, str, bool], Pages]      # (period, "analysis" | "photos", lang, with Word)
    name: Callable[[Period, str, str], str]               # (period, report, format) -> the download's name
    has_visits: Callable[[datetime, datetime], bool]
    photos: Callable[[], bool]                            # the site takes haircut photos (hp_photos)
    print_pdf: Callable[[str], bytes]
    day_start_hour: Callable[[], int]


def write_atomic(path: Path, data: bytes) -> None:
    """Under a temporary name, then renamed: never a half file."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_bytes(data)
        tmp.replace(path)
    except BaseException:                       # 2026-10-25 final review: no stray .tmp after a failed write
        tmp.unlink(missing_ok=True)
        raise


def write_manifest(folder: Path, data: dict) -> None:
    write_atomic(folder / MANIFEST, json.dumps(data, indent=1, ensure_ascii=False).encode())


def list_saved(root: Path, kind: str) -> list[dict]:
    """The complete folders of a kind (those with a manifest), newest first; a manifest that does not parse is skipped."""
    items = []
    for m in Path(root).glob(f"{kind}/**/{MANIFEST}"):
        try:
            data = json.loads(m.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("not a JSON object")
            files = data.get("files")           # 2026-10-25 final review: the page reads these; a bad one must not blank it
            if not isinstance(data.get("first"), str) or not isinstance(files, list) or not all(
                    isinstance(f, dict) and all(isinstance(f.get(k), str) for k in ("name", "report", "lang", "format"))
                    for f in files):
                raise ValueError("no usable first/files")
        except (OSError, ValueError) as exc:
            log.warning("saved report %s not listed: %s", m, exc)
            continue
        items.append({**data, "id": m.parent.relative_to(root).as_posix()})
    return sorted(items, key=lambda i: (str(i.get("first") or ""), str(i.get("lap_ended") or "")), reverse=True)


def saved_file(root: Path, folder_id: str, name: str) -> Path | None:
    """A saved file by its folder id and name, only when that folder's manifest lists it and it is there."""
    if Path(name).name != name or folder_id.split("/")[0] not in ("day", "week", "month"):
        return None                             # 2026-10-25 final review: no separators, no .., only the layout's kinds
    try:
        base = Path(root).resolve()
        folder = (base / folder_id).resolve()
    except (ValueError, OSError):               # a NUL byte
        return None
    if folder == base or not folder.is_relative_to(base):
        return None
    try:
        listed = {f["name"] for f in json.loads((folder / MANIFEST).read_text(encoding="utf-8"))["files"]}
    except (OSError, ValueError, KeyError, TypeError):
        return None
    path = folder / name
    return path if name in listed and path.is_file() else None


class ReportSaver:
    """live: a site on cameras -- each pass stages the due periods (due_periods) that have visits; a run on
    recordings stages nothing by the clock, only its laps (save_lap). Every pass then finishes the staged folders."""

    def __init__(self, cfg: SavedReports | None, root: Path, hooks: Hooks, live: bool,
                 clock: Callable[[], datetime] = datetime.now, every_s: float = 60.0):
        self.cfg, self.root, self.hooks, self.live = cfg, Path(root), hooks, live
        self.clock, self.every_s = clock, every_s
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    def _reports(self, period: Period) -> list[tuple[str, tuple[str, ...]]]:
        out = [("analysis", self.cfg.formats)]
        photo_formats = tuple(f for f in self.cfg.formats if f in PHOTO_FORMATS)
        if period.kind == "day" and photo_formats and self.hooks.photos():
            out.append(("photos", photo_formats))
        return out

    def stage(self, period: Period, folder: Path, lap_ended: datetime | None = None, partial: bool = False) -> bool:
        """Build the period's HTML (and Word) into `folder` now, its print pages into .print/, and last its
        manifest-to-be (.print/pending.json). Files already there are kept. False: a build failed (retried).
        partial: a build that still fails is left out of the manifest-to-be (the rest is saved) and logged."""
        spool = folder / PRINT_DIR
        files, failed = [], []
        for report, formats in self._reports(period):
            for lang in self.cfg.langs:
                names = {f: out_name(self.hooks.name(period, report, f), lang) for f in formats}
                entries = [{"report": report, "lang": lang, "format": f, "name": n} for f, n in names.items()]
                todo = [f for f, n in names.items()
                        if not (folder / n).exists() and not (f == "pdf" and (spool / f"{n}.html").exists())]
                if todo:
                    try:                        # 2026-10-25 final review: a write error counts the pair as failed too
                        pages = self.hooks.build(period, report, lang, "docx" in todo)
                        if "docx" in todo and pages.docx is None:
                            raise RuntimeError("no Word file built")
                        spool.mkdir(parents=True, exist_ok=True)
                        for f in todo:
                            if f == "pdf":
                                write_atomic(spool / f"{names[f]}.html", pages.printing.encode())
                            else:
                                write_atomic(folder / names[f], pages.screen.encode() if f == "html" else pages.docx)
                    except Exception:
                        log.exception("saved report %s: its %s %s report failed to build or write", folder, report, lang)
                        failed.append(f"{report} {lang}")
                        continue
                files += entries
        if failed and not partial:
            return False
        spool.mkdir(parents=True, exist_ok=True)
        start, end = period_bounds(period.kind, period.first, int(self.hooks.day_start_hour()))
        write_atomic(spool / PENDING, json.dumps({
            "kind": period.kind, "first": period.first.isoformat(), "last": period.last.isoformat(),
            "start": start.isoformat(timespec="seconds"), "end": end.isoformat(timespec="seconds"),
            "lap_ended": lap_ended.isoformat(timespec="seconds") if lap_ended else None, "label": None,
            "files": files}, ensure_ascii=False).encode())
        if failed:
            log.error("saved reports %s saved without: %s", folder, ", ".join(failed))
            return False
        return True

    def finish(self, folder: Path) -> bool:
        """Print the folder's PDFs from .print/, then write its manifest and remove .print/. False: a PDF failed
        (its print page is kept for the next pass)."""
        spool = folder / PRINT_DIR
        pending = json.loads((spool / PENDING).read_text(encoding="utf-8"))
        ok = True
        for page in sorted(spool.glob("*.html")):
            name = page.name[: -len(".html")]
            try:
                write_atomic(folder / name, self.hooks.print_pdf(page.read_text(encoding="utf-8")))
            except Exception:
                log.exception("saved report %s failed to print (retried in a minute)", folder / name)
                ok = False
                continue
            page.unlink()
        if not ok:
            return False
        write_manifest(folder, {**pending, "saved_at": self.clock().isoformat(timespec="seconds")})
        shutil.rmtree(spool, ignore_errors=True)
        log.info("saved reports complete: %s", folder)
        return True

    def pending(self) -> list[Path]:
        return sorted(p.parent.parent for p in self.root.glob(f"**/{PRINT_DIR}/{PENDING}"))

    def run_once(self) -> list[Path]:
        """One pass. Returns the folders it completed."""
        if self.cfg is None:
            return []
        if self.live:
            hour = int(self.hooks.day_start_hour())
            for p in due_periods(self.clock(), hour, self.cfg.week_start):
                folder = self.root / p.folder
                if (folder / MANIFEST).exists() or (folder / PRINT_DIR / PENDING).exists():
                    continue
                try:
                    if self.hooks.has_visits(*period_bounds(p.kind, p.first, hour)):
                        self.stage(p, folder)
                except Exception:
                    log.exception("saved reports: %s failed (retried in a minute)", p.folder)
        done = []
        for f in self.pending():
            try:
                if self.finish(f):
                    done.append(f)
            except Exception:
                log.exception("saved reports: %s failed (retried in a minute)", f)
        return done

    def save_lap(self, day: date, ended: datetime) -> Path | None:
        """A run on recordings' lap ended -- called on the consumer's thread while the lap's rows are still there:
        its day's HTML now, its PDFs on the saver's thread. A build that fails is retried once at once, while the
        rows are still there; what still fails is left out and the rest is saved. A folder name already taken (two
        laps in one minute) gets -2, -3, ..."""
        if self.cfg is None:
            return None
        base = self.root / lap_folder(day, ended)
        folder, n = base, 2
        while folder.exists():
            folder, n = base.with_name(f"{base.name}-{n}"), n + 1
        try:
            staged = self.stage(day_of(day), folder, lap_ended=ended)
        except Exception:
            log.exception("saved reports %s: staging failed (the partial stage is tried)", folder)
            staged = False
        if not staged:
            try:
                self.stage(day_of(day), folder, lap_ended=ended, partial=True)
            except Exception:
                log.error("saved reports %s: the lap's reports are lost (could not be staged)", folder, exc_info=True)
        self._wake.set()
        return folder

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception:                           # never stop the thread
                log.exception("saved reports pass failed")
            self._wake.wait(self.every_s)
            self._wake.clear()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="saved-reports", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
