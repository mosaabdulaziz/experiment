"""The haircut photo report (2026-10-22, spec 2026-10-22-barber-haircut-photos-design.md): one shop day, per chair
every haircut with its start photo and end photo side by side -- the analysis report's model (report_data:
the haircut rows, the header, the footage clock on a run on recordings) with the photos of haircut_photo, the analysis
report's header and look. HTML; the PDF is this page printed (report_pdf). Downloaded from the Report panel only."""
from __future__ import annotations

import base64
from datetime import date, datetime, timedelta
from html import escape
from typing import Callable

from footage_clock import FootageClock
from report_data import build_report, file_name
from report_html import AMBER, CSS, GROUND, INK, LINE, LOW, MUTE, _end, _page_css, header_html
from report_text import clock, label, stamp, timer
from storage import haircut_photos

KINDS = ("start", "end")

PHOTO_CSS = f"""
.photoSummary {{ color: {MUTE}; margin: 10px 2px 4px; }}
.cut {{ border: 1px solid {LINE}; border-radius: 8px; padding: 8px 10px; margin: 8px 0 12px; break-inside: avoid; }}
.cut .when {{ display: flex; gap: 12px; align-items: baseline; font-weight: 600; margin-bottom: 6px; }}
.cut .when .dur {{ color: {MUTE}; font-weight: 400; }}
.cut .when .long {{ color: {AMBER}; }}
/* 2026-11-14 (user): a low-occupancy haircut -- its card highlighted, a note beside its times */
.cut.low {{ background: {LOW}; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
.cut .when .lowNote {{ color: {INK}; font-weight: 400; font-size: 10.5px; }}
.shots {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
.shot {{ margin: 0; }}
/* a fixed height, the whole photo in it: chair close-ups are portrait, and at half the page's width one card filled
   an A4 page (layout check, 2026-10-22); 64 mm leaves about three cards a page */
.shot img {{ width: 100%; height: 64mm; object-fit: contain; display: block; border-radius: 6px; background: {GROUND}; }}
.shot .ph {{ height: 64mm; display: flex; align-items: center; justify-content: center; border-radius: 6px; }}
.shot figcaption {{ font-size: 10.5px; color: {MUTE}; margin-top: 3px; }}
"""


def photo_file_name(start: datetime, end: datetime, place: str, slug: str, ext: str) -> str:
    return file_name("day", start, end, place, slug, ext).replace("_analysis_report.", "_haircut_photos.")


def build_photo_report(conn, first: date, now: datetime, policy: dict, site: dict, chair_names: dict[str, str],
                       live: list[dict] | None = None, clock: FootageClock | None = None) -> dict:
    model = build_report(conn, "day", first, now, policy, site, chair_names, live=live, clock=clock)
    sh = clock.shift if clock else timedelta(0)
    shown = lambda v: (datetime.fromisoformat(v) + sh).isoformat(timespec="seconds")
    rows = [r for c in model["chairs"] for r in c["rows"]]
    prows = [r for c in model.get("pedicure", {}).get("chairs", []) for r in c["rows"]]   # 2026-11-10: pedicure chairs
    got = haircut_photos(conn, [r["id"] for r in rows + prows])
    for r in rows + prows:
        mine = got.get(r["id"], {})
        r["photos"] = {k: {"path": mine[k]["path"], "taken_at": shown(mine[k]["taken_at"])} if k in mine else None
                       for k in KINDS}
        r.setdefault("excessive", False)
    model["photos"] = {"haircuts": len(rows), "both": sum(all(r["photos"].values()) for r in rows)}
    return model


def _shot(p: dict | None, kind: str, lang: str, running: bool, image: Callable[[str], bytes | None]) -> str:
    jpg = image(p["path"]) if p else None
    cap = label(lang, f"photos.{kind}") + (f" {clock(p['taken_at'])}" if p else "")
    if p and kind == "end" and running:
        cap += f" · {label(lang, 'photos.soFar')}"
    pic = (f'<img src="data:image/jpeg;base64,{base64.b64encode(jpg).decode()}" alt="">' if jpg
           else f'<div class="ph">{escape(label(lang, "photos.none"))}</div>')
    return f'<figure class="shot">{pic}<figcaption>{escape(cap)}</figcaption></figure>'


def _card(r: dict, lang: str, image: Callable[[str], bytes | None]) -> str:
    running = r["state"] == "running"
    long = f'<span class="long">{escape(label(lang, "col.excessive"))}</span>' if r["excessive"] else ""
    low = r.get("low_occupancy", False)             # 2026-11-14 (user): highlighted, with a note beside its times
    note = f'<span class="lowNote">{escape(label(lang, "lowOcc.tag"))}</span>' if low else ""
    return (f'<div class="cut{" low" if low else ""}"><div class="when"><span>{clock(r["start"])} – {escape(_end(r, lang))}</span>'
            f'{note}<span class="dur">{timer(r["duration_s"])}</span>{long}</div><div class="shots">'
            + "".join(_shot(r["photos"][k], k, lang, running, image) for k in KINDS) + "</div></div>")


def render_photo_html(model: dict, lang: str, image: Callable[[str], bytes | None], printing: bool = False,
                      shop_logo: tuple[bytes, str] | None = None) -> str:
    """image(rel_path): a photo's bytes, None when its file is gone (a "No picture" box)."""
    L = lambda k: escape(label(lang, k))
    s, n = model["site"], model["photos"]
    generated = stamp(lang, model["generated_at"])
    body = header_html(model, lang, shop_logo, title_key="photos.title")
    body += f'<p class="photoSummary">{escape(label(lang, "photos.summary").format(n=n["haircuts"], m=n["both"]))}</p>'
    for c in model["chairs"]:
        body += f'<h2>{escape(c["chair"])}</h2>' + "".join(_card(r, lang, image) for r in c["rows"])
    if not model["chairs"]:
        body += f'<p class="note">{L("ch.noData")}</p>'
    ped = model.get("pedicure") or {"chairs": []}
    if any(c["rows"] for c in ped["chairs"]):       # 2026-11-10: pedicure chairs, their own cards
        body += f'<h2>{L("pedicure")}</h2>'
        for c in ped["chairs"]:
            body += f'<h3>{escape(c["chair"])}</h3>' + "".join(_card(r, lang, image) for r in c["rows"])
    body += (f'<div class="foot"><span>{L("generatedBy")} · {escape(generated)}</span>'
             f'<span>{escape(s["place"])} — {escape(s["branch"])}</span></div>')
    css = CSS + PHOTO_CSS + (_page_css(lang, generated) if printing else "")
    title = f'{escape(s["place"])} {escape(s["branch"])} — {L("photos.title")}'
    return (f'<!doctype html><html lang="{lang}" dir="{"rtl" if lang == "ar" else "ltr"}"><head><meta charset="utf-8">'
            f'<title>{title}</title><style>{css}</style></head><body><div class="wrap">{body}</div></body></html>')
