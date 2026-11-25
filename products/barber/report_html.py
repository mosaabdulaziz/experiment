"""The report as one self-contained HTML page (spec section 3): inline CSS, server-drawn SVG bar charts, snapshots
embedded as base64 JPEG, dir="rtl" for Arabic; week / month lists collapse in <details> and a print stylesheet opens
them (report_pdf prints this page). 2026-10-15 (user: a more professional layout): a header band with the IRIS mark and
the shop's logo (or its wordmark), figure cards, shaded tables, A4 pages numbered in the footer, the appendix on a new
page. The row and figure helpers are shared with report_docx."""
from __future__ import annotations

import base64
import re
from html import escape
from typing import Callable

from report_text import clock, day_name, kind_title, label, number, percent, stamp, timer

GREEN, AMBER, INK, MUTE, LINE, GROUND = "#1b7a52", "#b7791f", "#17251f", "#5c6b63", "#d8dedb", "#eef1ef"
LOW = "#fff1b8"                     # a low-occupancy haircut's row (2026-11-14)

# The IRIS app icon (frontend ui/BrandLogo.tsx, the same brand mark as the dashboard).
IRIS_SVG = ('<svg class="irisMark" viewBox="0 0 1024 1024" width="44" height="44" xmlns="http://www.w3.org/2000/svg">'
            '<rect width="1024" height="1024" rx="287.31" fill="#020302"/><g fill="#f9996c">'
            '<path d="M339.87,516.59v-159.49l-136.81,136.81c-9.99,9.99-9.99,26.19,0,36.18l82.55,82.55h150.32l-96.05-96.05Z"/>'
            '<path d="M714.26,387.16l-75.82-75.82c-4.8-4.8-11.31-7.49-18.09-7.49h-41.5l-83.31,83.31h218.72Z"/>'
            '<path d="M620.35,720.15c6.79,0,13.29-2.69,18.09-7.49l21.04-21.04v-148.92l-177.45,177.45h138.32Z"/>'
            '<path d="M310.44,637.73l74.93,74.93c4.8,4.8,11.31,7.49,18.09,7.49h43.03l82.42-82.42h-218.47Z"/>'
            '<path d="M403.71,303.85c-6.79,0-13.29,2.7-18.09,7.49l-21.04,21.04v150.83l179.36-179.36h-140.23Z"/>'
            '<path d="M820.95,493.85l-.06-.06-81.91-81.91h-149.81l95.03,95.03v160l136.69-136.69v-.13c10.05-9.98,10.08-26.23.06-36.24Z"/>'
            '<path d="M530.96,481.9c-.72-.72-1.69-1.13-2.71-1.13h-32.49c-1.02,0-2,.4-2.72,1.13l-27.38,27.39c-1.5,1.5-1.5,3.93,0,5.43l27.35,27.39c.72.72,1.69,1.12,2.71,1.12h32.53c1.02,0,1.99-.4,2.71-1.12l27.37-27.37v-.02c1.51-1.5,1.51-3.93,0-5.44h0s-27.37-27.37-27.37-27.37Z"/>'
            '</g></svg>')

CSS = f"""
@page {{ size: A4; margin: 16mm 13mm 18mm; }}
* {{ box-sizing: border-box; }}
body {{ font-family: "Noto Sans", "Noto Naskh Arabic", "DejaVu Sans", Arial, sans-serif; color: {INK}; margin: 0;
       font-size: 11.5px; line-height: 1.45; background: #fff; }}
.wrap {{ max-width: 980px; margin: 0 auto; padding: 22px; }}
.band {{ display: flex; align-items: center; gap: 16px; padding: 14px 18px; border-radius: 10px;
        background: linear-gradient(90deg, {GREEN}, #16593f); color: #fff; }}
.band .titles {{ flex: 1; min-width: 0; }}
.band h1 {{ font-size: 20px; margin: 0; letter-spacing: .2px; }}
.band .sub {{ font-size: 13px; opacity: .92; margin-top: 2px; }}
.band .period {{ font-size: 12px; opacity: .85; margin-top: 2px; }}
.shopMark {{ font-weight: 800; letter-spacing: 2px; font-size: 15px; padding: 8px 12px; border: 2px solid #fff;
            border-radius: 6px; white-space: nowrap; }}
.shopLogo {{ max-height: 48px; max-width: 160px; background: #fff; border-radius: 6px; padding: 4px; }}
.meta {{ display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; color: {MUTE}; font-size: 11px;
        margin: 8px 2px 4px; }}
h2 {{ font-size: 14px; color: {GREEN}; margin: 22px 0 8px; padding-bottom: 4px; border-bottom: 2px solid {GREEN};
     break-after: avoid; }}
h3 {{ font-size: 12px; margin: 14px 0 6px; color: {INK}; break-after: avoid; }}
h4 {{ font-size: 11.5px; margin: 10px 0 4px; color: {MUTE}; }}
.cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }}
.card {{ background: {GROUND}; border-radius: 8px; padding: 9px 11px; border-inline-start: 4px solid {GREEN};
        break-inside: avoid; }}
.card.time {{ border-inline-start-color: #3b7fd6; }}
.card.alert {{ border-inline-start-color: {AMBER}; }}
.card .v {{ display: block; font-size: 19px; font-weight: 700; margin-top: 2px; }}
.card .k {{ color: {MUTE}; font-size: 10.5px; }}
.chart {{ border: 1px solid {LINE}; border-radius: 8px; padding: 8px 10px; margin: 6px 0 12px; break-inside: avoid; }}
.chart svg {{ width: 100%; height: auto; display: block; }}
table {{ border-collapse: collapse; width: 100%; margin: 4px 0 10px; font-size: 11px; }}
thead th {{ background: {GREEN}; color: #fff; font-weight: 600; text-align: start; padding: 6px 8px; }}
tbody td {{ padding: 5px 8px; border-bottom: 1px solid {LINE}; text-align: start; }}
tbody tr:nth-child(even) td {{ background: #f6f8f7; }}
tbody tr.low td, .lowKey {{ background: {LOW}; }}
.lowKey {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; vertical-align: middle; }}
.card .n {{ display: block; font-size: 10.5px; color: {INK}; }}
td.num, th.num {{ text-align: end; font-variant-numeric: tabular-nums; }}
.snaps {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }}
.snap {{ border: 1px solid {LINE}; border-radius: 8px; overflow: hidden; break-inside: avoid; }}
.snap img {{ width: 100%; display: block; }}
.snap .cap {{ padding: 6px 8px; font-size: 10.5px; color: {MUTE}; }}
.ph {{ padding: 34px 10px; text-align: center; color: {MUTE}; background: {GROUND}; }}
.note {{ color: {MUTE}; }}
summary {{ cursor: pointer; font-weight: 600; margin: 8px 0; color: {GREEN}; }}
.foot {{ margin-top: 24px; padding-top: 8px; border-top: 1px solid {LINE}; color: {MUTE}; font-size: 10.5px;
        display: flex; justify-content: space-between; }}
.pageBreak {{ break-before: page; }}
@media print {{
  .wrap {{ max-width: none; padding: 0; }}
  details {{ display: block; }} details > summary {{ list-style: none; }}
  .foot {{ display: none; }}
  .band, thead th, .card, tbody tr:nth-child(even) td, tbody tr.low td, .lowKey, .ph {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
}}
"""


def _page_css(lang: str, generated: str) -> str:
    """The printed pages' footer: who made it and Page X of Y (Chrome's @page margin boxes)."""
    g = escape(f"{label(lang, 'generatedBy')} · {generated}").replace('"', "'")
    page, of = label(lang, "page"), label(lang, "of")
    left, right = ("bottom-right", "bottom-left") if lang == "ar" else ("bottom-left", "bottom-right")
    font = f'font-size: 9px; color: {MUTE}; font-family: "Noto Sans", "Noto Naskh Arabic", Arial, sans-serif;'
    return (f'@page {{ @{left} {{ content: "{g}"; {font} }} '
            f'@{right} {{ content: "{page} " counter(page) " {of} " counter(pages); {font} }} }}')


def wordmark(place: str) -> str:
    """BekkyBarber -> BEKKY BARBER (the shop's name when no logo file is set)."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", place).upper()


def bars_svg(items: list[tuple[str, float]], width: int, height: int, rtl: bool, empty: str) -> str:
    # viewBox (2026-10-18): the page's chart card is narrower than `width`; without it the PDF cut the chart at the
    # card's edge -- no hours under it, the evening bars gone
    font = (f'viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet" '
            'font-family="Noto Sans, DejaVu Sans, Arial, sans-serif"')
    if not items or not any(v for _, v in items):
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" {font}>'
                f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" fill="{MUTE}">{escape(empty)}</text></svg>')
    top = max(v for _, v in items) or 1
    left, right, ceil, base = 34, 12, 18, height - 24
    step = (width - left - right) / len(items)
    bw = min(60.0, max(2.0, step * 0.62))     # a lone bar stays a bar, not a block
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-size="11" {font}>']
    for k in (0.5, 1.0):                       # light gridlines with their values
        y = base - (base - ceil) * k
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="{LINE}" stroke-dasharray="3 3"/>')
        out.append(f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end" fill="{MUTE}">{number(top * k, 0 if top >= 4 else 1)}</text>')
    out.append(f'<line x1="{left}" y1="{base}" x2="{width - right}" y2="{base}" stroke="{MUTE}"/>')
    for i, (lab, v) in enumerate(items):
        k = len(items) - 1 - i if rtl else i
        x = left + k * step + (step - bw) / 2
        h = (base - ceil) * v / top
        out.append(f'<rect x="{x:.1f}" y="{base - h:.1f}" width="{bw:.1f}" height="{h:.1f}" rx="2" fill="{GREEN}"/>')
        if v:
            out.append(f'<text x="{x + bw / 2:.1f}" y="{base - h - 4:.1f}" text-anchor="middle" font-weight="600" '
                       f'fill="{INK}">{number(v)}</text>')
        out.append(f'<text x="{x + bw / 2:.1f}" y="{height - 7}" text-anchor="middle" fill="{MUTE}">{escape(lab)}</text>')
    out.append("</svg>")
    return "".join(out)


def series_items(model: dict, lang: str) -> list[tuple[str, float]]:
    day = model["kind"] == "day"
    return [(p["bucket"][11:13] if day else str(int(p["bucket"][8:10])), p["haircuts"]) for p in model["series"]]


def barber_items(model: dict, lang: str) -> list[tuple[str, float]]:
    return [(b["name"] or label(lang, "noBarber"), b["haircuts"]) for b in model["barbers"]]


def _table(head: list[str], rows: list[list[str]], num: tuple[int, ...] = (), mark: frozenset = frozenset()) -> str:
    """mark: the rows (by index) to highlight (low occupancy)."""
    cls = lambda i: ' class="num"' if i in num else ""
    th = "".join(f"<th{cls(i)}>{escape(h)}</th>" for i, h in enumerate(head))
    trs = "".join(('<tr class="low">' if k in mark else "<tr>") + "".join(f"<td{cls(i)}>{escape(str(c))}</td>" for i, c in enumerate(r)) + "</tr>" for k, r in enumerate(rows))
    return f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"


def _end(r: dict, lang: str) -> str:
    if r["end"]:
        return clock(r["end"])
    return label(lang, "interrupted") if r.get("state") == "interrupted" else label(lang, "running")


def haircut_rows(rows: list[dict], lang: str) -> list[list[str]]:
    return [[clock(r["start"]), _end(r, lang), timer(r["duration_s"]),
             r["barber"] or label(lang, "noBarber"), label(lang, "yes") if r["excessive"] else ""] for r in rows]


def low_marks(rows: list[dict]) -> frozenset:
    """The indexes of the low-occupancy haircuts among rows (2026-11-14): highlighted in every haircut list."""
    return frozenset(i for i, r in enumerate(rows) if r.get("low_occupancy"))


def low_list(model: dict) -> list[dict]:
    return model["figures"].get("low_occupancy") or []        # a report saved before 2026-11-14 has none


def served_note(model: dict, lang: str) -> str:
    """Beside Served customers: "(+1 with low occupancy: L6@19:47)" -- a day names them, a week or month counts them."""
    low = low_list(model)
    if not low:
        return ""
    names = ": " + ", ".join(f'{x["chair"]}@{clock(x["start"])}' for x in low) if model["kind"] == "day" else ""
    return f'(+{number(len(low))} {label(lang, "lowOcc")}{names})'


def alert_rows(rows: list[dict], lang: str) -> list[list[str]]:
    return [[label(lang, f"alert.{a['kind']}"), a["chair"] or "—", a["barber"] or "—", stamp(lang, a["start"]),
             clock(a["end"]) if a["end"] else label(lang, "running"), timer(a["duration_s"])] for a in rows]


def figure_cards(model: dict, lang: str) -> list[tuple[str, str, str]]:
    """(label, value, tone): count | time | alert -- the summary as cards, in the spec's order. A longest wait or
    haircut still in progress (time so far, 2026-10-15) says so. 2026-10-18 (user): no walk-outs or average days
    between visits -- the tool cannot tell them well yet."""
    f = model["figures"]
    run = lambda k: f" · {label(lang, 'running')}" if f.get(k) else ""
    served = f["served_customers"] - len(low_list(model)) if f["served_customers"] is not None else None
    return [(label(lang, "kpi.served"), number(served), "count"),          # + served_note beside it
            (label(lang, "kpi.visits"), number(f["visitors"]), "count"),
            (label(lang, "kpi.completed"), number(f["completed_haircuts"]), "count"),
            (label(lang, "kpi.avgWait"), timer(f["average_wait_s"]), "time"),
            (label(lang, "kpi.longestWait"), timer(f["longest_wait_s"]) + run("longest_wait_running"), "time"),
            (label(lang, "kpi.avgCut"), timer(f["average_haircut_s"]), "time"),
            (label(lang, "kpi.longestCut"), timer(f["longest_haircut_s"]) + run("longest_haircut_running"), "time"),
            (label(lang, "kpi.excessWait"), percent(f["excessive_wait"]), "alert"),
            (label(lang, "kpi.excessCut"), percent(f["excessive_haircut"]), "alert"),
            (label(lang, "kpi.returning"), percent(f["returning_visits"]), "count")]


def figure_items(model: dict, lang: str) -> list[tuple[str, str]]:
    return [(k, v) for k, v, _ in figure_cards(model, lang)]


def period_text(model: dict, lang: str) -> str:
    """2026-10-18 (user): the time the report covers, from the shop day's start: "Covers Fri 16 Oct 09:00 – 19:00"
    (made while the day runs: to then), "Covers Sat 17 Oct 09:00 – Sun 18 Oct 09:00" (a whole day); weeks and
    months alike. A run on recordings gives the footage's times and says so."""
    a, b = model["start"], model["until"] or model["end"]
    text = f"{label(lang, 'covers')} {stamp(lang, a)} – {clock(b) if a[:10] == b[:10] else stamp(lang, b)}"
    return f"{text} ({label(lang, 'footageTime')})" if model.get("footage") else text


def header_html(model: dict, lang: str, shop_logo: tuple[bytes, str] | None, title_key: str = "title") -> str:
    """The band (IRIS mark, title, place — branch · kind, the time covered, the shop's logo or wordmark) and the
    generated / day-start line. Shared with the haircut photo report (photo_report.py, 2026-10-22)."""
    L = lambda k: escape(label(lang, k))
    s = model["site"]
    shop = (f'<img class="shopLogo" alt="" src="data:{shop_logo[1]};base64,{base64.b64encode(shop_logo[0]).decode()}">'
            if shop_logo else f'<div class="shopMark">{escape(wordmark(s["place"]))}</div>')
    return (f'<header class="band">{IRIS_SVG}<div class="titles"><h1>{L(title_key)}</h1>'
            f'<div class="sub">{escape(s["place"])} — {escape(s["branch"])} · {escape(kind_title(lang, model["kind"]))}</div>'
            f'<div class="period">{escape(period_text(model, lang))}</div></div>{shop}</header>'
            f'<div class="meta"><span>{L("generated")}: {escape(stamp(lang, model["generated_at"]))}</span>'
            f'<span>{L("dayStarts")} {model["day_start_hour"]:02d}:00</span></div>')


def render_html(model: dict, lang: str, image: Callable[[str], bytes | None], lan: bool, printing: bool = False,
                shop_logo: tuple[bytes, str] | None = None) -> str:
    """printing: every <details> starts open (Chrome prints a closed one without its rows; final review C1) and the
    pages are numbered. shop_logo: (bytes, media type) of the shop's logo; None = its wordmark."""
    L = lambda k: escape(label(lang, k))
    rtl, day = lang == "ar", model["kind"] == "day"
    det = "<details open>" if printing else "<details>"
    s = model["site"]
    generated = stamp(lang, model["generated_at"])
    head = header_html(model, lang, shop_logo)
    note = served_note(model, lang)                     # the first card is Served customers
    cards = "".join(f'<div class="card {tone}"><span class="k">{escape(k)}</span><span class="v">{escape(v)}</span>'
                    + (f'<span class="n">{escape(note)}</span>' if note and i == 0 else "") + "</div>"
                    for i, (k, v, tone) in enumerate(figure_cards(model, lang)))
    charts = (f'<h3>{L("ch.haircuts")}</h3><div class="chart">'
              + bars_svg(series_items(model, lang), 900, 230, rtl, label(lang, "ch.noData")) + "</div>"
              + f'<h3>{L("ch.perBarber")}</h3><div class="chart">'
              + bars_svg(barber_items(model, lang), 900, 230, rtl, label(lang, "ch.noData")) + "</div>")
    barbers = (_table([label(lang, "col.barber"), label(lang, "col.haircuts"), label(lang, "col.avg")],
                      [[b["name"] or label(lang, "noBarber"), number(b["haircuts"]), timer(b["average_haircut_s"])]
                       for b in model["barbers"]], num=(1, 2)) if model["barbers"] else f'<p class="note">{L("ch.noData")}</p>')
    hc_head = [label(lang, k) for k in ("col.start", "col.end", "col.duration", "col.barber", "col.excessive")]
    chairs = _table([label(lang, k) for k in ("col.chair", "col.haircuts", "col.total", "col.avg")],
                    [[c["chair"], number(c["haircuts"]), timer(c["total_s"]), timer(c["average_s"])] for c in model["chairs"]],
                    num=(1, 2, 3))
    if low_list(model):
        chairs += f'<p class="note"><span class="lowKey"></span> {L("lowOcc.note")}</p>'
    for c in model["chairs"]:
        if day:
            chairs += f'<h3>{escape(c["chair"])}</h3>' + _table(hc_head, haircut_rows(c["rows"], lang), num=(2,),
                                                                mark=low_marks(c["rows"]))
        else:
            chairs += f'<h3>{escape(c["chair"])} — {L("perDay")}</h3>' + _table(
                [label(lang, k) for k in ("col.day", "col.haircuts", "col.total")],
                [[day_name(lang, d["day"]), number(d["haircuts"]), timer(d["total_s"])] for d in c["days"]], num=(1, 2))
    if not model["chairs"]:
        chairs = f'<p class="note">{L("ch.noData")}</p>'
    ped = model.get("pedicure") or {"count": 0, "chairs": []}
    pedicure = ""
    if ped["count"]:                                # 2026-11-10: pedicure chairs, never customers
        when = clock if day else (lambda v: stamp(lang, v))
        pedicure = (f'<h2 id="pedicure">{L("pedicure")}</h2><p>'
                    + escape(label(lang, "pedicure.line").format(n=number(ped["count"]), t=timer(ped["total_s"]))) + "</p>")
        for c in ped["chairs"]:
            pedicure += f'<h3>{escape(c["chair"])}</h3>' + _table(
                [label(lang, k) for k in ("col.start", "col.end", "col.duration")],
                [[when(r["start"]), _end(r, lang), timer(r["duration_s"])] for r in c["rows"]], num=(2,))
    al = model["alerts"]
    alerts = _table([label(lang, "col.kind"), label(lang, "col.count")],
                    [[label(lang, "alert.excessive_wait"), number(al["counts"]["excessive_wait"])],
                     [label(lang, "alert.excessive_service"), number(al["counts"]["excessive_service"])]], num=(1,))
    by_id = {a["id"]: a for a in al["rows"]}
    snaps = []
    for aid in al["shown"]:
        a = by_id[aid]
        cap = (f'{escape(label(lang, "alert." + a["kind"]))} · {escape(a["chair"] or "—")} · '
               f'{escape(stamp(lang, a["start"]))} · {timer(a["duration_s"])}')
        jpg = image(aid) if lan else None
        pic = (f'<img src="data:image/jpeg;base64,{base64.b64encode(jpg).decode()}" alt="">' if jpg
               else f'<div class="ph">{L("lanOnly") if not lan else L("noSnapshot")}</div>')
        snaps.append(f'<div class="snap">{pic}<div class="cap">{cap}</div></div>')
    if snaps:
        alerts += f'<h3>{L("snapshots") if day else L("longest")}</h3><div class="snaps">{"".join(snaps)}</div>'
    alert_head = [label(lang, k) for k in ("col.kind", "col.chair", "col.barber", "col.start", "col.end", "col.duration")]
    if day and al["rows"]:
        alerts += _table(alert_head, alert_rows(al["rows"], lang), num=(5,))
    body = (head + f'<h2 id="summary">{L("summary")}</h2><div class="cards">{cards}</div>'
            + f'<h2 id="charts">{L("charts")}</h2>{charts}' + f'<h2 id="barbers">{L("barbers")}</h2>{barbers}'
            + f'<h2 id="chairs">{L("chairs")}</h2>{chairs}' + pedicure + f'<h2 id="alerts">{L("alerts")}</h2>{alerts}')
    if not day:
        app = "".join(f'{det}<summary>{escape(c["chair"])} — {L("haircutList")} ({number(c["haircuts"])})</summary>'
                      + "".join(f'<h4>{escape(day_name(lang, d["day"]))}</h4>'
                                + (lambda rs: _table(hc_head, haircut_rows(rs, lang), num=(2,), mark=low_marks(rs)))(
                                    [r for r in c["rows"] if r["day"] == d["day"]])
                                for d in c["days"]) + "</details>" for c in model["chairs"])
        if al["rows"]:
            app += (f'{det}<summary>{L("alertList")} ({number(len(al["rows"]))})</summary>'
                    + _table(alert_head, alert_rows(al["rows"], lang), num=(5,)) + "</details>")
        body += (f'<div class="pageBreak"></div><h2 id="appendix">{L("appendix")}</h2>'
                 f'{app or "<p class=note>" + L("ch.noData") + "</p>"}')
    body += (f'<div class="foot"><span>{L("generatedBy")} · {escape(generated)}</span>'
             f'<span>{escape(s["place"])} — {escape(s["branch"])}</span></div>')
    title = f'{escape(s["place"])} {escape(s["branch"])} — {escape(kind_title(lang, model["kind"]))}'
    css = CSS + (_page_css(lang, generated) if printing else "")
    return (f'<!doctype html><html lang="{lang}" dir="{"rtl" if rtl else "ltr"}"><head><meta charset="utf-8">'
            f'<title>{title}</title><style>{css}</style></head><body><div class="wrap">{body}</div></body></html>')
