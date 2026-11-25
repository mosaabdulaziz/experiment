/** The Insights charts, drawn in SVG at the container's real width: single-series bar charts (counts per
 * bucket, haircuts per barber), so no legend. Bars <= 24px with a 4px rounded data-end and a 2px gap between
 * neighbours, a solid hairline grid, and a hover / focus tooltip on every bar.
 * Series hue --chart-1 (run through the palette validator). */
import { useEffect, useRef, useState } from 'react'
import type { Bucket, PerBarber, Summary } from '../api/types'
import { fmtShortDate, useLang, type Lang } from '../i18n'
import { fmtTimer } from './format'

type Point = Summary & { bucket: string }

const C_GREEN = 'var(--chart-1)'
const LOCALE: Record<Lang, string> = { en: 'en-GB', ar: 'ar-EG' }

/** A period's name in the current language: "25 Sep 18:00", "25 Sep", "Sep 2026".
 * A week is named by its first day. */
export function bucketLabel(iso: string, bucket: Bucket, lang: Lang): string {
  if (bucket === 'hour') return `${fmtShortDate(lang, iso)} ${iso.slice(11, 13)}:00`
  if (bucket === 'month') {
    return new Intl.DateTimeFormat(LOCALE[lang], { numberingSystem: 'latn', month: 'short', year: 'numeric' })
      .format(new Date(`${iso.slice(0, 7)}-15T12:00:00`))
  }
  return fmtShortDate(lang, iso)
}

/** The shorter x-axis tick: an hour bucket shows only its clock time. */
function tickLabel(iso: string, bucket: Bucket, lang: Lang): string {
  return bucket === 'hour' ? `${iso.slice(11, 13)}:00` : bucketLabel(iso, bucket, lang)
}

function useWidth() {
  const ref = useRef<HTMLDivElement | null>(null)
  const [w, setW] = useState(0)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(() => setW(el.clientWidth))
    ro.observe(el)
    setW(el.clientWidth)
    return () => ro.disconnect()
  }, [])
  return { ref, w }
}

/** Clean axis ticks (1 / 2 / 5 x 10^n steps) covering `maxVal`. */
function axis(maxVal: number, integer: boolean): { max: number; ticks: number[] } {
  const raw = Math.max(maxVal, integer ? 1 : 0.1) / 4
  const p = 10 ** Math.floor(Math.log10(raw))
  const n = raw / p
  let step = (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * p
  if (integer) step = Math.max(1, Math.round(step))
  const max = Math.ceil(maxVal / step) * step || step
  const ticks: number[] = []
  for (let t = 0; t <= max + 1e-9; t += step) ticks.push(Number(t.toFixed(6)))
  return { max, ticks }
}

/** A column with a 4px rounded top and a square base. */
function barPath(x: number, top: number, w: number, base: number): string {
  const h = base - top
  if (h <= 0.5) return ''
  const r = Math.min(4, w / 2, h)
  return `M${x},${base}V${top + r}Q${x},${top} ${x + r},${top}H${x + w - r}Q${x + w},${top} ${x + w},${top + r}V${base}Z`
}

interface Tip { x: number; y: number; title: string; rows: { color: string; label: string; value: string; dash?: boolean }[] }

function Tooltip({ tip }: { tip: Tip | null }) {
  if (!tip) return null
  return (
    <div className="tooltip" style={{ left: tip.x + 12, top: Math.max(0, tip.y - 10) }}>
      <div className="muted small">{tip.title}</div>
      {tip.rows.map(r => (
        <div key={r.label} className="tipRow">
          <span className={r.dash ? 'keyDash' : 'keyLine'} style={{ borderColor: r.color }} />
          <b>{r.value}</b> <span className="muted">{r.label}</span>
        </div>
      ))}
    </div>
  )
}

/* ── a count per period ────────────────────────────────────────────────── */

export function CountBars({ series, bucket, value, label }: {
  series: Point[]
  bucket: Bucket
  value: (p: Point) => number
  label: string
}) {
  const { fmt, lang } = useLang()
  const { ref, w } = useWidth()
  const [tip, setTip] = useState<Tip | null>(null)
  const H = 200, padL = 34, padR = 6, padT = 10, padB = 24
  const plotW = Math.max(0, w - padL - padR)
  const plotH = H - padT - padB
  const { max, ticks } = axis(Math.max(0, ...series.map(value)), true)
  const y = (v: number) => padT + plotH - (v / max) * plotH
  const base = padT + plotH
  const band = series.length ? plotW / series.length : 0
  const barW = Math.max(2, Math.min(24, band - 6))
  const every = Math.max(1, Math.ceil(series.length / Math.max(1, Math.floor(plotW / 58))))

  const show = (i: number, clientX: number, clientY: number) => {
    const p = series[i]
    const r = ref.current!.getBoundingClientRect()
    setTip({
      x: clientX - r.left, y: clientY - r.top, title: bucketLabel(p.bucket, bucket, lang),
      rows: [{ color: C_GREEN, label, value: fmt.number(value(p)) }],
    })
  }

  return (
    <div className="chartWrap" ref={ref}>
      {w > 0 && (
        <svg className="chartSvg" direction="ltr" width={w} height={H} role="img" aria-label={label}>
          {ticks.map(tk => (
            <g key={tk}>
              <line className="chartGrid" x1={padL} x2={w - padR} y1={y(tk)} y2={y(tk)} />
              <text className="chartAxis" x={padL - 6} y={y(tk) + 3} textAnchor="end">{fmt.number(tk)}</text>
            </g>
          ))}
          {series.map((p, i) => {
            const cx = padL + band * i + band / 2
            return (
              <g
                key={p.bucket}
                className="markHit"
                tabIndex={0}
                aria-label={`${bucketLabel(p.bucket, bucket, lang)}: ${label} ${fmt.number(value(p))}`}
                onPointerMove={e => show(i, e.clientX, e.clientY)}
                onPointerLeave={() => setTip(null)}
                onFocus={e => { const b = (e.currentTarget as SVGGElement).getBoundingClientRect(); show(i, b.left + b.width / 2, b.top) }}
                onBlur={() => setTip(null)}
              >
                <rect x={padL + band * i} y={padT} width={band} height={plotH + padB} fill="transparent" />
                <path className="mark" d={barPath(cx - barW / 2, y(value(p)), barW, base)} fill={C_GREEN} />
                {i % every === 0 && <text className="chartAxis" x={cx} y={H - 7} textAnchor="middle">{tickLabel(p.bucket, bucket, lang)}</text>}
              </g>
            )
          })}
          <line x1={padL} x2={w - padR} y1={base} y2={base} stroke="var(--line-strong)" />
        </svg>
      )}
      <Tooltip tip={tip} />
    </div>
  )
}

/* ── haircuts per barber ───────────────────────────────────────────────── */

export function BarberBars({ rows, names }: { rows: PerBarber[]; names: Record<string, string> }) {
  const { t, fmt } = useLang()
  const withUnattributed = rows.some(r => r.barber_id === null)
    ? rows : [...rows, { barber_id: null, name: null, haircuts: 0, average_haircut_s: null }]
  const max = Math.max(1, ...withUnattributed.map(r => r.haircuts))
  return (
    <div className="barberBars" role="list" aria-label={t('ch.perBarber')}>
      {withUnattributed.map(r => {
        const label = r.barber_id === null ? t('ch.unassigned') : (names[r.barber_id] ?? r.name ?? r.barber_id)
        return (
          <div key={r.barber_id ?? 'none'} className={r.barber_id === null ? 'bbRow bbUnattributed' : 'bbRow'} role="listitem"
               title={`${label}: ${fmt.number(r.haircuts)} ${t('ch.haircuts')}${r.average_haircut_s ? ` · ${fmtTimer(r.average_haircut_s)}` : ''}`}>
            <span className="bbName">{label}</span>
            <span className="bbTrack">
              <span className="bbBar" style={{ width: `calc((100% - 40px) * ${r.haircuts / max})` }} />
              <bdi dir="ltr" className="bbValue num">{fmt.number(r.haircuts)}</bdi>
            </span>
          </div>
        )
      })}
    </div>
  )
}
