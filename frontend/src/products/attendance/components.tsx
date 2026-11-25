import { useEffect, useMemo, useState } from 'react'
import * as XLSX from 'xlsx'
import {
  api, type AttendancePolicy, type AttendanceRecord, type AttendanceStatus, type CameraInfo, type RecordingInfo, type Zone,
} from './api'
import { CameraView } from './MonitoringPage'
import { navigate, productHref } from '../../shared/router'

/* ── shared formatting ─────────────────────────────────────────────── */

export function fmtTime(iso: string | null): string {
  if (!iso) return '—'
  return new Date(iso).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

function fmtClock(d: Date): string {
  return d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

export const STATUS_LABEL: Record<AttendanceStatus, string> = {
  present: 'حاضر',
  absent: 'غائب',
  leave: 'في إجازة',
}

/* ── KPI cards ─────────────────────────────────────────────────────── */

/** Line icons for the KPI cards (prompt.md image-8, 2026-09-14), drawn in the
 * card's accent colour inside its tinted square. 24x24, stroke only. */
const KPI_ICON: Record<string, string> = {
  // people (total)
  kpiTotal: 'M9 11a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7zM2.5 20v-1c0-2.8 2.9-4.5 6.5-4.5s6.5 1.7 6.5 4.5v1M16 4.3a3.5 3.5 0 0 1 0 6.4M18.5 14.8c1.9.6 3 1.9 3 3.7V20',
  // clipboard with check (present)
  kpiPresent: 'M9 4h6v3H9zM8 5.5H6.5A1.5 1.5 0 0 0 5 7v12.5A1.5 1.5 0 0 0 6.5 21h11a1.5 1.5 0 0 0 1.5-1.5V7a1.5 1.5 0 0 0-1.5-1.5H16M8.5 13.5l2.3 2.3 4.7-4.8',
  // calendar with a cross (absent)
  kpiAbsent: 'M4 7a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2zM4 10h16M8 3v4M16 3v4M10 13.5l4 4M14 13.5l-4 4',
  // suitcase (on leave)
  kpiLeave: 'M3.5 9a2 2 0 0 1 2-2h13a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2zM9 7V5.5A1.5 1.5 0 0 1 10.5 4h3A1.5 1.5 0 0 1 15 5.5V7M3.5 12.5h17',
}

export function KpiCards({ records }: { records: AttendanceRecord[] }) {
  const total = records.length
  const present = records.filter(r => r.status === 'present').length
  const leave = records.filter(r => r.status === 'leave').length
  const absent = records.filter(r => r.status === 'absent').length
  const cards = [
    { label: 'إجمالي الموظفين', value: total, tag: 'المسجلون بالنظام', cls: 'kpiTotal' },
    { label: 'حاضر اليوم', value: present, tag: 'منضبطون بالحضور ↑', cls: 'kpiPresent' },
    { label: 'غائب', value: absent, tag: 'بدون عذر معتمد ↓', cls: 'kpiAbsent' },
    { label: 'في إجازة', value: leave, tag: 'إجازات مسجلة', cls: 'kpiLeave' },
  ]
  return (
    <div className="kpiRow">
      {cards.map(c => (
        <div key={c.label} className={`kpiCard ${c.cls}`}>
          <div className="kpiHead">
            <span className="kpiLabel">{c.label}</span>
            <span className="kpiIcon kpiIconSvg" aria-hidden="true">
              <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor"
                   strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
                <path d={KPI_ICON[c.cls]} />
              </svg>
            </span>
          </div>
          <div className="kpiValue">{c.value}</div>
          <span className="kpiTag">{c.tag}</span>
        </div>
      ))}
    </div>
  )
}

/** Top-left export controls (prompt.md point 6) -- lives in the page header,
 * not the table toolbar, and exports the day's full dataset regardless of
 * whatever filter is currently applied in the table below (that filter is
 * local state owned by AttendanceTable; decoupling here keeps this a plain
 * (records, day) -> file function instead of threading filter state up). */
/** File icon with a short label (XLS / PDF) for the export buttons
 * (prompt.md point 5, 2026-09-14). */
function FileIcon({ label }: { label: string }) {
  return (
    <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
      <path d="M6 2.5h8.5L19 7v13.5a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1v-17a1 1 0 0 1 1-1z" fill="currentColor" />
      <path d="M14.5 2.5V7H19" fill="none" stroke="#fff" strokeWidth="1.2" strokeLinejoin="round" opacity="0.8" />
      <text x="12" y="17.2" textAnchor="middle" fontSize="5.6" fontWeight="800" fill="#fff"
            fontFamily="Segoe UI, Arial, sans-serif">{label}</text>
    </svg>
  )
}

/** Today's departure is shown only from `departure_reveal_time` (default
 * 19:00) on; before that the table shows the last sighting in "آخر ظهور" and
 * leaves "وقت الانصراف" empty (prompt.md point 1, 2026-09-14). Past days
 * always show it. `now` is passed in so the table can re-render on a timer. */
export function revealedDeparture(r: AttendanceRecord, policy: AttendancePolicy | null, now: Date): string | null {
  const t = new Date()
  const today = `${t.getFullYear()}-${String(t.getMonth() + 1).padStart(2, '0')}-${String(t.getDate()).padStart(2, '0')}`
  if (r.day < today) return r.last_departure
  if (r.day > today) return null
  const revealAt = hhmmMinutes(policy?.departure_reveal_time, 19 * 60)
  return now.getHours() * 60 + now.getMinutes() >= revealAt ? r.last_departure : null
}

export function ExportButtons({ records, day, policy = null }: {
  records: AttendanceRecord[]
  day: string
  policy?: AttendancePolicy | null
}) {
  const exportXlsx = () => {
    const now = new Date()
    const rows = [
      ['الرقم الوظيفي', 'اسم الموظف', 'القسم', 'وقت الحضور', 'آخر ظهور', 'وقت الانصراف', 'الحالة'],
      ...records.map(r => [
        r.code ?? '',
        r.display_name_ar ?? r.name,
        r.department ?? '',
        fmtTime(r.first_arrival),
        fmtTime(r.last_departure),
        fmtTime(revealedDeparture(r, policy, now)),
        STATUS_LABEL[r.status],
      ]),
    ]
    const ws = XLSX.utils.aoa_to_sheet(rows)
    const wb = XLSX.utils.book_new()
    wb.Workbook = { Views: [{ RTL: true }] }
    XLSX.utils.book_append_sheet(wb, ws, 'الحضور')
    XLSX.writeFile(wb, `attendance-${day}.xlsx`)
  }
  return (
    <div className="exportBtns noPrint">
      <button className="btnXls" onClick={exportXlsx} title="تصدير Excel" aria-label="تصدير Excel">
        <FileIcon label="XLS" />
      </button>
      <button className="btnPdf" onClick={() => window.print()} title="تصدير PDF (طباعة)" aria-label="تصدير PDF">
        <FileIcon label="PDF" />
      </button>
    </div>
  )
}

/* ── dashboard diagrams (prompt.md image-7, 2026-09-13) ─────────────── */
// Replaces the earlier "نظرة سريعة" donut. Hand-drawn SVG like the donut was
// -- no chart library in this app. Everything is derived from the day's
// attendance records plus the attendance policy (work hours, late grace).

/** "…THH:MM…" (server-local ISO) -> minutes after midnight, read straight
 * off the string rather than via Date, so a viewer in another time zone
 * sees the same hours the table shows. */
function isoMinutes(iso: string | null): number | null {
  if (!iso || iso.length < 16) return null
  const h = Number(iso.slice(11, 13))
  const m = Number(iso.slice(14, 16))
  return Number.isFinite(h) && Number.isFinite(m) ? h * 60 + m : null
}

function hhmmMinutes(hhmm: string | undefined, fallback: number): number {
  const [h, m] = (hhmm ?? '').split(':').map(Number)
  return Number.isFinite(h) && Number.isFinite(m) ? h * 60 + m : fallback
}

type ArrivalKind = 'early' | 'onTime' | 'late' | 'absent'
const ARRIVAL_KINDS: ArrivalKind[] = ['early', 'onTime', 'late', 'absent']

/** How one arrival time compares with the work day ("early" added
 * 2026-09-13): early = before work_start, on time = from work_start up to
 * work_start + late_grace_minutes, late = after that, absent = no arrival.
 * Same rule as the history page (sessions.py derive_day). */
function arrivalKind(arrived: number | null, policy: AttendancePolicy | null): ArrivalKind {
  if (arrived === null) return 'absent'
  const start = hhmmMinutes(policy?.work_start, 9 * 60)
  if (arrived < start) return 'early'
  return arrived <= start + (policy?.late_grace_minutes ?? 0) ? 'onTime' : 'late'
}

/** Everyone counted in the diagrams (not on leave -- that has its own
 * summary card) and how they arrived. */
function classifyArrivals(records: AttendanceRecord[], policy: AttendancePolicy | null) {
  return records
    .filter(r => r.status !== 'leave')
    .map(r => {
      const arrived = isoMinutes(r.first_arrival)
      return { arrived, kind: arrivalKind(arrived, policy) }
    })
}

/** Hourly bars "إجمالي عدد الموظفين": for each hour of the work day, how many
 * had arrived on time / arrived late / not arrived yet by the END of that
 * hour. Cumulative arrivals, not "inside right now" -- the daily data only
 * has each person's first arrival and last sighting. On today's page, hours
 * that haven't started yet are left empty. */
const HOUR_W = 62
const BAR_W = 11
const BAR_GAP = 3
const PLOT_TOP = 10
const PLOT_H = 160
const LABEL_H = 22
const AXIS_W = 30

export function HourlyAttendanceChart({ records, policy, isToday }: {
  records: AttendanceRecord[]
  policy: AttendancePolicy | null
  isToday: boolean
}) {
  const people = classifyArrivals(records, policy)
  const start = hhmmMinutes(policy?.work_start, 8 * 60)
  const end = hhmmMinutes(policy?.work_end, 17 * 60)
  const firstHour = Math.floor(start / 60)
  const lastHour = Math.max(firstHour, Math.floor(end / 60))
  const now = new Date()
  const nowMin = isToday ? now.getHours() * 60 + now.getMinutes() : null

  const groups = []
  for (let h = firstHour; h <= lastHour; h++) {
    const future = nowMin !== null && h * 60 > nowMin
    const counts: Record<ArrivalKind, number> = { early: 0, onTime: 0, late: 0, absent: 0 }
    if (!future) {
      // Arrived by the END of hour h: counted under how they arrived
      // (early / on time / late); everyone else hasn't arrived yet.
      for (const p of people) {
        if (p.arrived !== null && p.arrived < (h + 1) * 60) counts[p.kind]++
      }
      counts.absent = people.length - counts.early - counts.onTime - counts.late
    }
    groups.push({ h, future, counts })
  }

  const yMax = Math.max(1, people.length)
  const width = AXIS_W + groups.length * HOUR_W
  const height = PLOT_TOP + PLOT_H + LABEL_H
  const y = (v: number) => PLOT_TOP + PLOT_H - (v / yMax) * PLOT_H
  const ticks = [...new Set([0, Math.round(yMax / 2), yMax])]

  return (
    <section className="dashCard">
      <div className="dashCardHead">
        <h2>إجمالي عدد الموظفين</h2>
        <ArrivalLegend />
      </div>
      <svg className="hourSvg" viewBox={`0 0 ${width} ${height}`} style={{ direction: 'ltr' }} role="img"
           aria-label="عدد الحاضرين والمتأخرين والغائبين لكل ساعة">
        {ticks.map(t => (
          <g key={t}>
            <line className="hourGrid" x1={0} x2={width - AXIS_W} y1={y(t)} y2={y(t)} />
            <text className="hourAxis" x={width - AXIS_W / 2} y={y(t) + 3} textAnchor="middle">{t}</text>
          </g>
        ))}
        {groups.map((g, i) => {
          // RTL: the first hour sits next to the axis on the right.
          const cx = width - AXIS_W - (i + 0.5) * HOUR_W
          return (
            <g key={g.h} className={g.future ? 'hourFuture' : undefined}>
              {!g.future && ARRIVAL_KINDS.map((k, j) => {
                const bx = cx + (1.5 - j) * (BAR_W + BAR_GAP) - BAR_W / 2  // 4 bars, centred
                const v = g.counts[k]
                return (
                  <rect key={k} className={`bar-${k}`} x={bx} y={y(v)} width={BAR_W}
                        height={Math.max(0, PLOT_H + PLOT_TOP - y(v))} rx={3}>
                    <title>{`${ARRIVAL_LABEL[k]}: ${v}`}</title>
                  </rect>
                )
              })}
              <text className="hourAxis" x={cx} y={height - 6} textAnchor="middle">
                {`${String(g.h).padStart(2, '0')}:00`}
              </text>
            </g>
          )
        })}
      </svg>
    </section>
  )
}

const ARRIVAL_LABEL: Record<ArrivalKind, string> = {
  // onTime was 'الحاضرين' -- read as "present", which late people also are
  // (prompt.md point 9, 2026-09-14)
  early: 'المبكرين', onTime: 'في الموعد', late: 'المتأخرين', absent: 'الغائبون',
}

function ArrivalLegend() {
  return (
    <ul className="dashLegend">
      {ARRIVAL_KINDS.map(k => (
        <li key={k}><span className={`dot-${k}`} aria-hidden="true" />{ARRIVAL_LABEL[k]}</li>
      ))}
    </ul>
  )
}

/** "حالة الحضور حسب القسم": per department, the share present today (on
 * leave excluded) and the average hours from arrival to last sighting of
 * those present. Dot: green at 100%, red at 0%, dark in between. Employees
 * without a department are grouped as "بدون قسم" (set it on the Employees
 * page). */
const NO_DEPT = 'بدون قسم'

export function DepartmentStatus({ records }: { records: AttendanceRecord[] }) {
  const byDept = new Map<string, AttendanceRecord[]>()
  for (const r of records) {
    const key = r.department?.trim() || NO_DEPT
    byDept.set(key, [...(byDept.get(key) ?? []), r])
  }
  const rows = [...byDept.entries()]
    .sort(([a], [b]) => (a === NO_DEPT ? 1 : b === NO_DEPT ? -1 : a.localeCompare(b, 'ar')))
    .map(([name, rs]) => {
      const counted = rs.filter(r => r.status !== 'leave')
      const present = counted.filter(r => r.status === 'present')
      const pct = counted.length ? Math.round((present.length / counted.length) * 100) : 0
      const spans = present
        .map(r => {
          const a = isoMinutes(r.first_arrival)
          const d = isoMinutes(r.last_departure)
          return a !== null && d !== null && d >= a ? (d - a) / 60 : null
        })
        .filter((v): v is number => v !== null)
      const hours = spans.length ? `${(spans.reduce((s, v) => s + v, 0) / spans.length).toFixed(1)}h` : '—'
      return { name, pct, hours, cls: pct === 100 ? 'deptFull' : pct === 0 ? 'deptNone' : '' }
    })

  return (
    <section className="dashCard">
      <div className="dashCardHead"><h2>حالة الحضور حسب القسم</h2></div>
      {rows.length === 0 ? (
        <p className="camEmpty">لا توجد بيانات لهذا اليوم</p>
      ) : (
        <ul className="deptList">
          {rows.map(r => (
            <li key={r.name} className="deptRow">
              <div className="deptMain">
                <span className="deptName">{r.name}</span>
                <div className="deptBarRow">
                  <div className="deptBar"><span style={{ width: `${r.pct}%` }} /></div>
                  <span className="deptPct">{r.pct}%</span>
                </div>
              </div>
              <span className={`deptHours ${r.cls}`} title="متوسط ساعات التواجد">{r.hours}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}

/** Bubble chart "حالة الحضور": one circle per status (on time / late /
 * absent), AREA proportional to its count, overlapping like the mock. */
// onTime was 'حاضر' (prompt.md point 9, 2026-09-14): it counts only people who
// arrived within the grace period, so "حاضر 0" beside "متأخر 7" looked wrong.
const BUBBLE_LABEL: Record<ArrivalKind, string> = { early: 'مبكر', onTime: 'في الموعد', late: 'متأخر', absent: 'غائب' }
const BUBBLE_R_MAX = 88  // 64 -> 88 (2026-09-14): bigger circles; viewBox and centre scaled to match

export function AttendanceBubbles({ records, policy }: {
  records: AttendanceRecord[]
  policy: AttendancePolicy | null
}) {
  const people = classifyArrivals(records, policy)
  const total = people.length
  const counts: Record<ArrivalKind, number> = { early: 0, onTime: 0, late: 0, absent: 0 }
  for (const p of people) counts[p.kind]++
  const maxCount = Math.max(1, ...ARRIVAL_KINDS.map(k => counts[k]))
  const bySize = ARRIVAL_KINDS.filter(k => counts[k] > 0).sort((a, b) => counts[b] - counts[a])
  const radius = (k: ArrivalKind) => Math.max(22, BUBBLE_R_MAX * Math.sqrt(counts[k] / maxCount))
  // Biggest on the left-middle, second overlapping it to the upper right,
  // third tucked underneath between them -- the mock's arrangement; a
  // fourth (early, added 2026-09-13) goes to the lower left.
  const r1 = bySize[0] ? radius(bySize[0]) : 0
  const CX = 132, CY = 128
  const spots = [
    { x: CX, y: CY },
    { x: CX + r1 * 0.62 + (bySize[1] ? radius(bySize[1]) : 0) * 0.5, y: CY - r1 * 0.3 },
    { x: CX + r1 * 0.5, y: CY + r1 * 0.68 },
    { x: CX - r1 * 0.55, y: CY + r1 * 0.62 },
  ]
  const pct = (k: ArrivalKind) => (total ? Math.round((counts[k] / total) * 10000) / 100 : 0)

  return (
    <section className="dashCard">
      <div className="dashCardHead"><h2>حالة الحضور</h2></div>
      {total === 0 ? (
        <p className="camEmpty">لا توجد بيانات لهذا اليوم</p>
      ) : (
        <>
          <svg className="bubbleSvg" viewBox="0 0 320 264" style={{ direction: 'ltr' }} role="img"
               aria-label="نسب الحاضرين والمتأخرين والغائبين">
            {bySize.map((k, i) => (
              <g key={k} className="bubble">
                <circle className={`bubble-${k}`} cx={spots[i].x} cy={spots[i].y} r={radius(k)}>
                  <title>{`${BUBBLE_LABEL[k]}: ${counts[k]} (${pct(k)}%)`}</title>
                </circle>
                <text className={k === 'absent' ? 'bubbleText bubbleTextDark' : 'bubbleText'}
                      x={spots[i].x} y={spots[i].y + 4} textAnchor="middle">{`${pct(k)}%`}</text>
              </g>
            ))}
          </svg>
          <ul className="chartLegend">
            {ARRIVAL_KINDS.map(k => (
              <li key={k}>
                <span className={`legendDot dot-${k}`} aria-hidden="true" />
                <span className="legendLabel">{BUBBLE_LABEL[k]}</span>
                <span className="legendValue">{counts[k]}</span>
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  )
}

/* ── camera panel ──────────────────────────────────────────────────── */

/** Whether a camera is ACTUALLY producing footage right now, per /api/cameras
 * (parsed live from sidecar.yaml -- see stream_manager.list_streams). NOTE:
 * CameraInfo.running is NOT this signal -- it's true for every configured
 * source the instant its pipeline is up, even an RTSP source nobody is
 * publishing to (verified live 2026-09-06: every sidecar.yaml source
 * reported running=true, most stuck at offset=0 forever). `offset`
 * (perception_source.py: "timestamp of the most recent SAMPLED frame; 0.0
 * if none") only moves once a real frame has been decoded, so it's the one
 * to use anywhere "is this camera live" actually matters. */
export function camIsLive(cam: CameraInfo): boolean {
  return (cam.offset ?? 0) > 0
}

export const ATTENDANCE_CAM_IDS = ['cam-3', 'cam-9', 'cam-10', 'cam-11', 'cam-12', 'cam-13', 'cam-14', 'cam-15', 'cam-16', 'cam-17', 'cam-18']

export const CAM_LABELS: Record<string, string> = {
  'cam-3': 'كاميرا 3 — قاعة العمل',
  'cam-9': 'كاميرا 9 — المدخل الرئيسي',
  'cam-10': 'كاميرا 10 — بهو الموظفين',
  'cam-11': 'كاميرا 11 — الممر الشرقي',
  'cam-12': 'كاميرا 12 — الممر الغربي',
  'cam-13': 'كاميرا 13 — قاعة الاجتماعات',
  'cam-14': 'كاميرا 14 — مخرج الطوارئ',
  'cam-15': 'كاميرا 15 — مكتب التصميم',
  'cam-16': 'كاميرا 16 — صالة المكاتب',
  'cam-17': 'كاميرا 17 — الممر الداخلي',
  'cam-18': 'كاميرا 18 — المكاتب المفتوحة',
}

/** Frame polling: re-request the next snapshot only after the previous one
 * loaded (mirrors iris-frontend's StreamTile cadence logic) so a slow link
 * never piles up requests. */
export function LiveFrame({ camId, className }: { camId: string; className?: string }) {
  const [tick, setTick] = useState(0)
  const [failed, setFailed] = useState(false)
  return (
    <>
      <img
        className={className ?? 'camFrame'}
        src={`/stream/${camId}/frame?t=${tick}`}
        alt={CAM_LABELS[camId] ?? camId}
        onLoad={() => {
          setFailed(false)
          // 300 -> 60 ms (2026-09-13): 300 capped every tile at ~3 fps even
          // though the pipeline runs at ~12; 60 lets tiles follow it.
          setTimeout(() => setTick(t => t + 1), 60)
        }}
        onError={() => {
          setFailed(true)
          setTimeout(() => setTick(t => t + 1), 1500)
        }}
        style={failed ? { opacity: 0.3 } : undefined}
      />
      {failed && (
        <span className="camReconnecting">
          جارٍ إعادة الاتصال…
        </span>
      )}
    </>
  )
}

/** The camera's clock overlay. 2026-09-12 (issue 1, Phase 1): this used to
 * simulate "position inside a looping recording" (elapsed-time-since-stream-
 * start, modulo the file's duration, offset from a shared fake CHIRLA date)
 * -- correct only for a looping test file, meaningless for a real live
 * camera with no loop or duration at all. Now just the real current time,
 * ticking every second, still gated on the camera actually streaming
 * (`stream_started_at` present) so it disappears the same way it always did
 * when a tile goes offline.
 * Phase 2 (2026-09-12): when the camera sends its own clock over RTCP, the
 * backend reports how far it is from the server's (`clock_offset_ms`) and
 * the tile shows the camera's time; null offset = server time. See
 * ADJUSTMENTS_2026-09-12.md issue 1.
 *
 * Corrected to the SERVER's clock (2026-09-12): the table's times are
 * written by the server, but Date.now() is the viewer's own machine clock --
 * a different computer. Any drift between the two showed up as a steady
 * offset between this tile and the table (user-reported ~30s). One shared
 * fetch of the server's time per page load measures that offset; until it
 * lands (or if it fails) the clock falls back to plain browser time. */
let serverOffsetMs = 0
let serverOffsetSync: Promise<void> | null = null
function syncServerClock(): void {
  if (serverOffsetSync) return
  const t0 = Date.now()
  serverOffsetSync = api.serverTime()
    .then(({ epoch_ms }) => {
      // Midpoint of the round trip: assumes request and response legs took
      // about equally long, which is plenty at a 1-second display resolution.
      serverOffsetMs = epoch_ms - (t0 + Date.now()) / 2
    })
    .catch(() => { serverOffsetSync = null }) // retry on the next mount
}

export function useWallClock(recording: RecordingInfo | undefined): Date | null {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    syncServerClock()
    const id = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(id)
  }, [])
  if (!recording?.stream_started_at) return null
  return new Date(now + serverOffsetMs + (recording.clock_offset_ms ?? 0))
}

function CameraTile({ cam, recording, onExpand }: {
  cam: CameraInfo
  recording: RecordingInfo | undefined
  onExpand: () => void
}) {
  const wallClock = useWallClock(recording)
  return (
    <div
      className="camTile"
      role="button"
      tabIndex={0}
      title="اضغط للتكبير"
      onClick={onExpand}
      onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') onExpand() }}
    >
      <LiveFrame camId={cam.id} />
      <span className="camLive">بث مباشر</span>
      <span className="camName">{CAM_LABELS[cam.id] ?? cam.id}</span>
      {wallClock && <span className="camClock" title={recording?.clock_source === 'camera' ? 'ساعة الكاميرا (RTCP)' : 'ساعة الخادم'}>{fmtClock(wallClock)}</span>}
    </div>
  )
}

/** View-only camera info modal -- the Home dashboard's "click a camera, see
 * its info" (prompt.md point 2). Adjusting zones/restriction/place-type is
 * the Cameras page's job now (point 4), not duplicated here. */
function CameraModal({ cam, recording, zones, onClose }: {
  cam: CameraInfo
  recording: RecordingInfo | undefined
  zones: Zone[]
  onClose: () => void
}) {
  const wallClock = useWallClock(recording)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <div className="camModalBackdrop" onClick={onClose}>
      <div className="camModal" onClick={e => e.stopPropagation()}>
        <div className="camModalHead">
          <div className="camModalTitle">
            <span className="camLiveInline">بث مباشر</span>
            <strong>{CAM_LABELS[cam.id] ?? cam.id}</strong>
            <span className="zoneHint">{zones.length} مناطق</span>
            {wallClock && <span className="camModalClock" title={recording?.clock_source === 'camera' ? 'ساعة الكاميرا (RTCP)' : 'ساعة الخادم'}>{fmtClock(wallClock)}</span>}
          </div>
          <div className="camModalActions">
            <a className="btnGhost btnSmall btnOnDark" href={productHref('/cameras')}>تعديل المناطق ←</a>
            <button className="camModalClose" onClick={onClose} aria-label="إغلاق">✕</button>
          </div>
        </div>
        <CameraView camId={cam.id} zones={zones} scene={[]} recording={recording} large />
      </div>
    </div>
  )
}

/** Home dashboard's compact camera panel: first 3 live cameras + a link to
 * the full Cameras page (prompt.md point 2 -- "Show all" navigates, it
 * doesn't expand in place; that in-place toggle is now the Cameras page). */
export function CameraPanel({ recordings, zones }: {
  recordings: RecordingInfo[]
  zones: Record<string, Zone[]>
}) {
  const [cameras, setCameras] = useState<CameraInfo[]>([])
  const [expandedId, setExpandedId] = useState<string | null>(null)
  useEffect(() => {
    let cancelled = false
    const tick = () => {
      api.listCameras().then(c => {
        if (!cancelled) setCameras(c)
      }).catch(() => {})
    }
    tick()
    const id = setInterval(tick, 2000)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [])

  const recordingByCam = Object.fromEntries(recordings.map(r => [r.cam_id, r]))
  // Attendance-role cameras that are ACTUALLY streaming right now (see
  // camIsLive's doc for why that's `offset`, not `running`). This used to be
  // a hardcoded ATTENDANCE_CAM_IDS whitelist ANDed with the attendance-
  // service's own CHIRLA-specific recording registry, so a camera the
  // sidecar wasn't processing (or one CHIRLA doesn't know about, e.g.
  // cam-1/2/4/5/6) either never showed up or sat as a permanent placeholder.
  // Reading roles/offset straight off sidecar.yaml's own live status means
  // this panel tracks whatever the sidecar is actually configured AND
  // streaming right now, with no hand-maintained list to fall out of sync.
  const live = cameras.filter(c => c.roles.includes('attendance') && camIsLive(c))
  const visible = live.slice(0, 3)

  return (
    <aside className="camPanel">
      <div className="camPanelHead">
        <span className="camPanelDot" aria-hidden="true" />
        <div>
          <h2>كاميرات المراقبة</h2>
          <p>متابعة حية للحضور عبر كاميرات التعرف على الوجوه</p>
        </div>
      </div>
      {live.length === 0 ? (
        <p className="camEmpty">لا توجد كاميرات تبث حاليًا…</p>
      ) : (
        visible.map(cam => (
          <CameraTile
            key={cam.id}
            cam={cam}
            recording={recordingByCam[cam.id]}
            onExpand={() => setExpandedId(cam.id)}
          />
        ))
      )}
      <a className="camMore" href={productHref('/cameras')}>عرض جميع الكاميرات ←</a>
      {expandedId && (() => {
        // Look the camera up fresh each render: if its stream stops while
        // expanded, the modal simply closes instead of ticking a dead clock.
        const cam = live.find(c => c.id === expandedId)
        return cam ? (
          <CameraModal
            cam={cam} recording={recordingByCam[cam.id]} zones={zones[cam.id] ?? []}
            onClose={() => setExpandedId(null)}
          />
        ) : null
      })()}
    </aside>
  )
}

/* ── attendance table ──────────────────────────────────────────────── */

const PAGE_SIZE = 10

export function AttendanceTable({
  records,
  search,
  policy = null,
}: {
  records: AttendanceRecord[]
  search: string
  policy?: AttendancePolicy | null  // work hours, for the "مبكر" (early) tag
}) {
  const [dept, setDept] = useState('')
  const [status, setStatus] = useState('')
  // Table's own search field (2026-09-14), applied together with the header search.
  const [tableQuery, setTableQuery] = useState('')
  const [page, setPage] = useState(1)
  // Re-render every 30 s so today's departure column appears at the reveal
  // time even if no new sighting arrives then.
  const [now, setNow] = useState(() => new Date())
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 30_000)
    return () => clearInterval(id)
  }, [])

  const departments = useMemo(
    () => [...new Set(records.map(r => r.department).filter((d): d is string => !!d))].sort(),
    [records],
  )

  const filtered = useMemo(() => {
    const queries = [search, tableQuery].map(s => s.trim().toLowerCase()).filter(Boolean)
    return records.filter(r => {
      if (dept && r.department !== dept) return false
      if (status && r.status !== status) return false
      if (queries.length) {
        const hay = `${r.code ?? ''} ${r.name} ${r.display_name_ar ?? ''} ${r.department ?? ''}`.toLowerCase()
        if (!queries.every(q => hay.includes(q))) return false
      }
      return true
    })
  }, [records, dept, status, search, tableQuery])

  const pages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE))
  const safePage = Math.min(page, pages)
  const pageRows = filtered.slice((safePage - 1) * PAGE_SIZE, safePage * PAGE_SIZE)

  // Leave toggle moved to the Employees page (prompt.md point 4, 2026-09-14).

  return (
    <section className="tableCard printArea dashTableCard">
      <div className="tableHead">
        <h2>الموظفين</h2>
        <span className="tableCount">{filtered.length} موظف</span>
      </div>
      <div className="tableToolbar">
        <div className="toolbarFilters">
          <span className="filterTag">⏳ فلترة</span>
          <input
            className="tableSearch" type="search" value={tableQuery}
            placeholder="بحث بالاسم أو الرقم أو القسم…"
            onChange={e => { setTableQuery(e.target.value); setPage(1) }}
          />
          <select value={status} onChange={e => { setStatus(e.target.value); setPage(1) }}>
            <option value="">حالة الدوام</option>
            <option value="present">حاضر</option>
            <option value="absent">غائب</option>
            <option value="leave">في إجازة</option>
          </select>
          <select value={dept} onChange={e => { setDept(e.target.value); setPage(1) }}>
            <option value="">القسم</option>
            {departments.map(d => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
        </div>
        <button
          className="btnGhost btnSmall"
          onClick={() => {
            setDept('')
            setStatus('')
            setTableQuery('')
            setPage(1)
          }}
        >
          مسح الفلاتر
        </button>
      </div>

      <div className="dashTableScroll">
      <table className="attTable">
        <thead>
          <tr>
            <th>الرقم الوظيفي</th>
            <th>اسم الموظف</th>
            <th>القسم</th>
            <th>وقت الحضور</th>
            <th title="آخر مرة رأته فيها الكاميرا">آخر ظهور</th>
            <th>وقت الانصراف</th>
            <th>الحالة</th>
          </tr>
        </thead>
        <tbody>
          {pageRows.map(r => (
            <tr
              key={r.employee_id}
              className="rowClickable"
              onClick={() => { navigate(`/employee/${r.employee_id}`) }}
              title="عرض السجل الشهري"
            >
              <td className="cellCode">{r.code ?? '—'}</td>
              <td className="cellName">
                <img className="avatar" src={api.employeePhotoUrl(r.employee_id)} alt="" />
                <span>{r.display_name_ar ?? r.name}</span>
              </td>
              <td>{r.department ?? '—'}</td>
              <td className="cellTime">{fmtTime(r.first_arrival)}</td>
              <td className="cellTime">{fmtTime(r.last_departure)}</td>
              <td className="cellTime">{fmtTime(revealedDeparture(r, policy, now))}</td>
              <td>
                <span className={`chip chip-${r.status}`}>
                  <span className="chipDot" aria-hidden="true" />
                  {STATUS_LABEL[r.status]}
                </span>
                {r.status === 'present' && arrivalKind(isoMinutes(r.first_arrival), policy) === 'early' && (
                  <span className="earlyTag" title="وصل قبل بداية الدوام">مبكر</span>
                )}
              </td>
            </tr>
          ))}
          {pageRows.length === 0 && (
            <tr>
              <td colSpan={7} className="tableEmpty">لا توجد نتائج مطابقة</td>
            </tr>
          )}
        </tbody>
      </table>
      </div>

      <div className="pagination noPrint">
        {Array.from({ length: pages }, (_, i) => i + 1).map(p => (
          <button
            key={p}
            className={p === safePage ? 'pageBtn pageActive' : 'pageBtn'}
            onClick={() => setPage(p)}
          >
            {String(p).padStart(2, '0')}
          </button>
        ))}
      </div>
    </section>
  )
}
