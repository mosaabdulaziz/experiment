import { useMemo, useState } from 'react'
import { alertClipUrl, type CameraInfo, type IrisEvent } from './api'
import { CAM_LABELS } from './components'
import { IncidentDetailModal } from './IncidentDetailModal'
import {
  KIND_LABEL, camLabel, effectiveClips, fireKind, fmtDateTime, fmtDuration, onsetOf, type FireKind,
} from './useFireAlerts'
import { productHref } from '../../shared/router'

type TimeRange = '1h' | '24h' | '7d' | 'all'
const RANGE_MS: Record<TimeRange, number | null> = {
  '1h': 3600_000, '24h': 24 * 3600_000, '7d': 7 * 24 * 3600_000, all: null,
}

/** History / incident log (3c): read-only investigation view over the same
 * live-fetched window the live wall uses (see useFireAlerts) -- no separate
 * server-side paginated query. At this app's incident volume (a handful of
 * fire-tagged test streams) filtering the already-fetched 500-event window
 * client-side is the right amount of engineering; a true paginated backend
 * query is a clean later step if history genuinely outgrows this. CSV/PDF
 * export is cut for the same reason: not worth building for a page whose
 * point is investigation, not reporting. */
export function FireHistoryPage({ alerts, cameras, now, onAcknowledge }: {
  alerts: IrisEvent[]
  cameras: CameraInfo[]
  now: number
  onAcknowledge: (id: number) => void
}) {
  const [openId, setOpenId] = useState<number | null>(null)
  const openEvent = alerts.find(a => a.id === openId) ?? null
  const [range, setRange] = useState<TimeRange>('24h')
  const [cls, setCls] = useState<'all' | FireKind>('all')
  const [activeOnly, setActiveOnly] = useState(false)
  const [search, setSearch] = useState('')

  const filtered = useMemo(() => {
    const rangeMs = RANGE_MS[range]
    const cutoff = rangeMs != null ? now - rangeMs : null
    const q = search.trim().toLowerCase()
    return alerts.filter(ev => {
      if (cutoff != null && new Date(onsetOf(ev)).getTime() < cutoff) return false
      if (cls !== 'all' && fireKind(ev) !== cls) return false
      if (activeOnly && ev.incident_status !== 'ongoing') return false
      if (q) {
        const cam = cameras.find(c => c.id === ev.camera)
        const name = cam ? camLabel(cam, CAM_LABELS) : (ev.camera ?? '')
        const zone = cam?.zone ?? ''
        if (!name.toLowerCase().includes(q) && !zone.toLowerCase().includes(q)) return false
      }
      return true
    })
  }, [alerts, range, cls, activeOnly, search, cameras, now])

  const stats = useMemo(() => computeStats(filtered, now), [filtered, now])

  return (
    <>
      <div className="pageHead">
        <div>
          <h1>سجل حوادث الحريق والدخان</h1>
          <nav className="crumbs" aria-label="مسار التنقل">
            <span>كشف الحريق والدخان</span>
            <span className="crumbSep">‹</span>
            <span className="crumbHere">السجل</span>
          </nav>
        </div>
        <div className="pageHeadTools noPrint">
          <span className="filterTag">{filtered.length} حادثة</span>
          <a className="btnGhost btnSmall" href={productHref('/fire')}>البث المباشر</a>
        </div>
      </div>

      <div className="tableCard historyFilterBar">
        <div className="toolbarFilters">
          <select value={range} onChange={e => setRange(e.target.value as TimeRange)}>
            <option value="1h">آخر ساعة</option>
            <option value="24h">آخر 24 ساعة</option>
            <option value="7d">آخر 7 أيام</option>
            <option value="all">كل الفترات</option>
          </select>
          <select value={cls} onChange={e => setCls(e.target.value as typeof cls)}>
            <option value="all">حريق ودخان</option>
            <option value="fire">حريق فقط</option>
            <option value="smoke">دخان فقط</option>
          </select>
          <label className="historyActiveOnly">
            <input type="checkbox" checked={activeOnly} onChange={e => setActiveOnly(e.target.checked)} />
            المستمر فقط
          </label>
          <input
            className="historySearch"
            type="search"
            placeholder="بحث بالكاميرا أو المنطقة…"
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
        </div>
      </div>

      <div className="historyLayout">
        <section className="historyVideoSection">
          {filtered.length > 0 ? (
            <div className="historyVideoGrid">
              {filtered.map(ev => (
                <HistoryVideoCard key={ev.id} event={ev} cameras={cameras} now={now}
                                  onOpen={() => setOpenId(ev.id)} />
              ))}
            </div>
          ) : (
            <div className="tableCard tableEmpty">لا توجد حوادث مطابقة للفلاتر الحالية</div>
          )}
        </section>

        <aside className="historyRail">
          <MiniBarChart title="الحوادث لكل ساعة (24 ساعة)" bars={stats.perHour} />
          <MiniBarChart title="حسب الكاميرا" bars={stats.perStream} />
          <div className="historySummary">
            <span className="incidentLifecycleLabel">الملخص</span>
            <ul>
              <li>حريق: {stats.fireCount}</li>
              <li>دخان: {stats.smokeCount}</li>
              <li>إجمالي زمن الاحتراق: {fmtDuration(stats.totalBurnSeconds)}</li>
              <li>متوسط المدة: {stats.medianSeconds != null ? fmtDuration(stats.medianSeconds) : '—'}</li>
              <li>
                {stats.longest
                  ? `الأطول: ${fmtDuration(stats.longest.seconds)} · ${stats.longest.stream}`
                  : 'الأطول: —'}
              </li>
            </ul>
          </div>
        </aside>
      </div>

      {openEvent && (
        <IncidentDetailModal
          event={openEvent}
          cameras={cameras}
          now={now}
          onClose={() => setOpenId(null)}
          onAcknowledge={onAcknowledge}
        />
      )}
    </>
  )
}

/** One incident as a video-poster card: the clip is the point of this page
 * (an operator investigating an incident wants to SEE it, not read a row of
 * numbers about it), but every field the old table showed is still here --
 * just laid out on the card instead of in columns. `preload="metadata"`
 * (plus a `#t=` seek) gets a real frame from the clip as a poster without
 * downloading it -- no separate thumbnail endpoint exists, and this reuses
 * the exact clip URL the detail modal already plays. Click/Enter opens that
 * same modal, same as a row click did before. */
function HistoryVideoCard({ event, cameras, now, onOpen }: {
  event: IrisEvent; cameras: CameraInfo[]; now: number; onOpen: () => void
}) {
  const kind = fireKind(event)
  const cam = cameras.find(c => c.id === event.camera) ?? null
  const streamName = cam ? camLabel(cam, CAM_LABELS) : (event.camera ?? '—')
  const active = event.incident_status === 'ongoing'
  const started = new Date(onsetOf(event)).getTime()
  const ended = !active && event.incident_duration_seconds != null
    ? new Date(started + event.incident_duration_seconds * 1000).toISOString()
    : null
  const lasted = active ? (now - started) / 1000 : event.incident_duration_seconds
  const clip = effectiveClips(event)[0]

  return (
    <div
      className={`historyVideoCard ${kind === 'smoke' ? 'historyVideoCardSmoke' : 'historyVideoCardFire'}${active ? ' historyVideoCardActive' : ''}`}
      onClick={onOpen} role="button" tabIndex={0}
      onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onOpen() } }}
    >
      <div className="historyVideoThumb">
        {clip ? (
          <video className="camFrame" src={`${alertClipUrl(clip.clip_id, event.camera ?? '')}#t=0.5`}
                preload="metadata" muted playsInline aria-hidden="true" />
        ) : (
          <div className="camOffline historyVideoPlaceholder">
            {event.clip_status === 'pending' ? 'قيد تجهيز المقطع…' : 'لا يوجد مقطع'}
          </div>
        )}
        {clip && <span className="historyVideoPlay" aria-hidden="true">▶</span>}
        <div className="historyVideoTop">
          <span className={`chip ${kind === 'smoke' ? 'chip-leave' : 'chip-absent'}`}>
            <span className="chipDot" aria-hidden="true" />
            {KIND_LABEL[kind]}{active ? ' — مستمر' : ''}
          </span>
          <span dir="ltr" className={active ? 'fireOngoingChip' : undefined}>
            {lasted != null ? fmtDuration(lasted) : '—'}{active ? ' ▲' : ''}
          </span>
        </div>
      </div>
      <div className="historyVideoInfo">
        <strong dir="ltr">{streamName}</strong>
        <span className="streamTileMeta">{cam?.zone ?? '—'}</span>
        <span className="historyVideoTime" dir="ltr">
          {fmtDateTime(onsetOf(event))}
          {ended && <> – {fmtDateTime(ended)}</>}
          {active && <span className="incidentActiveText"> – لا يزال مستمراً</span>}
        </span>
      </div>
    </div>
  )
}

function MiniBarChart({ title, bars }: { title: string; bars: { label: string; value: number }[] }) {
  const max = Math.max(1, ...bars.map(b => b.value))
  return (
    <div className="miniChart">
      <span className="incidentLifecycleLabel">{title}</span>
      {bars.length === 0 ? (
        <p className="fireUnknown">لا توجد بيانات</p>
      ) : (
        <div className="miniChartBars">
          {bars.map(b => (
            <div key={b.label} className="miniChartRow">
              <span className="miniChartLabel" dir="ltr">{b.label}</span>
              <div className="miniChartTrack">
                <div className="miniChartFill" style={{ width: `${(b.value / max) * 100}%` }} />
              </div>
              <span className="miniChartValue">{b.value}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function computeStats(events: IrisEvent[], now: number) {
  const hourBuckets = new Map<string, number>()
  const streamCounts = new Map<string, number>()
  let fireCount = 0
  let smokeCount = 0
  let totalBurnSeconds = 0
  const durations: { seconds: number; stream: string }[] = []

  for (const ev of events) {
    const kind = fireKind(ev)
    if (kind === 'fire') fireCount += 1
    else if (kind === 'smoke') smokeCount += 1

    const started = new Date(onsetOf(ev)).getTime()
    const hourKey = new Date(started).toLocaleTimeString('en-GB', { hour: '2-digit' })
    hourBuckets.set(hourKey, (hourBuckets.get(hourKey) ?? 0) + 1)

    const streamKey = ev.camera ?? '—'
    streamCounts.set(streamKey, (streamCounts.get(streamKey) ?? 0) + 1)

    const seconds = ev.incident_status === 'ongoing' ? (now - started) / 1000
      : ev.incident_duration_seconds
    if (seconds != null) {
      totalBurnSeconds += seconds
      durations.push({ seconds, stream: streamKey })
    }
  }

  durations.sort((a, b) => a.seconds - b.seconds)
  const medianSeconds = durations.length > 0
    ? durations[Math.floor((durations.length - 1) / 2)].seconds
    : null
  const longest = durations.length > 0 ? durations[durations.length - 1] : null

  return {
    fireCount, smokeCount, totalBurnSeconds, medianSeconds, longest,
    perHour: [...hourBuckets.entries()].map(([label, value]) => ({ label, value }))
      .sort((a, b) => a.label.localeCompare(b.label)),
    perStream: [...streamCounts.entries()].map(([label, value]) => ({ label, value }))
      .sort((a, b) => b.value - a.value),
  }
}
