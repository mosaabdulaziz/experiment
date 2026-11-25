import { useEffect, useMemo, useState } from 'react'
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { eventsApi } from '@/api/events'
import { useConfigStore } from '@/stores/configStore'
import type { IrisEvent } from '@/types/event'
import {
  activityTimeline,
  alertRatio,
  computeStats,
  eventsPerHour,
  topTags,
} from '@/components/analytics/aggregations'
import styles from './AnalyticsPage.module.css'

const ACCENT = '#f9996c'
const ALERT = '#ff5e62'
const MUTED = '#6a6a72'
const GRID = 'rgba(255, 255, 255, 0.05)'

const tooltipStyle = {
  background: 'rgba(20, 20, 24, 0.85)',
  backdropFilter: 'blur(16px) saturate(180%)',
  WebkitBackdropFilter: 'blur(16px) saturate(180%)',
  border: '1px solid rgba(255, 255, 255, 0.1)',
  borderRadius: 12,
  fontSize: 12,
  color: '#ececec',
  boxShadow: '0 12px 32px rgba(0, 0, 0, 0.5)',
  padding: '8px 12px',
} as const

export function AnalyticsPage() {
  const [events, setEvents] = useState<IrisEvent[]>([])
  const [loading, setLoading] = useState(true)
  const activePersona = useConfigStore(s => s.persona)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    eventsApi
      .list(1000)
      .then(data => {
        if (!cancelled) setEvents(data)
      })
      .catch(err => console.error('Analytics: failed to load events', err))
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const stats = useMemo(() => computeStats(events), [events])
  const hourly = useMemo(() => eventsPerHour(events), [events])
  const ratio = useMemo(() => alertRatio(events), [events])
  const tags = useMemo(() => topTags(events), [events])
  const timeline = useMemo(() => activityTimeline(events), [events])

  const hasData = events.length > 0

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <div>
          <div className={styles.title}>Analytics</div>
          <div className={styles.subtitle}>
            Aggregated from the last {events.length} events {loading ? '· loading…' : ''}
          </div>
        </div>
      </div>

      <div className={styles.stats}>
        <Stat label="Events today" value={stats.totalToday} hint="Comments + alerts" />
        <Stat
          label="Alerts today"
          value={stats.alertsToday}
          tone="alert"
          hint={stats.totalToday ? `${Math.round((stats.alertsToday / stats.totalToday) * 100)}% of today` : '—'}
        />
        <Stat
          label="Avg per hour"
          value={stats.avgPerHour.toFixed(1)}
          tone="accent"
          hint="Last 24 hours"
        />
        <Stat label="Active persona" value={activePersona || '—'} hint="Current monitoring role" />
      </div>

      <div className={styles.grid}>
        <Card title="Events per hour · last 24h" wide={false}>
          {hasData ? (
            <div className={styles.chartWrap}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={hourly}>
                  <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
                  <XAxis dataKey="hour" stroke={MUTED} fontSize={11} />
                  <YAxis stroke={MUTED} fontSize={11} allowDecimals={false} />
                  <Tooltip contentStyle={tooltipStyle} cursor={{ stroke: ACCENT, opacity: 0.2 }} />
                  <Legend wrapperStyle={{ fontSize: 11, color: MUTED }} />
                  <Line type="monotone" dataKey="comments" stroke={ACCENT} strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="alerts" stroke={ALERT} strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className={styles.empty}>No data yet</div>
          )}
        </Card>

        <Card title="Alert ratio" wide={false}>
          {hasData && ratio.some(r => r.value > 0) ? (
            <div className={styles.chartWrap}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={ratio}
                    dataKey="value"
                    nameKey="name"
                    innerRadius={50}
                    outerRadius={80}
                    paddingAngle={2}
                  >
                    <Cell fill={ACCENT} />
                    <Cell fill={ALERT} />
                  </Pie>
                  <Tooltip contentStyle={tooltipStyle} />
                  <Legend wrapperStyle={{ fontSize: 11, color: MUTED }} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className={styles.empty}>No data yet</div>
          )}
        </Card>

        <Card title="Top tags" wide={true}>
          {tags.length > 0 ? (
            <div className={styles.chartWrap}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={tags} layout="vertical" margin={{ left: 16 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={GRID} horizontal={false} />
                  <XAxis type="number" stroke={MUTED} fontSize={11} allowDecimals={false} />
                  <YAxis type="category" dataKey="tag" stroke={MUTED} fontSize={11} width={120} />
                  <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'rgba(249,153,108,0.06)' }} />
                  <Bar dataKey="count" fill={ACCENT} radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className={styles.empty}>No tags yet</div>
          )}
        </Card>

        <Card title="Activity timeline · 15-min bins, last 24h" wide={true}>
          {hasData ? (
            <div className={styles.chartWrap}>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={timeline}>
                  <defs>
                    <linearGradient id="iris-area" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={ACCENT} stopOpacity={0.6} />
                      <stop offset="100%" stopColor={ACCENT} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
                  <XAxis dataKey="bucket" stroke={MUTED} fontSize={11} interval={Math.max(1, Math.floor(timeline.length / 12))} />
                  <YAxis stroke={MUTED} fontSize={11} allowDecimals={false} />
                  <Tooltip contentStyle={tooltipStyle} cursor={{ stroke: ACCENT, opacity: 0.2 }} />
                  <Area
                    type="monotone"
                    dataKey="count"
                    stroke={ACCENT}
                    strokeWidth={2}
                    fill="url(#iris-area)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className={styles.empty}>No data yet</div>
          )}
        </Card>
      </div>
    </div>
  )
}

function Stat({
  label,
  value,
  hint,
  tone,
}: {
  label: string
  value: number | string
  hint?: string
  tone?: 'accent' | 'alert'
}) {
  const valueClass =
    tone === 'accent' ? styles.statValueAccent : tone === 'alert' ? styles.statValueAlert : ''
  return (
    <div className={styles.stat}>
      <div className={styles.statLabel}>{label}</div>
      <div className={`${styles.statValue} ${valueClass}`}>{value}</div>
      {hint && <div className={styles.statHint}>{hint}</div>}
    </div>
  )
}

function Card({
  title,
  wide,
  children,
}: {
  title: string
  wide: boolean
  children: React.ReactNode
}) {
  return (
    <div className={`${styles.card} ${wide ? styles.cardWide : ''}`}>
      <div className={styles.cardTitle}>{title}</div>
      {children}
    </div>
  )
}
