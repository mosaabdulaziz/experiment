import type { IrisEvent } from '@/types/event'

export interface HourlyBucket {
  hour: string
  comments: number
  alerts: number
}

export interface TimelineBucket {
  bucket: string
  ts: number
  count: number
}

export interface TagCount {
  tag: string
  count: number
}

export interface AlertRatioSlice {
  name: string
  value: number
}

export interface Stats {
  totalToday: number
  alertsToday: number
  avgPerHour: number
  total: number
}

export function eventsPerHour(events: IrisEvent[], hours = 24): HourlyBucket[] {
  const now = new Date()
  const buckets = new Map<number, HourlyBucket>()
  for (let i = hours - 1; i >= 0; i--) {
    const d = new Date(now)
    d.setMinutes(0, 0, 0)
    d.setHours(d.getHours() - i)
    buckets.set(d.getTime(), {
      hour: `${String(d.getHours()).padStart(2, '0')}:00`,
      comments: 0,
      alerts: 0,
    })
  }

  for (const e of events) {
    const t = new Date(e.timestamp)
    if (Number.isNaN(t.getTime())) continue
    t.setMinutes(0, 0, 0)
    const key = t.getTime()
    const b = buckets.get(key)
    if (!b) continue
    if (e.event_type === 'alert') b.alerts += 1
    else b.comments += 1
  }

  return Array.from(buckets.values())
}

export function activityTimeline(events: IrisEvent[], windowHours = 24, binMinutes = 15): TimelineBucket[] {
  const now = Date.now()
  const cutoff = now - windowHours * 3600_000
  const binMs = binMinutes * 60_000
  const numBins = Math.ceil((windowHours * 60) / binMinutes)
  const buckets: TimelineBucket[] = []

  for (let i = numBins - 1; i >= 0; i--) {
    const ts = now - i * binMs
    const d = new Date(ts)
    const label = `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`
    buckets.push({ bucket: label, ts, count: 0 })
  }

  for (const e of events) {
    const t = new Date(e.timestamp).getTime()
    if (Number.isNaN(t) || t < cutoff) continue
    const idx = Math.floor((t - (now - windowHours * 3600_000)) / binMs)
    if (idx >= 0 && idx < buckets.length) {
      buckets[idx].count += 1
    }
  }

  return buckets
}

export function topTags(events: IrisEvent[], n = 10): TagCount[] {
  const counts = new Map<string, number>()
  for (const e of events) {
    for (const tag of e.tags) {
      counts.set(tag, (counts.get(tag) ?? 0) + 1)
    }
  }
  return Array.from(counts.entries())
    .map(([tag, count]) => ({ tag, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, n)
}

export function alertRatio(events: IrisEvent[]): AlertRatioSlice[] {
  let comments = 0
  let alerts = 0
  for (const e of events) {
    if (e.event_type === 'alert') alerts += 1
    else if (e.event_type === 'comment') comments += 1
  }
  return [
    { name: 'Comments', value: comments },
    { name: 'Alerts', value: alerts },
  ]
}

function startOfToday(): number {
  const d = new Date()
  d.setHours(0, 0, 0, 0)
  return d.getTime()
}

export function computeStats(events: IrisEvent[]): Stats {
  const todayStart = startOfToday()
  let totalToday = 0
  let alertsToday = 0
  for (const e of events) {
    const t = new Date(e.timestamp).getTime()
    if (Number.isNaN(t) || t < todayStart) continue
    totalToday += 1
    if (e.event_type === 'alert') alertsToday += 1
  }

  // Average over last 24h based on non-zero coverage
  const hourly = eventsPerHour(events, 24)
  const totalPerHour = hourly.reduce((sum, b) => sum + b.comments + b.alerts, 0)
  const avgPerHour = totalPerHour / 24

  return {
    totalToday,
    alertsToday,
    avgPerHour,
    total: events.length,
  }
}
