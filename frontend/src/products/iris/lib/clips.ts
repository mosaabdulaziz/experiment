import type { IrisEvent } from '@/types/event'

/** Backend default for `alerts.clip.ttl_seconds` (6 hours). */
export const DEFAULT_CLIP_TTL_SECONDS = 21600

export type ClipState = 'none' | 'pending' | 'ready' | 'expired' | 'unavailable'

/**
 * What can actually be done with this event's footage right now.
 *
 * Age is measured from the *event* timestamp while the store's TTL runs from when
 * the clip was added, 15-20s later. That marks a clip expired up to ~20s early out
 * of a 6h window — the safe direction, since the alternative is offering a play
 * button for bytes that were already pruned.
 */
export function clipState(
  ev: IrisEvent,
  ttlSeconds: number = DEFAULT_CLIP_TTL_SECONDS,
  now: number = Date.now(),
): ClipState {
  if (ev.event_type !== 'alert' || !ev.clip_status) return 'none'
  if (ev.clip_status === 'pending') return 'pending'
  if (ev.clip_status === 'unavailable') return 'unavailable'
  // 'ready' with no id can never be played; treat it as a failure, not a promise.
  if (!ev.clip_id) return 'unavailable'
  const started = new Date(ev.timestamp).getTime()
  if (Number.isNaN(started)) return 'ready'
  return (now - started) / 1000 > ttlSeconds ? 'expired' : 'ready'
}

export type AlertKind = 'intrusion' | 'exit' | 'overcrowding' | 'proximity' | 'ai'

/**
 * Alert type is carried as a tag, not a column. Zone rules emit exactly one of the
 * three `zone_*` tags; anything else is a proactive VLM alert whose tags are
 * free-form model output.
 */
export function alertKind(ev: IrisEvent): AlertKind {
  if (ev.tags.includes('zone_intrusion')) return 'intrusion'
  if (ev.tags.includes('zone_exit')) return 'exit'
  if (ev.tags.includes('zone_overcrowding')) return 'overcrowding'
  if (ev.tags.includes('proximity')) return 'proximity'
  return 'ai'
}

export const ALERT_KIND_LABEL: Record<AlertKind, string> = {
  intrusion: 'Intrusion',
  exit: 'Zone exit',
  overcrowding: 'Overcrowding',
  proximity: 'Proximity',
  ai: 'AI alert',
}

export interface DayGroup {
  key: string
  label: string
  items: IrisEvent[]
}

function dayKey(d: Date): string {
  return `${d.getFullYear()}-${d.getMonth()}-${d.getDate()}`
}

/**
 * Split events into day buckets, newest day first and newest event first within
 * each day. Expects `events` already filtered; ordering is applied here so callers
 * cannot get it subtly wrong.
 */
export function groupByDay(events: IrisEvent[], now: number = Date.now()): DayGroup[] {
  const today = new Date(now)
  const yesterday = new Date(now - 86400_000)
  const buckets = new Map<string, IrisEvent[]>()

  const sorted = [...events].sort(
    (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime(),
  )

  for (const ev of sorted) {
    const d = new Date(ev.timestamp)
    if (Number.isNaN(d.getTime())) continue
    const key = dayKey(d)
    const bucket = buckets.get(key)
    if (bucket) bucket.push(ev)
    else buckets.set(key, [ev])
  }

  return [...buckets.entries()].map(([key, items]) => {
    const d = new Date(items[0].timestamp)
    let label: string
    if (key === dayKey(today)) label = 'Today'
    else if (key === dayKey(yesterday)) label = 'Yesterday'
    else label = d.toLocaleDateString([], { weekday: 'short', day: 'numeric', month: 'short' })
    return { key, label, items }
  })
}
