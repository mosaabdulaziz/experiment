/** The live event feed: how each /ws message from the IRIS app changes the events the console holds. Pure, so every
 *  message shape is tested (app/server.py broadcasts; see iris's types/ws.ts for the same contract). */
import type { SafetyEvent } from './types'

export type FeedMessage =
  | { type: 'history'; events: SafetyEvent[] }
  | { type: 'event'; event: SafetyEvent }
  | { type: 'event_update'; id: number; event?: SafetyEvent; [field: string]: unknown }
  | { type: 'cleared' }
  | { type: string; [field: string]: unknown }

/** The fields a partial event_update may carry (each path sends only its own: clips, descriptions, incidents, acks). */
const PARTIAL_FIELDS = ['acknowledged_at', 'incident_status', 'clip_status', 'clip_id', 'clips', 'description',
  'description_status', 'incident_duration_seconds'] as const

export function applyMessage(events: Map<number, SafetyEvent>, msg: FeedMessage): Map<number, SafetyEvent> {
  switch (msg.type) {
    case 'history':
      return new Map(((msg as { events: SafetyEvent[] }).events ?? []).map(e => [e.id, e]))
    case 'event': {
      const e = (msg as { event: SafetyEvent }).event
      return new Map(events).set(e.id, e)
    }
    case 'event_update': {
      const m = msg as { id: number; event?: SafetyEvent } & Record<string, unknown>
      const current = events.get(m.id)
      if (!current) return events
      const patch: Partial<SafetyEvent> = m.event ? m.event : {}
      if (!m.event) {
        for (const f of PARTIAL_FIELDS) {
          if (f in m) (patch as Record<string, unknown>)[f] = m[f]
        }
      }
      return new Map(events).set(m.id, { ...current, ...patch })
    }
    case 'cleared':
      return new Map()
    default:
      return events
  }
}

/** A reload of the recent list (on load, and every time the socket reconnects) is the truth for what it covers: what
 *  changed while the console was not listening -- an incident the app's restart closed, an update broadcast while the
 *  socket was down -- shows as it is now, and anything older than the list is let go rather than kept stale. Only what
 *  the socket delivered while the request was on its way (newer than the list) is kept beside it. */
export function resync(events: Map<number, SafetyEvent>, list: SafetyEvent[]): Map<number, SafetyEvent> {
  const newest = list.reduce((m, e) => Math.max(m, e.id), -Infinity)
  const next = new Map(list.map(e => [e.id, e]))
  for (const [id, e] of events) if (id > newest) next.set(id, e)
  return next
}
