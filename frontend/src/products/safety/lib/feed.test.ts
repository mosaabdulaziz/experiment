import { describe, expect, it } from 'vitest'
import { applyMessage, resync } from './feed'
import type { SafetyEvent } from './types'

const ev = (id: number, over: Partial<SafetyEvent> = {}): SafetyEvent => ({
  id, timestamp: '2026-11-09T10:00:00Z', event_type: 'alert', message: `m${id}`, tags: [], camera: 'store@fire',
  camera_id: 'store', feature: 'fire', severity: 'warning', incident_status: 'ongoing', incident_started_at: null,
  incident_duration_seconds: null, clip_id: null, clip_status: 'pending', clips: [], acknowledged_at: null,
  details: null, ...over,
})

const start = new Map([[1, ev(1)]])

describe('the live event feed', () => {
  it('history replaces everything; an event is added', () => {
    const after = applyMessage(start, { type: 'history', events: [ev(5), ev(6)] })
    expect([...after.keys()]).toEqual([5, 6])
    expect([...applyMessage(after, { type: 'event', event: ev(7) }).keys()]).toEqual([5, 6, 7])
  })
  it('an acknowledgement sets acknowledged_at and nothing else', () => {
    const after = applyMessage(start, { type: 'event_update', id: 1, acknowledged_at: '2026-11-09T10:02:00Z' })
    expect(after.get(1)).toEqual({ ...ev(1), acknowledged_at: '2026-11-09T10:02:00Z' })
  })
  it('a whole record replaces the event; partial updates touch only their fields', () => {
    expect(applyMessage(start, { type: 'event_update', id: 1, event: ev(1, { severity: 'critical' }) })
      .get(1)?.severity).toBe('critical')
    expect(applyMessage(start, { type: 'event_update', id: 1, incident_status: 'resolved' })
      .get(1)?.incident_status).toBe('resolved')
    const clip = applyMessage(start, { type: 'event_update', id: 1, clip_status: 'ready', clip_id: 'c1' }).get(1)
    expect([clip?.clip_status, clip?.clip_id, clip?.message]).toEqual(['ready', 'c1', 'm1'])
  })
  it('ignores updates for events it never saw, and empties on cleared', () => {
    expect(applyMessage(start, { type: 'event_update', id: 99, incident_status: 'resolved' })).toBe(start)
    expect(applyMessage(start, { type: 'cleared' }).size).toBe(0)
  })
})

describe('reloading the list (on load, and whenever the socket reconnects)', () => {
  it('takes the reloaded copy: an incident the app closed while the console was away shows closed', () => {
    // the app restarted: its startup closed the open incident to "unknown" with nobody connected to tell
    const after = resync(start, [ev(1, { incident_status: 'unknown' })])
    expect(after.get(1)?.incident_status).toBe('unknown')
  })
  it('drops what fell out of the reloaded window rather than keep a stale copy of it', () => {
    const held = new Map([[1, ev(1)], [5, ev(5)]])
    expect([...resync(held, [ev(6), ev(5)]).keys()].sort()).toEqual([5, 6])
  })
  it('keeps what the socket delivered while the reload was on its way', () => {
    const held = new Map([[1, ev(1)], [9, ev(9)]])
    expect([...resync(held, [ev(6), ev(5)]).keys()].sort()).toEqual([5, 6, 9])
    expect([...resync(new Map([[3, ev(3)]]), []).keys()]).toEqual([3])
  })
})
