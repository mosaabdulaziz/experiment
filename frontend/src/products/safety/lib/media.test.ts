import { describe, expect, it } from 'vitest'
import { mediaOf } from './media'
import type { SafetyEvent } from './types'

const now = Date.parse('2026-11-09T12:00:00Z')
const ev = (over: Partial<SafetyEvent>): SafetyEvent => ({
  id: 9, timestamp: '2026-11-09T11:59:00Z', event_type: 'alert', message: 'm', tags: [], camera: 'store@fire',
  camera_id: 'store', feature: 'fire', severity: 'critical', incident_status: null, incident_started_at: null,
  incident_duration_seconds: null, clip_id: 'c1', clip_status: 'ready', clips: [], acknowledged_at: null,
  details: null, ...over,
})

describe('what an incident shows as its picture', () => {
  it('its snapshot when it has one, else its clip at the alert moment', () => {
    expect(mediaOf(ev({ snapshot_id: 's1' }), now)).toEqual(
      { kind: 'snapshot', src: '/api/alerts/snapshot/s1/store%40fire/9', clip: '/api/alerts/clip/c1/store%40fire' })
    expect(mediaOf(ev({}), now)).toEqual({ kind: 'clip', clipId: 'c1', clip: '/api/alerts/clip/c1/store%40fire' })
  })
  it('an escalation clip when the first one failed', () => {
    expect(mediaOf(ev({ clip_id: null, clip_status: 'unavailable', clips: [{ clip_id: 'c2', status: 'ready' }] }), now))
      .toMatchObject({ kind: 'clip', clipId: 'c2' })
  })
  it('says why there is no picture', () => {
    expect(mediaOf(ev({ clip_status: 'pending', clip_id: null }), now)).toEqual({ kind: 'pending' })
    expect(mediaOf(ev({ clip_status: 'unavailable', clip_id: null }), now)).toEqual({ kind: 'unavailable' })
    expect(mediaOf(ev({ timestamp: '2026-11-08T00:00:00Z' }), now)).toEqual({ kind: 'expired' })
    expect(mediaOf(ev({ clip_status: null, clip_id: null }), now)).toEqual({ kind: 'none' })
  })
  it('a snapshot expires with its clip (it is a still of that clip: app/clip_store.py)', () => {
    expect(mediaOf(ev({ snapshot_id: 's1', timestamp: '2026-11-08T00:00:00Z' }), now)).toEqual({ kind: 'expired' })
  })
})
