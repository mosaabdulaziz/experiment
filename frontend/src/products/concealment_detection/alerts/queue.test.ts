import { describe, expect, it } from 'vitest'
import type { Candidate } from '../api'
import { ARM_DELAY_MS, acknowledge, canAcknowledge, discard, enqueue, type AlarmState } from './queue'

function makeCandidate(id: number): Candidate {
  return {
    id,
    camera: 'aisle_4',
    location: 'Aisle 4',
    track_id: 10 + id,
    pts: id * 2,
    display_time: '00:00:02.00',
    tier: 'high',
    severity: 'warning',
    category: 'concealment',
    kind: 'suspected_concealment',
    score: 0.85,
    zone_id: 'cosmetics',
    target_kind: 'clothing',
    body_zone_name: 'waist',
    terminal_stage: 'RETURNED',
    hold_duration_s: 1.4,
    dwell_duration_s: 3.1,
    description: 'Suspected concealment sequence requiring human verification',
    stage_timeline: [{ stage: 'DWELL', entered_pts: 1.0 }],
    clip_window: [0, 10],
    keyframes_pts: [1, 2, 3],
    clip_path: null,
    thumbnail_path: null,
    requires_human_verification: true,
    status: 'candidate',
    reviewer: null,
    reviewed_at: null,
    note: '',
    created_at: '2026-11-13 12:00:00',
  }
}

describe('alarm queue', () => {
  it('promotes the first event to current and queues subsequent arrivals behind it', () => {
    let q: AlarmState = { current: null, queued: [] }
    q = enqueue(q, makeCandidate(1))
    q = enqueue(q, makeCandidate(2))
    expect(q.current?.id).toBe(1)
    expect(q.queued.map(e => e.id)).toEqual([2])
  })

  it('enforces the 1000 ms arming delay before acknowledge and advances on acknowledge', () => {
    let q: AlarmState = { current: makeCandidate(1), queued: [makeCandidate(2)] }
    expect(canAcknowledge(1000 + ARM_DELAY_MS - 1, 1000)).toBe(false)
    expect(canAcknowledge(1000 + ARM_DELAY_MS, 1000)).toBe(true)
    q = acknowledge(q)
    expect(q.current?.id).toBe(2)
    expect(q.queued).toEqual([])
  })

  it('discards a candidate when reviewed from the Review table', () => {
    let q: AlarmState = { current: makeCandidate(1), queued: [makeCandidate(2)] }
    q = discard(q, 1)
    expect(q.current?.id).toBe(2)
  })
})
