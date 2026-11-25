import { describe, expect, it } from 'vitest'
import { unseenCount } from './badge'
import type { FallEvent } from './severity'

const ev = (id: number, over: Partial<FallEvent> = {}): FallEvent => ({
  id, camera: 'cam-1', pts: 100, track: 1, event_id: `ev-${id}`, kind: 'fall',
  evaluation: 'falling', combined: 0.8, severity: 'critical', path_scores: {},
  clip_window: null, review_id: null, review_status: null, created_at: '2026-10-28T10:00:00+00:00',
  ...over,
})

describe('the rail badge', () => {
  it('counts what arrived after the cursor', () => {
    expect(unseenCount([ev(3), ev(2), ev(1)], 1)).toBe(2)
  })

  it('counts nothing when the operator has seen everything', () => {
    expect(unseenCount([ev(3), ev(2), ev(1)], 3)).toBe(0)
  })

  it('does not count the quiet events', () => {
    const events = [ev(2, { severity: null, evaluation: 'upright' }), ev(1, { severity: null, evaluation: 'upright' })]
    expect(unseenCount(events, 0)).toBe(0)
  })
})
