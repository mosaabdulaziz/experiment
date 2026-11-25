// The rule that decides whether an operator is interrupted (phase-5 §3):
//   a confirmed fall            -> toast, card, badge
//   a candidate below the gate  -> toast, card, badge  (it is queued; a human must act)
//   a confirmed lying/immobile  -> toast, badge        (nobody is waiting on it)
//   a closed upright            -> nothing             (the false alarm the product rejected)
import { describe, expect, it } from 'vitest'
import { channels, type FallEvent } from './severity'

function event(over: Partial<FallEvent>): FallEvent {
  return {
    id: 1, camera: 'cam-1', pts: 100, track: 1, event_id: 'ev-1', kind: 'fall',
    evaluation: 'falling', combined: 0.81, severity: 'critical',
    path_scores: {}, clip_window: null, review_id: null, review_status: null,
    created_at: '2026-10-28T10:00:00+00:00', ...over,
  }
}

describe('which channel an event speaks on', () => {
  it('a confirmed fall takes over and counts', () => {
    expect(channels(event({ severity: 'critical', review_id: null })))
      .toEqual({ toast: true, card: true, badge: true })
  })

  it('a candidate below the threshold takes over and counts', () => {
    expect(channels(event({ severity: 'warning', evaluation: 'lying', review_id: 7 })))
      .toEqual({ toast: true, card: true, badge: true })
  })

  it('a confirmed lying only notifies and counts', () => {
    expect(channels(event({ severity: 'warning', evaluation: 'lying', review_id: null })))
      .toEqual({ toast: true, card: false, badge: true })
  })

  it('the closed false alarm says nothing at all', () => {
    expect(channels(event({ severity: null, evaluation: 'upright' })))
      .toEqual({ toast: false, card: false, badge: false })
  })
})
