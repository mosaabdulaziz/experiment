import { describe, expect, it } from 'vitest'
import { acknowledge, ARM_DELAY_MS, canAcknowledge, discard, enqueue, type AlarmState } from './queue'
import type { FallEvent } from './severity'

const ev = (id: number, reviewId: number | null = null): FallEvent => ({
  id, camera: 'cam-1', pts: 100, track: 1, event_id: `ev-${id}`, kind: 'fall',
  evaluation: 'falling', combined: 0.8, severity: 'critical', path_scores: {},
  clip_window: null, review_id: reviewId, review_status: null, created_at: '2026-10-28T10:00:00+00:00',
})

describe('the alarm queue', () => {
  it('the first alarm is the current one', () => {
    const state = enqueue({ current: null, queued: [] }, ev(1))
    expect(state.current?.id).toBe(1)
    expect(state.queued).toEqual([])
  })

  it('the ones behind it wait their turn', () => {
    let state: AlarmState = { current: null, queued: [] }
    state = enqueue(state, ev(1))
    state = enqueue(state, ev(2))
    state = enqueue(state, ev(3))
    expect(state.current?.id).toBe(1)
    expect(state.queued.map(e => e.id)).toEqual([2, 3])
  })

  it('acknowledging brings the next one forward', () => {
    let state: AlarmState = { current: null, queued: [] }
    state = enqueue(state, ev(1))
    state = enqueue(state, ev(2))
    state = acknowledge(state)
    expect(state.current?.id).toBe(2)
    expect(state.queued).toEqual([])
  })

  it('acknowledging the last one leaves nothing', () => {
    const state = acknowledge(enqueue({ current: null, queued: [] }, ev(1)))
    expect(state.current).toBeNull()
  })

  it('a verdict from the table takes its card with it', () => {
    let state: AlarmState = { current: null, queued: [] }
    state = enqueue(state, ev(1, 10))
    state = enqueue(state, ev(2, 11))
    state = enqueue(state, ev(3, 12))
    // the queued one, resolved in the table: it leaves the queue, nothing else moves
    expect(discard(state, 12)).toEqual({ current: state.current, queued: [state.queued[0]] })
    // the current one, resolved in the table: the next comes forward
    const promoted = discard(state, 10)
    expect(promoted.current?.id).toBe(2)
    expect(promoted.queued.map(e => e.id)).toEqual([3])
  })
})

describe('the arm delay', () => {
  it('an alarm cannot be acknowledged before it has been on screen', () => {
    expect(canAcknowledge(1000, 1000)).toBe(false)
    expect(canAcknowledge(1000 + ARM_DELAY_MS - 1, 1000)).toBe(false)
  })

  it('after the delay it can', () => {
    expect(canAcknowledge(1000 + ARM_DELAY_MS, 1000)).toBe(true)
  })
})
