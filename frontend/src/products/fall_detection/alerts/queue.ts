/** The alarm queue: one card at a time, the rest waiting behind it.
 *
 * A fall is not a thing the operator can batch: the first one gets the card, and the ones
 * that arrive while it is up wait their turn with their count on it (phase-5 §3).
 */
import type { FallEvent } from './severity'

/** Attendance's arm delay: an alarm cannot be dismissed before it has been seen. */
export const ARM_DELAY_MS = 1000

export type AlarmState = { current: FallEvent | null; queued: FallEvent[] }

export function enqueue(state: AlarmState, event: FallEvent): AlarmState {
  if (!state.current) return { current: event, queued: state.queued }
  return { current: state.current, queued: [...state.queued, event] }
}

export function acknowledge(state: AlarmState): AlarmState {
  const [next, ...rest] = state.queued
  return { current: next ?? null, queued: rest }
}

/** A resolved row takes its card with it: the table and the card are the same verdict. */
export function discard(state: AlarmState, reviewId: number): AlarmState {
  if (state.current?.review_id === reviewId) {
    const [next, ...rest] = state.queued
    return { current: next ?? null, queued: rest }
  }
  return { current: state.current, queued: state.queued.filter(e => e.review_id !== reviewId) }
}

/** Whether the card may be acknowledged yet. */
export function canAcknowledge(now: number, shownAt: number, armDelayMs: number = ARM_DELAY_MS): boolean {
  return now - shownAt >= armDelayMs
}
