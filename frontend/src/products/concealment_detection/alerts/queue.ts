/** The alarm queue: one suspected candidate card at a time, the rest waiting behind it. */
import type { Candidate } from '../api'

/** Attendance & Fall Detection arm delay: an alarm cannot be dismissed before it has been seen. */
export const ARM_DELAY_MS = 1000

export type AlarmState = { current: Candidate | null; queued: Candidate[] }

export function enqueue(state: AlarmState, event: Candidate): AlarmState {
  if (!state.current) return { current: event, queued: state.queued }
  return { current: state.current, queued: [...state.queued, event] }
}

export function acknowledge(state: AlarmState): AlarmState {
  const [next, ...rest] = state.queued
  return { current: next ?? null, queued: rest }
}

/** A resolved row takes its card with it: the table and the card share the same candidate id. */
export function discard(state: AlarmState, candidateId: number): AlarmState {
  if (state.current?.id === candidateId) {
    const [next, ...rest] = state.queued
    return { current: next ?? null, queued: rest }
  }
  return { current: state.current, queued: state.queued.filter(e => e.id !== candidateId) }
}

/** Whether the card may be acknowledged yet. */
export function canAcknowledge(now: number, shownAt: number, armDelayMs: number = ARM_DELAY_MS): boolean {
  return now - shownAt >= armDelayMs
}
