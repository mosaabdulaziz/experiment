/** What the product saw, and where each kind of event speaks.
 *
 * The three channels are the house's: a quiet notice, the alarm card, and the rail badge.
 * Which one an event uses is a rule, not a taste -- the rule is what decides whether an
 * operator is interrupted, so it lives here as a pure function (phase-5 §3).
 */

export type Severity = 'critical' | 'warning'

export type FallEvent = {
  id: number
  camera: string
  pts: number | null
  track: number | null
  event_id: string | null
  kind: string | null
  evaluation: string
  combined: number
  severity: Severity | null
  path_scores: Record<string, number | null>
  clip_window: [number, number] | null
  review_id: number | null
  review_status: string | null
  created_at: string
}

export type Channels = { toast: boolean; card: boolean; badge: boolean }

/** Where an event speaks.
 *
 * A confirmed fall takes over: the person is on the floor. A candidate below the confirm
 * threshold takes over too: it was queued, and a queued candidate is a human waiting on it.
 * A confirmed lying or immobile only notifies -- nobody is queued on it. The closed upright
 * reading is the false alarm the product rejected, and it says nothing at all.
 */
export function channels(event: FallEvent): Channels {
  if (!event.severity) return { toast: false, card: false, badge: false }
  const queued = event.review_id != null
  return { toast: true, card: event.severity === 'critical' || queued, badge: true }
}
