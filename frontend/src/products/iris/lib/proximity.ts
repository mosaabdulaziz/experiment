import type { IrisEvent } from '@/types/event'

// Only for the "≈ cm" hint: gaps are measured in person heights, with no metric calibration.
const PERSON_HEIGHT_CM = 170

export interface ProximityFacts {
  zone: string | null
  vehicleClass: string | null
  gap: number | null            // person heights
  gapCm: number | null
  closeFor: number | null       // seconds
  persons: string[]
  vehicles: string[]
  unconfirmed: boolean
  depthConfirmed: boolean
  escalatedFrom: string | null
  snapshotExact: boolean
  depthGap: number | null       // the depth check's relative depth gap for the pair (a depth-check zone)
  depthMaxGap: number | null    // ... at or under which it counts as the same depth
  depthImage: boolean           // the app keeps a depth image of the pair (GET /api/alerts/depth/<event id>)
  heldBack: boolean             // a pair the depth check held back (Ticket 6): kept, never alerted
  wouldBe: string | null        // ... and the severity it would have alerted at
}

/** A pair the depth check held back: a quiet event the Incidents page shows behind a filter and the live view never
 *  shows. Told by the app's tag, or by the module's own field. */
export function isDepthSuppressed(ev: IrisEvent): boolean {
  return (ev.tags ?? []).includes('depth-suppressed') || ev.details?.suppressed === 'depth'
}

/** What the live view keeps of the events the WebSocket sends: comments and alerts, never a held-back pair. */
export function forLiveView(events: IrisEvent[]): IrisEvent[] {
  return events.filter(ev => (ev.event_type === 'comment' || ev.event_type === 'alert') && !isDepthSuppressed(ev))
}

const num = (v: string | undefined): number | null => {
  const n = v === undefined ? NaN : Number(v)
  return Number.isFinite(n) ? n : null
}
const list = (v: string | undefined): string[] => (v ? v.split(',').map(s => s.trim()).filter(Boolean) : [])

export const SEVERITY_LABEL: Record<string, string> = { critical: 'Critical', warning: 'Warning', info: 'Unconfirmed' }

/** A proximity incident's facts, from the fields the module sent (event.details, all strings). */
export function proximityFacts(ev: IrisEvent): ProximityFacts {
  const d = ev.details ?? {}
  const gap = num(d.min_gap) ?? num(d.gap)
  return {
    zone: d.zone_name ?? null,
    vehicleClass: d.vehicle_class ?? null,
    gap,
    gapCm: gap === null ? null : Math.round(gap * PERSON_HEIGHT_CM),
    closeFor: ev.incident_duration_seconds ?? num(d.duration_s),
    persons: list(d.person_ids),
    vehicles: list(d.vehicle_ids),
    unconfirmed: d.confirmation === 'none',
    depthConfirmed: d.confirmation === 'depth',
    escalatedFrom: d.escalated_from ?? null,
    snapshotExact: d.snapshot_exact !== '0',
    depthGap: num(d.depth_gap),
    depthMaxGap: num(d.depth_max_gap),
    depthImage: d.depth_image === '1',
    heldBack: isDepthSuppressed(ev),
    wouldBe: d.would_be ?? null,
  }
}

/** "Ongoing" / "Resolved after 3.4 s" / "Ended — not followed to the end", plus "escalated to …". */
export function incidentStatusLine(ev: IrisEvent, facts: ProximityFacts): string | null {
  const parts: string[] = []
  if (ev.incident_status === 'ongoing') parts.push('Ongoing')
  else if (ev.incident_status === 'resolved')
    parts.push(facts.closeFor !== null ? `Resolved after ${facts.closeFor.toFixed(1)} s` : 'Resolved')
  else if (ev.incident_status === 'unknown') parts.push('Ended — not followed to the end (a restart)')
  if (facts.escalatedFrom && ev.severity)
    parts.push(`escalated to ${(SEVERITY_LABEL[ev.severity] ?? ev.severity).toLowerCase()}`)
  return parts.length ? parts.join(' · ') : null
}
