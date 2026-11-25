import type { Alert, ChairOcc, LiveSnapshot, Staff, Today } from '../api/types'
import type { AreaTable } from '../floor/areas'
import { storedPicture, type StoredPicture } from './alertPreview'
import type { Key } from '../i18n'
import { fmtTimer } from '../ui/format'

/** Snapshots can arrive partial (an older backend, a hand-edited test
 * frame); the side panel reads every field defensively through here. */
export type LooseSnapshot = Partial<LiveSnapshot>

/** An open alert as a sentence with a live timer: how long the customer has
 * been waiting, or in the chair (`since` = the start of the wait / haircut).
 * No ticket: a customer re-identified mid-visit makes it stale -- the
 * Preview picture shows who it is. */
export interface SideAlert {
  id: string
  kind: Alert['kind']
  visitId: string
  chairId: string | null
  chair: string | null
  barber: string | null
  since: string
  target: number | null
  /** the picture saved when it opened: Preview's fallback while the live scene has no target */
  stored: StoredPicture | null
}
export interface Figure { label: Key; value: string }
export interface SidePanelModel { alerts: SideAlert[]; figures: Figure[]; rosterTodo: boolean; zoneProblems: number }

/** A figure shown as held at `at` (server ms). */
export interface Held { value: number | null; at: number }

/** Keep showing `prev` until `ms` has passed, so a figure that moves on every
 * snapshot (the average wait counts live waits) doesn't tick like a timer.
 * The first value, and the first known value after an unknown one, show at once. */
export function holdFor(prev: Held | null, value: number | null, now: number, ms: number): Held {
  if (prev && prev.value != null && now - prev.at < ms) return prev
  return prev && prev.value === value ? prev : { value, at: now }
}

/** A naive local ISO time `s` seconds before `iso`. */
function secondsBefore(iso: string, s: number): string {
  const d = new Date(Date.parse(iso) - s * 1000)
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
}

/** Who is waiting now, then today's figures, as display strings; '—' when unknown. */
export function todayFigures(today: Partial<Today> | null | undefined, waitingNow: number | null | undefined, num: (n: number) => string): Figure[] {
  const n = (v: number | null | undefined) => (v == null ? '—' : num(v))
  return [
    { label: 'side.waitingNow', value: n(waitingNow) },
    { label: 'side.visits', value: n(today?.visitors) },
    { label: 'side.served', value: n(today?.served_customers) },
    { label: 'side.avgWait', value: fmtTimer(today?.average_wait_s) },
  ]
}

/** A long haircut's customer is in his chair: it sits with no re-ID check running, or its camera can't see it
 * (unknown, kept), or an older backend sends no chair state. Away -- or someone sitting down / being checked
 * there -- hides the alert from the panel until he is back (2026-10-11, user: its Preview showed the empty
 * chair); the alert itself stays open until the haircut ends. */
function inChair(occ: ChairOcc | undefined): boolean {
  const st = occ?.state
  return st === undefined || st === 'no_view' || (st === 'sitting' && !occ?.checking)
}

export function sidePanelModel(snap: LooseSnapshot, areas: AreaTable, staff: Staff[], num: (n: number) => string): SidePanelModel {
  const customers = snap.customers ?? []
  const shown = (snap.open_alerts ?? []).filter(a => {
    const chair = a.chair_id ?? customers.find(x => x.visit_id === a.visit_id)?.chair_id
    return a.kind !== 'excessive_service' || !chair || inChair(snap.chairs?.[chair])
  })
  const alerts = shown.map((a): SideAlert => {
    const c = customers.find(x => x.visit_id === a.visit_id)
    const wait = a.kind !== 'excessive_service'
    const target = wait ? c?.wait_target_s ?? snap.targets?.wait_target_s ?? null : snap.targets?.service_target_s ?? null
    // The alert starts when the wait / haircut passes its target: without the
    // live customer, that start minus the target is when it began.
    const began = wait ? c?.waiting_started_at : c?.haircut_started_at
    return {
      id: a.id,
      kind: wait ? 'excessive_wait' : 'excessive_service',
      visitId: a.visit_id,
      chairId: a.chair_id,
      chair: a.chair_id ? areas.byId[a.chair_id]?.name ?? a.chair_id : null,
      barber: a.barber_id ? staff.find(s => s.id === a.barber_id)?.name ?? null : null,
      since: began ?? (target != null ? secondsBefore(a.started_at, target) : a.started_at),
      target,
      stored: storedPicture(a),
    }
  })
  // the server's count: bodies the centre camera sees in the waiting area (2026-09-28, user)
  return {
    alerts,
    figures: todayFigures(snap.today, snap.customers_waiting, num),
    // Unknown counts as confirmed: no nudge on a snapshot that doesn't say.
    rosterTodo: snap.roster_confirmed === false,
    zoneProblems: (snap.zone_problems ?? []).length,
  }
}
