import type { LiveSnapshot, SceneEntry, Staff } from '../api/types'
import type { AreaTable, GroupKey } from '../floor/areas'

export const ticketLabel = (t: number | null | undefined) => (t == null ? '#—' : `#${t}`)

/** How the camera overlay draws a person: staff, a customer in a state (with
 * a ticket), or anyone else -- unknown, or seen but not yet a confirmed
 * visit ("entered") -- as a thin plain box. */
export function boxKind(e: Pick<SceneEntry, 'staff' | 'person_id' | 'state'>): 'staff' | 'customer' | 'other' {
  if (e.staff) return 'staff'
  return e.person_id && e.state && e.state !== 'entered' ? 'customer' : 'other'
}

export type ChairStatus = { kind: 'free' } | { kind: 'haircut'; ticket: number | null; since: string } | { kind: 'washing'; ticket: number | null; since: string }
export interface ChairCard { area: string; name: string; barber: { id: string; name: string; active: boolean } | null; status: ChairStatus }
export interface WaitingTicket { visitId: string; ticket: number | null; since: string; target: number | null }
export interface Board { left: ChairCard[]; right: ChairCard[]; groups: { key: GroupKey; cards: ChairCard[] }[]; waiting: WaitingTicket[] }

export function buildBoard(snap: Partial<Pick<LiveSnapshot, 'customers' | 'roster'>>, areas: AreaTable, staff: Staff[]): Board {
  const customers = snap.customers ?? []
  const roster = snap.roster ?? {}
  const byStaff = new Map(staff.map(s => [s.id, s]))
  const card = (area: string): ChairCard => {
    const sid = roster[area]
    const s = sid ? byStaff.get(sid) : undefined
    const c = customers.find(x => x.chair_id === area && (x.state === 'haircut' || x.state === 'washing'))
    let status: ChairStatus = { kind: 'free' }
    if (c?.state === 'haircut' && c.haircut_started_at) status = { kind: 'haircut', ticket: c.ticket, since: c.haircut_started_at }
    else if (c?.state === 'washing') status = { kind: 'washing', ticket: c.ticket, since: c.state_since ?? c.haircut_started_at ?? c.entered_at }
    return { area, name: areas.byId[area]?.name ?? area,
             barber: sid ? { id: sid, name: s?.name ?? sid, active: s?.active ?? false } : null, status }
  }
  const waiting = customers.filter(c => c.state === 'waiting')
    .map(c => ({ visitId: c.visit_id, ticket: c.ticket, since: c.waiting_started_at ?? c.entered_at, target: c.wait_target_s }))
    .sort((a, b) => a.since.localeCompare(b.since))
  return { left: areas.rows.left.map(card), right: areas.rows.right.map(card), groups: areas.groups.map(g => ({ key: g.key, cards: g.ids.map(card) })), waiting }
}

/** The chair `staffId` leaves when assigned to `chair` (a barber works one chair). */
export function rosterMove(before: Record<string, string>, chair: string, staffId: string | null): string | null {
  if (!staffId) return null
  return Object.entries(before).find(([c, s]) => s === staffId && c !== chair)?.[0] ?? null
}
