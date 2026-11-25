import { describe, expect, it } from 'vitest'
import { buildAreas } from '../floor/areas'
import { boxKind, buildBoard, rosterMove, ticketLabel } from './selectors'
import type { LiveCustomer, LiveSnapshot, Staff } from '../api/types'
// reuse SHOP from areas.test.ts by moving it to src/floor/fixtures.ts (export const SHOP) and importing it in both
import { SHOP } from '../floor/fixtures'

const cust = (p: Partial<LiveCustomer>): LiveCustomer => ({
  visit_id: 'v', person_id: 'p', ticket: 1, state: 'waiting', state_since: '2026-09-25T10:00:00', entered_at: '2026-09-25T10:00:00',
  entry_observed: true, waiting_started_at: '2026-09-25T10:00:00', service_started_at: null, wait_target_s: 900,
  chair_id: null, session_id: null, haircut_started_at: null, barber_id: null, cameras: [], ...p })
const staff: Staff[] = [{ id: 's1', name: 'Ahmed', active: true, created_at: '' }, { id: 's2', name: 'Karim', active: false, created_at: '' }]
const snapWith = (customers: LiveCustomer[], roster: Record<string, string> = {}) =>
  ({ customers, roster, zones: SHOP, chairs: {}, targets: { wait_target_s: 900, service_target_s: 1800 } } as unknown as LiveSnapshot)

describe('buildBoard', () => {
  const areas = buildAreas(SHOP)
  it('shows free chairs with their barber', () => {
    const b = buildBoard(snapWith([], { L1: 's1' }), areas, staff)
    expect(b.left.map(c => c.area)).toEqual(['L1', 'L2', 'r-L5'])
    expect(b.left[0]).toMatchObject({ name: 'L1', barber: { name: 'Ahmed', active: true }, status: { kind: 'free' } })
    expect(b.left[1].barber).toBeNull()
  })
  it('shows an inactive barber still on the roster', () => {
    expect(buildBoard(snapWith([], { L2: 's2' }), areas, staff).left[1].barber).toMatchObject({ name: 'Karim', active: false })
  })
  it('puts a haircut and a wash on their chair', () => {
    const b = buildBoard(snapWith([
      cust({ visit_id: 'a', ticket: 12, state: 'haircut', chair_id: 'L1', haircut_started_at: '2026-09-25T10:05:00' }),
      cust({ visit_id: 'b', ticket: 14, state: 'washing', chair_id: 'R1', state_since: '2026-09-25T10:20:00' }),
    ]), areas, staff)
    expect(b.left[0].status).toEqual({ kind: 'haircut', ticket: 12, since: '2026-09-25T10:05:00' })
    expect(b.right[0].status).toEqual({ kind: 'washing', ticket: 14, since: '2026-09-25T10:20:00' })
  })
  it('lists the waiting longest first', () => {
    const b = buildBoard(snapWith([
      cust({ visit_id: 'late', ticket: 3, waiting_started_at: '2026-09-25T10:10:00' }),
      cust({ visit_id: 'early', ticket: 2, waiting_started_at: '2026-09-25T10:01:00' }),
    ]), areas, staff)
    expect(b.waiting.map(w => w.ticket)).toEqual([2, 3])
  })
  it('tolerates a sparse snapshot', () => {
    const sparse = { customers: [cust({ state: 'haircut', chair_id: 'gone', ticket: null })] } as unknown as LiveSnapshot
    expect(() => buildBoard(sparse, buildAreas({}), [])).not.toThrow()
    expect(buildBoard(sparse, buildAreas({}), [])).toEqual({ left: [], right: [], groups: [], waiting: [] })
  })
})

describe('labels and roster', () => {
  it('ticketLabel', () => {
    expect(ticketLabel(12)).toBe('#12')
    expect(ticketLabel(null)).toBe('#—')
  })
  it('rosterMove names the chair a barber left', () => {
    expect(rosterMove({ R1: 's1', L2: 's2' }, 'L1', 's1')).toBe('R1')
    expect(rosterMove({ L1: 's1' }, 'L1', 's1')).toBeNull()
    expect(rosterMove({ R1: 's1' }, 'L1', null)).toBeNull()
  })
})

describe('boxKind', () => {
  const e = { staff: false, person_id: 'p1', state: 'waiting' as const }
  it('boxes staff, stated customers, and everyone else apart', () => {
    expect(boxKind({ ...e, staff: true })).toBe('staff')
    expect(boxKind(e)).toBe('customer')
    expect(boxKind({ ...e, person_id: null })).toBe('other')
    expect(boxKind({ ...e, state: null })).toBe('other')
  })
  it('keeps an unconfirmed (entered) customer a plain box, not a ticket', () => {
    expect(boxKind({ ...e, state: 'entered' })).toBe('other')
  })
})
