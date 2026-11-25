import { describe, expect, it } from 'vitest'
import { buildAreas } from '../floor/areas'
import { SHOP } from '../floor/fixtures'
import type { ChairOcc, LiveSnapshot } from '../api/types'
import { holdFor, sidePanelModel, todayFigures } from './sidePanel'

const num = (n: number) => String(Math.round(n))
const areas = buildAreas(SHOP, ['barber-c', 'barber-l', 'barber-r'])

describe('side panel model', () => {
  it('survives a sparse snapshot (no today, alerts, problems, customers)', () => {
    const m = sidePanelModel({ type: 'live_snapshot', server_time: '2026-09-25T12:00:00' }, areas, [], num)
    expect(m.alerts).toEqual([])
    expect(m.zoneProblems).toBe(0)
    expect(m.rosterTodo).toBe(false)
    expect(m.figures.map(f => f.value)).toEqual(['—', '—', '—', '—'])
  })
  it('fills alerts from customers, areas and staff', () => {
    const snap: Partial<LiveSnapshot> = {
      customers: [{ visit_id: 'v1', ticket: 12, haircut_started_at: '2026-09-25T18:10:05' } as LiveSnapshot['customers'][number]],
      targets: { wait_target_s: 600, service_target_s: 1800 },
      open_alerts: [
        { id: 'a', kind: 'excessive_service', camera: 'barber-l', visit_id: 'v1', session_id: 's', started_at: '2026-09-25T18:40:05', ended_at: null, open: true, chair_id: 'R1', barber_id: 'o' },
        { id: 'b', kind: 'excessive_wait', camera: 'barber-c', visit_id: 'gone', session_id: null, started_at: '2026-09-25T18:41:00', ended_at: null, open: true, chair_id: null, barber_id: null,
          has_snapshot: true, subject_cam: 'barber-c', subject_box: [0.4, 0.5, 0.1, 0.3] },
      ],
      roster_confirmed: false,
      zone_problems: [{ code: 'pair_twice_in_camera', pair: 'L1' }],
    }
    const m = sidePanelModel(snap, areas, [{ id: 'o', name: 'Omar', active: true, created_at: '' }], num)
    // the timer runs from the haircut's start; a customer no longer in the
    // live list falls back to the alert's start minus the target
    expect(m.alerts).toEqual([
      { id: 'a', kind: 'excessive_service', visitId: 'v1', chairId: 'R1', chair: 'R1', barber: 'Omar', since: '2026-09-25T18:10:05', target: 1800, stored: null },
      { id: 'b', kind: 'excessive_wait', visitId: 'gone', chairId: null, chair: null, barber: null, since: '2026-09-25T18:31:00', target: 600,
        stored: { cam: 'barber-c', box: [0.4, 0.5, 0.1, 0.3], openedAt: '2026-09-25T18:41:00' } },
    ])
    expect(m.rosterTodo).toBe(true)
    expect(m.zoneProblems).toBe(1)
  })
  it("hides a long haircut while its customer is out of the chair", () => {
    const alert = { id: 'a', kind: 'excessive_service', camera: 'barber-l', visit_id: 'v1', session_id: 'v1', started_at: '2026-09-25T18:40:05', ended_at: null, open: true, chair_id: 'L4', barber_id: null } as const
    const wait = { id: 'b', kind: 'excessive_wait', camera: 'barber-c', visit_id: 'w1', session_id: null, started_at: '2026-09-25T18:41:00', ended_at: null, open: true, chair_id: null, barber_id: null } as const
    const ids = (chairs?: Record<string, Partial<ChairOcc>>) =>
      sidePanelModel({ open_alerts: [alert, wait], chairs: chairs as Record<string, ChairOcc> }, areas, [], num).alerts.map(a => a.id)
    expect(ids({ L4: { state: 'sitting' } })).toEqual(['a', 'b'])
    expect(ids({ L4: { state: 'away' } })).toEqual(['b'])
    expect(ids({ L4: { state: 'starting' } })).toEqual(['b'])
    expect(ids({ L4: { state: 'sitting', checking: '2026-09-25T18:50:00' } })).toEqual(['b'])
    // can't tell (no view), or an older backend without chair states: kept
    expect(ids({ L4: { state: 'no_view' } })).toEqual(['a', 'b'])
    expect(ids()).toEqual(['a', 'b'])
  })
  it('a waiting customer times from the start of his wait', () => {
    const snap: Partial<LiveSnapshot> = {
      customers: [{ visit_id: 'v1', ticket: 7, waiting_started_at: '2026-09-25T18:20:00', wait_target_s: 900 } as LiveSnapshot['customers'][number]],
      open_alerts: [{ id: 'a', kind: 'excessive_wait', camera: 'barber-c', visit_id: 'v1', session_id: null, started_at: '2026-09-25T18:35:00', ended_at: null, open: true, chair_id: null, barber_id: null }],
    }
    expect(sidePanelModel(snap, areas, [], num).alerts[0]).toMatchObject({ visitId: 'v1', since: '2026-09-25T18:20:00', target: 900 })
  })
  it('shows waiting now, visits, served customers and the average wait', () => {
    const f = todayFigures({ visitors: 13, served_customers: 9, average_wait_s: 125 }, 2, num)
    expect(f.map(x => [x.label, x.value])).toEqual([
      ['side.waitingNow', '2'], ['side.visits', '13'], ['side.served', '9'], ['side.avgWait', '02:05'],
    ])
  })
  it('shows a dash for unknown values', () => {
    expect(todayFigures(null, null, num).map(x => x.value)).toEqual(['—', '—', '—', '—'])
  })
  it("shows the server's waiting-area head count, not the live list", () => {
    const c = (state: string) => ({ state }) as LiveSnapshot['customers'][number]
    const snap: Partial<LiveSnapshot> = { customers: [c('waiting'), c('haircut'), c('waiting')], customers_waiting: 9 }
    expect(sidePanelModel(snap, areas, [], num).figures[0].value).toBe('9')
    expect(sidePanelModel({ customers_waiting: 4 }, areas, [], num).figures[0].value).toBe('4')
  })
  it("shows the engine's customers_waiting as the Waiting now figure, '—' when absent", () => {
    expect(sidePanelModel({ customers_waiting: 2 }, areas, [], num).figures[0]).toEqual({ label: 'side.waitingNow', value: '2' })
    expect(sidePanelModel({}, areas, [], num).figures[0].value).toBe('—')
  })
  it('holds a value for a while so it does not tick like a timer', () => {
    let h = holdFor(null, 300, 0, 60_000)
    expect(h).toEqual({ value: 300, at: 0 })
    h = holdFor(h, 305, 30_000, 60_000)
    expect(h.value).toBe(300)
    h = holdFor(h, 310, 60_000, 60_000)
    expect(h).toEqual({ value: 310, at: 60_000 })
    // a first real value after "unknown" shows at once
    expect(holdFor({ value: null, at: 0 }, 42, 1_000, 60_000)).toEqual({ value: 42, at: 1_000 })
  })
})
