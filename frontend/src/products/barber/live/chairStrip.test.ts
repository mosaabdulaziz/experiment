import { describe, expect, it } from 'vitest'
import { buildAreas } from '../floor/areas'
import { SHOP } from '../floor/fixtures'
import type { ChairOcc, LiveCustomer, LiveSnapshot, Staff } from '../api/types'
import { buildBoard } from './selectors'
import { chairStrip, effectiveState } from './chairStrip'

const cust = (p: Partial<LiveCustomer>): LiveCustomer => ({
  visit_id: 'v', person_id: 'p', ticket: 1, state: 'waiting', state_since: '2026-09-25T10:00:00', entered_at: '2026-09-25T10:00:00',
  entry_observed: true, waiting_started_at: '2026-09-25T10:00:00', service_started_at: null, wait_target_s: 900,
  chair_id: null, session_id: null, haircut_started_at: null, barber_id: null, cameras: [], ...p })
const staff: Staff[] = [{ id: 's1', name: 'Ahmed', active: true, created_at: '' }]
const snap = (customers: LiveCustomer[], roster: Record<string, string> = {}) =>
  ({ customers, roster } as unknown as LiveSnapshot)

describe('chairStrip', () => {
  const areas = buildAreas(SHOP)

  it('lists every chair area in row order, left row then right row', () => {
    const s = chairStrip(buildBoard(snap([]), areas, staff))
    expect(s.groups.map(g => g.side)).toEqual(['left', 'right'])
    expect(s.groups[0].chips.map(c => c.area)).toEqual(['L1', 'L2', 'r-L5'])
    expect(s.groups[1].chips.map(c => c.area)).toEqual(['R1'])
    expect(s.groups.flatMap(g => g.chips).every(c => c.kind === 'free')).toBe(true)
    expect(s).toMatchObject({ total: 4, busy: 0, free: 4, noView: 0 })
  })

  it('marks haircuts busy and washes washing, and counts a wash as busy', () => {
    const s = chairStrip(buildBoard(snap([
      cust({ visit_id: 'a', ticket: 12, state: 'haircut', chair_id: 'L2', haircut_started_at: '2026-09-25T10:05:00' }),
      cust({ visit_id: 'b', ticket: 14, state: 'washing', chair_id: 'R1', state_since: '2026-09-25T10:20:00' }),
    ], { L2: 's1' }), areas, staff))
    const [left, right] = s.groups
    expect(left.chips[1]).toMatchObject({ area: 'L2', name: 'L2', kind: 'busy', ticket: 12, since: '2026-09-25T10:05:00',
                                          barber: { name: 'Ahmed' } })
    expect(left.chips[0]).toMatchObject({ kind: 'free', ticket: null, since: null, barber: null })
    expect(right.chips[0]).toMatchObject({ kind: 'washing', ticket: 14, since: '2026-09-25T10:20:00' })
    expect(s).toMatchObject({ total: 4, busy: 2, free: 2 })
    expect(left).toMatchObject({ busy: 1, free: 2 })
    expect(right).toMatchObject({ busy: 1, free: 0 })
  })

  it('is empty without chairs', () => {
    expect(chairStrip(buildBoard(snap([]), buildAreas({}), []))).toEqual({
      groups: [], total: 0, busy: 0, free: 0, noView: 0 })
  })

  describe('with chair states (chair engine)', () => {
    const occ = (p: Partial<ChairOcc>): ChairOcc => ({ person_id: null, session_id: null, sitting: 0, ...p })
    const haircut = cust({ visit_id: 'a', ticket: 12, state: 'haircut', chair_id: 'L2', haircut_started_at: '2026-09-25T10:05:00' })

    it('shows starting, sitting (the haircut timer), away (time left), no view and free', () => {
      const s = chairStrip(buildBoard(snap([
        haircut,
        cust({ visit_id: 'b', ticket: 14, state: 'washing', chair_id: 'R1', state_since: '2026-09-25T10:20:00' }),
      ]), areas, staff), {
        L1: occ({ state: 'starting', since: '2026-09-25T10:30:00', camera: 'c' }),
        L2: occ({ state: 'sitting', since: '2026-09-25T10:25:00', camera: 'c', sitting: 1 }),
        'r-L5': occ({ state: 'no_view', since: null, camera: 'c' }),
        R1: occ({ state: 'away', since: '2026-09-25T10:20:00', resume_until: '2026-09-25T10:28:00', camera: 'c' }),
      })
      const [left, right] = s.groups
      expect(left.chips[0]).toMatchObject({ area: 'L1', kind: 'starting', since: '2026-09-25T10:30:00', ticket: null })
      // the haircut timer keeps running from the haircut start, not this sitting's start
      expect(left.chips[1]).toMatchObject({ area: 'L2', kind: 'busy', ticket: 12, since: '2026-09-25T10:05:00' })
      expect(left.chips[2]).toMatchObject({ area: 'r-L5', kind: 'no_view' })
      expect(right.chips[0]).toMatchObject({ area: 'R1', kind: 'away', ticket: 14, resumeUntil: '2026-09-25T10:28:00' })
      // busy = starting / sitting / away; no view counted on its own
      expect(s).toMatchObject({ total: 4, busy: 3, free: 0, noView: 1 })
      expect(left).toMatchObject({ busy: 2, free: 0, noView: 1 })
    })

    it('gives every card of an ongoing haircut its start time, away included (2026-10-18)', () => {
      const s = chairStrip(buildBoard(snap([haircut]), areas, staff), {
        L1: occ({ state: 'sitting', since: '2026-09-25T10:31:00' }),
        L2: occ({ state: 'sitting', since: '2026-09-25T10:25:00', haircut_since: '2026-09-25T10:05:00' }),
        R1: occ({ state: 'away', since: '2026-09-25T10:20:00', haircut_since: '2026-09-25T10:01:00' }),
        'r-L5': occ({ state: 'empty', since: null }),
      })
      const [left, right] = s.groups
      expect(left.chips[0]).toMatchObject({ kind: 'busy', startedAt: '2026-09-25T10:31:00' })   // its timer's start
      expect(left.chips[1]).toMatchObject({ kind: 'busy', startedAt: '2026-09-25T10:05:00' })
      expect(right.chips[0]).toMatchObject({ kind: 'away', since: '2026-09-25T10:20:00', startedAt: '2026-09-25T10:01:00' })
      expect(left.chips[2]).toMatchObject({ kind: 'free', startedAt: null })
    })

    it('shows a sitter re-ID is still checking, with the previous ticket, as busy', () => {
      const s = chairStrip(buildBoard(snap([
        cust({ visit_id: 'a', ticket: 12, state: 'washing', chair_id: 'L2', state_since: '2026-09-25T10:20:00' }),
      ]), areas, staff), {
        L1: occ({ state: 'sitting', since: '2026-09-25T10:31:00', checking: '2026-09-25T10:31:00', check_ticket: null }),
        L2: occ({ state: 'sitting', since: '2026-09-25T10:24:00', checking: '2026-09-25T10:24:00', check_ticket: 12 }),
      })
      expect(s.groups[0].chips[0]).toMatchObject({ kind: 'checking', ticket: null, since: '2026-09-25T10:31:00' })
      expect(s.groups[0].chips[1]).toMatchObject({ kind: 'checking', ticket: 12, since: '2026-09-25T10:24:00' })
      expect(s.groups[0]).toMatchObject({ busy: 2 })
    })

    it('says how the check decided, with the score, for 5 minutes after it (2026-10-18)', () => {
      const back = { resumed: true, sim: 0.84, threshold: 0.62, gap_s: 95, at: '2026-09-25T10:30:00', ticket: 12 }
      const fresh = { resumed: false, sim: 0.41, threshold: 0.62, gap_s: 60, at: '2026-09-25T10:33:00', ticket: 9 }
      const chairs = {
        L1: occ({ state: 'sitting', since: '2026-09-25T10:29:00', resume: back }),
        L2: occ({ state: 'sitting', since: '2026-09-25T10:32:30', resume: fresh }),
        R1: occ({ state: 'sitting', since: '2026-09-25T10:32:30', checking: '2026-09-25T10:32:30', resume: back }),
      }
      const at = (hhmm: string) => Date.parse(`2026-09-25T${hhmm}:00`)
      let s = chairStrip(buildBoard(snap([]), areas, staff), chairs, at('10:34'))
      expect(s.groups[0].chips[0].decision).toEqual(back)
      expect(s.groups[0].chips[1].decision).toEqual(fresh)
      expect(s.groups[1].chips[0]).toMatchObject({ kind: 'checking', decision: null })    // deciding again: no old answer
      s = chairStrip(buildBoard(snap([]), areas, staff), chairs, at('10:36'))
      expect(s.groups[0].chips[0].decision).toBeNull()                                  // 6 min later: gone
      expect(s.groups[0].chips[1].decision).toEqual(fresh)
      expect(chairStrip(buildBoard(snap([]), areas, staff), chairs).groups[0].chips[0].decision).toEqual(back)
    })

    it('takes the sitting start when the board has no haircut yet, and empty wins over a stale board', () => {
      const s = chairStrip(buildBoard(snap([haircut]), areas, staff), {
        L1: occ({ state: 'sitting', since: '2026-09-25T10:31:00' }),
        L2: occ({ state: 'empty', since: null }),
      })
      expect(s.groups[0].chips[0]).toMatchObject({ kind: 'busy', ticket: null, since: '2026-09-25T10:31:00' })
      expect(s.groups[0].chips[1]).toMatchObject({ kind: 'free', ticket: null, since: null })
    })

    it('keeps the board status for chairs without a state (older backend) and never throws', () => {
      const board = buildBoard(snap([haircut]), areas, staff)
      const before = chairStrip(board)
      expect(chairStrip(board, {})).toEqual(before)
      expect(chairStrip(board, null)).toEqual(before)
      expect(chairStrip(board, { L2: occ({ sitting: 1 }) })).toEqual(before)
      expect(chairStrip(board, { L2: occ({ state: 'weird' as never }) })).toEqual(before)
      expect(before.groups[0].chips[1]).toMatchObject({ kind: 'busy', ticket: 12 })
    })
  })
})

describe('effectiveState (starting debounce)', () => {
  const o = (state: ChairOcc['state'], since?: string | null): ChairOcc => ({ person_id: null, session_id: null, sitting: 0, state, since })
  const now = Date.parse('2026-10-05T12:00:00')
  it('shows starting only once it is 3 s old; missing since counts as not yet', () => {
    expect(effectiveState(o('starting', '2026-10-05T11:59:58'), now)).toBe('empty')
    expect(effectiveState(o('starting', '2026-10-05T11:59:57'), now)).toBe('starting')
    expect(effectiveState(o('starting'), now)).toBe('empty')
    expect(effectiveState(o('starting', '2026-10-05T11:59:58'), now, 1000)).toBe('starting')
  })
  it('passes the other states through immediately, and does nothing without a clock', () => {
    for (const st of ['sitting', 'away', 'no_view', 'empty'] as const) expect(effectiveState(o(st, '2026-10-05T11:59:59'), now)).toBe(st)
    expect(effectiveState(undefined, now)).toBeUndefined()
    expect(effectiveState(o('starting'))).toBe('starting')
  })
  it('the strip shows a young starting chair as free', () => {
    const board = buildBoard(snap([]), buildAreas(SHOP), [])
    const chairs = { L1: o('starting', '2026-10-05T11:59:59') }
    expect(chairStrip(board, chairs, now).groups[0].chips[0]).toMatchObject({ kind: 'free', since: null })
    expect(chairStrip(board, chairs, now + 5000).groups[0].chips[0]).toMatchObject({ kind: 'starting' })
  })
})

it('builds one chip group per board group', () => {
  const card = (area: string, name: string) => ({ area, name, barber: null, status: { kind: 'free' as const } })
  const m = chairStrip({ groups: [{ key: 'right', cards: [card('a', 'R1')] }, { key: 'vip', cards: [card('b', 'VIP1')] }] }, {})
  expect(m.groups.map(g => g.side)).toEqual(['right', 'vip'])
  expect(m.total).toBe(2)
})

describe('chairStrip pedicure chairs (2026-11-10)', () => {
  it('shows a pedicure chair in the Pedicure group with its session start', () => {
    const areas = buildAreas({ 'ph-6': [{ id: 'p1', name: 'P1', type: 'chair', points: [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3]], kind: 'pedicure' }] })
    const s = chairStrip(buildBoard(snap([]), areas, staff), {
      p1: { person_id: null, session_id: 'x', sitting: 1, state: 'sitting', kind: 'pedicure',
            since: '2026-10-16T14:30:00', haircut_since: '2026-10-16T14:16:00' } as ChairOcc })
    expect(s.groups.map(g => g.side)).toEqual(['pedicure'])
    expect(s.groups[0].chips[0]).toMatchObject({ kind: 'busy', ticket: null, since: '2026-10-16T14:16:00' })
  })
})

describe('chairStrip pedicure chips carry no haircut target or barber (final review, 2026-11-10)', () => {
  it('marks a pedicure chip pedicure and a barber chip not', () => {
    const sqp = [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3]]
    const areas = buildAreas({ 'ph-6': [{ id: 'p1', name: 'P1', type: 'chair', points: sqp, kind: 'pedicure' },
                                        { id: 'r3', name: 'R3', type: 'chair', points: sqp }] })
    const s = chairStrip(buildBoard(snap([]), areas, staff), {
      p1: { person_id: null, session_id: 'x', sitting: 1, state: 'sitting', kind: 'pedicure', since: '2026-10-16T14:16:00' } as ChairOcc,
      r3: { person_id: null, session_id: null, sitting: 0, state: 'empty', kind: 'haircut' } as ChairOcc })
    const chips = Object.fromEntries(s.groups.flatMap(g => g.chips).map(c => [c.area, c]))
    expect(chips.p1.pedicure).toBe(true)
    expect(chips.r3.pedicure).toBe(false)
  })
})
