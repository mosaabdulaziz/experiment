import { describe, expect, it } from 'vitest'
import { buildAreas, zoneProblems } from './areas'
import type { Zone, ZonesByCam } from '../api/types'
import { SHOP } from './fixtures'

const sq = (x: number, y: number): number[][] => [[x, y], [x + 0.05, y], [x + 0.05, y + 0.05], [x, y + 0.05]]
const z = (p: Partial<Zone> & Pick<Zone, 'id' | 'type'>): Zone => ({ name: p.id, points: sq(0.1, 0.1), ...p })

// A stale camera still in zones.json (unhooked, e.g. dropped from the live
// camera list) with its own door line and two unpaired chairs -- it must
// not win the centre pick, and must vanish entirely once filtered out.
const WITH_STALE: ZonesByCam = {
  ...SHOP,
  'barber-1': [
    z({ id: '1-door', type: 'entrance_line', points: [[0.4, 0.05], [0.6, 0.05]], inside: [0.5, 0.2] }),
    z({ id: '1-a', type: 'chair', name: 'A' }),
    z({ id: '1-b', type: 'chair', name: 'B' }),
  ],
}

describe('buildAreas', () => {
  it('finds the centre camera by its door line', () => {
    expect(buildAreas(SHOP).centre).toBe('barber-c')
  })
  it('puts chairs in rows by the centre view, top to bottom, unpaired chairs after their camera mates', () => {
    expect(buildAreas(SHOP).rows).toEqual({ left: ['L1', 'L2', 'r-L5'], right: ['R1'] })
  })
  it('records owner, cameras and seat per area and names from the owner zone', () => {
    const a = buildAreas(SHOP).byId
    expect(a.L1).toMatchObject({ type: 'chair', name: 'L1', owner: 'barber-r', drawnIn: ['barber-c', 'barber-r'], hasSeat: true })
    expect(a['c-wash']).toMatchObject({ type: 'washing', owner: 'barber-c' })
  })
  it('copes with no zones at all', () => {
    expect(buildAreas({})).toMatchObject({ centre: null, areas: [], rows: { left: [], right: [] } })
  })
  it('picks the centre by pair count, not just having a door line, so a stale camera cannot win it', () => {
    const t = buildAreas(WITH_STALE)
    expect(t.centre).toBe('barber-c')
    expect(t.rows.right).toEqual(['R1'])
    expect(t.rows.left).toEqual(expect.arrayContaining(['L1', 'L2', 'r-L5', '1-a', '1-b']))
  })
  it('ignores a stale camera entirely when it is not in the given camera list', () => {
    expect(buildAreas(WITH_STALE, ['barber-c', 'barber-l', 'barber-r'])).toEqual(buildAreas(SHOP))
  })
  it('without a door camera, groups chairs by name: right, left, VIP, kids', () => {
    const zones: ZonesByCam = {
      'ph-5': [z({ id: 'z1', type: 'chair', name: 'R2' }), z({ id: 'z2', type: 'chair', name: 'R1' })],
      'ph-3': [z({ id: 'z3', type: 'chair', name: 'L6' }), z({ id: 'z4', type: 'chair', name: 'L5' })],
      'ph-7': [z({ id: 'z5', type: 'chair', name: 'VIP1' })],
      'ph-2': [z({ id: 'z6', type: 'chair', name: 'L10' }), z({ id: 'z7', type: 'chair', name: 'Kids' })],
    }
    const a = buildAreas(zones)
    expect(a.centre).toBeNull()
    expect(a.groups.map(g => [g.key, g.ids.map(id => a.byId[id].name)])).toEqual([
      ['right', ['R1', 'R2']], ['left', ['L5', 'L6', 'L10']], ['vip', ['VIP1']], ['kids', ['Kids']]])
  })
  it('with a door camera, the two rows are the groups', () => {
    const a = buildAreas(SHOP)
    expect(a.groups).toEqual([{ key: 'left', ids: a.rows.left }, { key: 'right', ids: a.rows.right }])
  })
})

describe('zoneProblems', () => {
  it('is empty for a consistent shop', () => {
    expect(zoneProblems(SHOP)).toEqual([])
  })
  it('reports what the backend reports', () => {
    const broken: ZonesByCam = { 'barber-c': [z({ id: 'c-x', type: 'chair', pair: 'X' })], 'barber-l': [z({ id: 'l-y', type: 'chair', pair: 'Y', owner: true, name: 'Y' })] }
    expect(zoneProblems(broken)).toEqual([
      { code: 'pair_single_camera', pair: 'X', camera: 'barber-c' },
      { code: 'pair_owner_count', pair: 'X', count: 0 },
      { code: 'pair_single_camera', pair: 'Y', camera: 'barber-l' },
      { code: 'chair_without_seat', camera: 'barber-l', zone: 'Y' },
    ])
  })
  it('drops a stale camera\'s own problems when filtered out', () => {
    expect(zoneProblems(WITH_STALE)).toEqual([
      { code: 'chair_without_seat', camera: 'barber-1', zone: 'A' },
      { code: 'chair_without_seat', camera: 'barber-1', zone: 'B' },
    ])
    expect(zoneProblems(WITH_STALE, ['barber-c', 'barber-l', 'barber-r'])).toEqual([])
  })
  it('reports a face area naming no chair, a chair of another camera being fine', () => {
    const chair = z({ id: 'z-r1', type: 'chair', name: 'R1' })
    const seat = z({ id: 'z-r1s', type: 'seat', seat_of: 'z-r1' })
    const good = z({ id: 'f1', type: 'face_area', name: 'R1 face', chair: 'z-r1' })
    const bad = z({ id: 'f2', type: 'face_area', name: 'ghost', chair: 'nope' })
    expect(zoneProblems({ 'ph-5': [chair, seat], 'ph-3': [good, bad] }))
      .toEqual([{ code: 'face_area_unknown_chair', camera: 'ph-3', zone: 'ghost' }])
  })
})

describe('pedicure chairs (2026-11-10)', () => {
  it('puts pedicure chairs in their own group, after the others, and out of the rows', () => {
    const t = buildAreas({ 'ph-6': [
      z({ id: 'p2', name: 'P2', type: 'chair', kind: 'pedicure' }),
      z({ id: 'p1', name: 'P1', type: 'chair', kind: 'pedicure' }),
      z({ id: 'r3', name: 'R3', type: 'chair' }),
    ] })
    expect(t.groups.map(g => g.key)).toEqual(['right', 'pedicure'])
    expect(t.groups[1].ids).toEqual(['p1', 'p2'])
    expect(t.byId.p1.kind).toBe('pedicure')
    expect(t.byId.r3.kind).toBe('haircut')
  })

  it('flags a pair drawn as two kinds, like the service', () => {
    const zones: ZonesByCam = {
      a: [z({ id: 'a1', name: 'P1', type: 'chair', pair: 'P1', owner: true, kind: 'pedicure' }), z({ id: 'a1s', type: 'seat', seat_of: 'a1' })],
      b: [z({ id: 'b1', name: 'P1', type: 'chair', pair: 'P1' })],
    }
    expect(zoneProblems(zones)).toContainEqual({ code: 'pair_kind_mismatch', pair: 'P1' })
  })
})
