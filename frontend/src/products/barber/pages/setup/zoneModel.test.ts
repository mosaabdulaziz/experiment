import { describe, expect, it } from 'vitest'
import { addPoint, deleteZone, flipInside, isClosable, movePoint, newZone, pairChoices, setFaceChair, setKind, setOwner, setPair } from './zoneModel'
import type { Zone, ZonesByCam } from '../../api/types'

const chair = (id: string, extra: Partial<Zone> = {}): Zone => ({ id, name: id, type: 'chair', points: [[0, 0], [1, 0], [1, 1]], ...extra })

describe('zoneModel', () => {
  it('makes unique ids per camera', () => {
    const a = newZone('chair', 'barber-l', [])
    const b = newZone('chair', 'barber-l', [a])
    expect(a.id).not.toBe(b.id)
    expect(a.points).toEqual([])
  })
  it('a door line takes two points', () => {
    let z = newZone('entrance_line', 'barber-c', [])
    for (const p of [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]] as [number, number][]) z = addPoint(z, p)
    expect(z.points).toHaveLength(2)
    expect(isClosable(z)).toBe(true)
  })
  it('clamps moved points to the frame', () => {
    expect(movePoint(chair('a'), 0, [-0.2, 1.4]).points[0]).toEqual([0, 1])
  })
  it('deleting a chair deletes its seat', () => {
    const zs = [chair('l-c1'), { id: 'l-s1', name: '', type: 'seat', points: [], seat_of: 'l-c1' } as Zone, chair('l-c2')]
    expect(deleteZone(zs, 'l-c1').map(z => z.id)).toEqual(['l-c2'])
  })
  it('owner is unique across the pair', () => {
    const all: ZonesByCam = { 'barber-c': [chair('c-L1', { pair: 'L1', owner: true })], 'barber-r': [chair('r-L1', { pair: 'L1' })] }
    const out = setOwner(all, 'barber-r', 'r-L1', true)
    expect(out['barber-r'][0].owner).toBe(true)
    expect(out['barber-c'][0].owner).toBeFalsy()
    expect(all['barber-c'][0].owner).toBe(true)                // input untouched
  })
  it('unpairing drops owner too, and keeps unknown fields', () => {
    const all: ZonesByCam = { 'barber-r': [chair('r-L1', { pair: 'L1', owner: true, custom: 7 })] }
    const z = setPair(all, 'barber-r', 'r-L1', null)['barber-r'][0]
    expect(z.pair).toBeUndefined()
    expect(z.owner).toBeUndefined()
    expect(z.custom).toBe(7)
  })
  it('flips the inside point across the door line', () => {
    const z: Zone = { id: 'd', name: '', type: 'entrance_line', points: [[0, 0.5], [1, 0.5]], inside: [0.5, 0.7] }
    expect(flipInside(z).inside![1]).toBeCloseTo(0.3)
  })
  it('lists pair choices of a type', () => {
    const all: ZonesByCam = { a: [chair('x', { pair: 'L2' }), chair('y', { pair: 'L1' })], b: [{ ...chair('w'), type: 'washing', pair: 'W' }] }
    expect(pairChoices(all, 'chair')).toEqual(['L1', 'L2'])
  })

  it('offers the waiting areas already paired, for the sofa seen from every camera', () => {
    const sofa = (id: string, extra: Partial<Zone> = {}): Zone => ({ ...chair(id, extra), type: 'waiting' })
    const all: ZonesByCam = { 'barber-c': [sofa('c-sofa', { pair: 'sofa', owner: true })], 'barber-l': [sofa('l-waiting-1')] }
    expect(pairChoices(all, 'waiting')).toEqual(['sofa'])
    expect(pairChoices(all, 'chair')).toEqual([])
  })

  // validate_zones (barber-service/zones.py) compatibility: an entrance line
  // is only valid once it carries an `inside` point that is not on the line
  // itself (abs(cross)/length >= 1e-3), within the 0..1 frame.
  it('a door line gets a default inside point as soon as it has 2 points, valid per validate_zones', () => {
    let z = newZone('entrance_line', 'barber-c', [])
    z = addPoint(z, [0.1, 0.1])
    expect(z.inside).toBeUndefined()                            // not closable yet, no default needed
    z = addPoint(z, [0.2, 0.2])
    expect(z.points).toHaveLength(2)
    expect(z.inside).toBeDefined()
    const [p1, p2] = z.points
    const inside = z.inside!
    const cross = (p2[0] - p1[0]) * (inside[1] - p1[1]) - (p2[1] - p1[1]) * (inside[0] - p1[0])
    const length = Math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    expect(Math.abs(cross) / length).toBeGreaterThan(1e-3)
    expect(inside[0]).toBeGreaterThanOrEqual(0)
    expect(inside[0]).toBeLessThanOrEqual(1)
    expect(inside[1]).toBeGreaterThanOrEqual(0)
    expect(inside[1]).toBeLessThanOrEqual(1)
  })
  it('a third point on a completed door line is ignored, keeping its inside', () => {
    let z = newZone('entrance_line', 'barber-c', [])
    z = addPoint(z, [0.1, 0.1])
    z = addPoint(z, [0.2, 0.2])
    const inside = z.inside
    z = addPoint(z, [0.3, 0.3])
    expect(z.points).toHaveLength(2)
    expect(z.inside).toEqual(inside)
  })

  // A line flush with a frame edge has its whole length AT that edge (x or y
  // == 0 or 1); naively offsetting outward and clamping lands right back on
  // the line. Cover all four edges, plus a line cutting a corner diagonally.
  const edgeCases: [string, [number, number], [number, number]][] = [
    ['left edge (x=0)', [0, 0.2], [0, 0.7]],
    ['right edge (x=1)', [1, 0.2], [1, 0.7]],
    ['top edge (y=0)', [0.2, 0], [0.7, 0]],
    ['bottom edge (y=1)', [0.2, 1], [0.7, 1]],
    ['diagonal corner line', [0, 0.3], [0.3, 0]],
  ]
  it.each(edgeCases)('gets a valid inside point for a door line on the %s', (_label, p1, p2) => {
    let z = newZone('entrance_line', 'barber-c', [])
    z = addPoint(z, p1)
    z = addPoint(z, p2)
    expect(z.inside).toBeDefined()
    const inside = z.inside!
    const dx = p2[0] - p1[0]
    const dy = p2[1] - p1[1]
    const cross = dx * (inside[1] - p1[1]) - dy * (inside[0] - p1[0])
    const length = Math.hypot(dx, dy)
    expect(Math.abs(cross) / length).toBeGreaterThanOrEqual(1e-3)    // validate_zones's own threshold
    expect(inside[0]).toBeGreaterThanOrEqual(0)
    expect(inside[0]).toBeLessThanOrEqual(1)
    expect(inside[1]).toBeGreaterThanOrEqual(0)
    expect(inside[1]).toBeLessThanOrEqual(1)
  })
  it('setFaceChair sets only that zone\'s chair', () => {
    const f = (id: string): Zone => ({ id, name: id, type: 'face_area', points: [[0, 0], [1, 0], [1, 1]], chair: 'z-r1' })
    const out = setFaceChair([f('f1'), f('f2')], 'f2', 'z-l7')
    expect(out.map(z => z.chair)).toEqual(['z-r1', 'z-l7'])
  })
})

describe('setKind (2026-11-10)', () => {
  it('marks a chair pedicure and clears it again', () => {
    const zs = [chair('c')]
    expect(setKind(zs, 'c', 'pedicure')[0].kind).toBe('pedicure')
    expect('kind' in setKind(setKind(zs, 'c', 'pedicure'), 'c', null)[0]).toBe(false)
  })
})
