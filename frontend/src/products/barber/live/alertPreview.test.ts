import { describe, expect, it } from 'vitest'
import type { Alert, LiveCustomer, Scene, SceneEntry, ZonesByCam } from '../api/types'
import type { Area, AreaTable } from '../floor/areas'
import { canPreview, previewTarget, shadow, storedPicture } from './alertPreview'

const area = (id: string, type: Area['type'], owner: string): Area => ({ id, type, name: id, owner, drawnIn: [], hasSeat: false, kind: 'haircut' })
const byId = { R1: area('R1', 'chair', 'barber-l'), 'c-sofa': area('c-sofa', 'waiting', 'barber-c') }
const areas: AreaTable = { centre: 'barber-c', areas: Object.values(byId), byId, rows: { left: [], right: ['R1'] }, groups: [{ key: 'right', ids: ['R1'] }] }
const customer = { visit_id: 'v1', person_id: 'p1', chair_id: 'R1' } as LiveCustomer
const entry = (person: string | null, box: SceneEntry['box'], a: string[] = []) => ({ track_id: 1, person_id: person, box, areas: a }) as SceneEntry
const alert = (kind: Alert['kind'], chair: string | null = null) => ({ kind, visit_id: 'v1', chair_id: chair }) as Alert

describe('alert preview target', () => {
  const square = [[0.5, 0.4], [0.7, 0.4], [0.7, 0.8], [0.5, 0.8]]
  const zones: ZonesByCam = {
    'barber-c': [{ id: 'c-R1', name: 'R1', type: 'chair', pair: 'R1', points: [[0, 0], [0.1, 0], [0.1, 0.1]] }],
    'barber-l': [{ id: 'l-R1-seat', name: 'R1 seat', type: 'seat', seat_of: 'l-R1', points: [[0.55, 0.6], [0.65, 0.6], [0.6, 0.7]] },
                 { id: 'l-R1', name: 'R1', type: 'chair', pair: 'R1', owner: true, points: square }],
  }
  it("a long haircut shows its chair: the owner camera's chair zone, whoever the scene sees there", () => {
    const want = { cam: 'barber-l', box: [0.5, 0.4, 0.2, 0.4], outline: square }
    const t = previewTarget(alert('excessive_service', 'R1'), [customer], {}, areas, zones)!
    expect(t.cam).toBe(want.cam)
    expect(t.outline).toEqual(want.outline)
    t.box.forEach((v, i) => expect(v).toBeCloseTo(want.box[i]))
    // no customer row, no one tracked: still the chair
    expect(previewTarget(alert('excessive_service', 'R1'), [], {}, areas, zones)?.cam).toBe('barber-l')
  })
  it('a long haircut has no preview without the chair zone on its owner camera', () => {
    expect(previewTarget(alert('excessive_service', 'R1'), [customer], {}, areas, { 'barber-c': zones['barber-c'] })).toBeNull()
    expect(previewTarget(alert('excessive_service', null), [], {}, areas, zones)).toBeNull()
  })
  it("a long wait shows the owner camera of the waiting area he's in", () => {
    const scene: Scene = { 'barber-r': [entry('p1', [0.2, 0.2, 0.1, 0.1])], 'barber-c': [entry('p1', [0.4, 0.3, 0.05, 0.1], ['c-sofa'])] }
    expect(previewTarget(alert('excessive_wait'), [customer], scene, areas)).toEqual({ cam: 'barber-c', box: [0.4, 0.3, 0.05, 0.1] })
  })
  it('a long wait uses the waiting area\'s owner camera even while he stands outside the zone', () => {
    const scene: Scene = { 'barber-r': [entry('p1', [0.2, 0.2, 0.1, 0.1], ['c-sofa'])], 'barber-c': [entry('p1', [0.4, 0.3, 0.05, 0.1])] }
    expect(previewTarget(alert('excessive_wait'), [customer], scene, areas)?.cam).toBe('barber-c')
  })
  it('never a camera that does not own the area: none when the owner does not see him', () => {
    const side: Scene = { 'barber-r': [entry('p1', [0.2, 0.2, 0.1, 0.1], ['c-sofa'])] }
    expect(previewTarget(alert('excessive_wait'), [customer], side, areas)).toBeNull()
  })
  it('no waiting area set up: the centre camera', () => {
    const bare: AreaTable = { ...areas, areas: [byId.R1], byId: { R1: byId.R1 } }
    const scene: Scene = { 'barber-r': [entry('p1', [0.2, 0.2, 0.1, 0.1])], 'barber-c': [entry('p1', [0.4, 0.3, 0.05, 0.1])] }
    expect(previewTarget(alert('excessive_wait'), [customer], scene, bare)?.cam).toBe('barber-c')
  })
  it('none while the customer is not in the live scene', () => {
    const scene: Scene = { 'barber-c': [entry('p2', [0, 0, 0.1, 0.1])] }
    expect(previewTarget(alert('excessive_wait'), [customer], scene, areas)).toBeNull()
    expect(previewTarget(alert('excessive_wait'), [], { 'barber-c': [entry('p1', [0, 0, 0.1, 0.1])] }, areas)).toBeNull()
  })
})

describe('shadow tracking', () => {
  const t1 = { cam: 'barber-c', box: [0.1, 0.1, 0.1, 0.2] as SceneEntry['box'] }
  const t2 = { cam: 'barber-c', box: [0.2, 0.1, 0.1, 0.2] as SceneEntry['box'] }
  it('keeps the last position through a short dropout, then lets go', () => {
    let s = shadow(null, t1, 0, 10_000)
    expect(s).toEqual({ target: t1, at: 0 })
    s = shadow(s, null, 4_000, 10_000)
    expect(s).toEqual({ target: t1, at: 0 })          // held: last seen at 0
    expect(shadow(s, null, 10_001, 10_000)).toBeNull()
    expect(shadow(s, t2, 5_000, 10_000)).toEqual({ target: t2, at: 5_000 })
  })
  it('never seen stays none', () => {
    expect(shadow(null, null, 0, 10_000)).toBeNull()
  })
})

describe('the stored picture fallback (2026-10-18)', () => {
  const saved = { has_snapshot: true, subject_cam: 'ph-3', subject_box: [0.1, 0.2, 0.1, 0.3] as [number, number, number, number],
                  started_at: '2026-10-18T11:20:30' }
  it('an alert with a saved snapshot has a stored picture: its camera, box and open time', () => {
    expect(storedPicture(saved)).toEqual({ cam: 'ph-3', box: [0.1, 0.2, 0.1, 0.3], openedAt: '2026-10-18T11:20:30' })
  })
  it('none without the snapshot, its camera or its box', () => {
    expect(storedPicture({ ...saved, has_snapshot: false })).toBeNull()
    expect(storedPicture({ ...saved, subject_cam: null })).toBeNull()
    expect(storedPicture({ ...saved, subject_box: null })).toBeNull()
  })
  it('Preview works with a live target or a stored picture, not with neither', () => {
    const seen = { target: { cam: 'ph-3', box: [0, 0, 0.1, 0.1] as SceneEntry['box'] }, at: 0 }
    expect(canPreview(null, null)).toBe(false)
    expect(canPreview(seen, null)).toBe(true)
    expect(canPreview(null, storedPicture(saved))).toBe(true)
  })
})
