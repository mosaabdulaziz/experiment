import { describe, expect, it } from 'vitest'
import type { Zone, ZonesByCam } from '../../api/types'
import { camerasToRevert, revertLinked } from './revertScope'
import { deleteZone, setOwner } from './zoneModel'
import { zoneProblems } from '../../floor/areas'

const chair = (id: string, extra: Partial<Zone> = {}): Zone => ({ id, name: id, type: 'chair', points: [[0, 0], [1, 0], [1, 1]], ...extra })
const CAMS = ['barber-c', 'barber-l', 'barber-r']
const SAVED: ZonesByCam = {
  'barber-c': [chair('c-L1', { pair: 'L1' }), chair('c-R1', { pair: 'R1' })],
  'barber-l': [chair('l-R1', { pair: 'R1', owner: true })],
  'barber-r': [chair('r-L1', { pair: 'L1', owner: true })],
}

describe('camerasToRevert', () => {
  it('is just this camera when nothing else changed', () => {
    const draft = { ...SAVED, 'barber-c': [...SAVED['barber-c'], chair('c-new')] }
    expect(camerasToRevert(draft, SAVED, 'barber-c', CAMS)).toEqual(['barber-c'])
  })
  it('takes along the other camera an owner move touched', () => {
    const draft = setOwner(SAVED, 'barber-c', 'c-L1', true)          // clears r-L1's owner
    expect(camerasToRevert(draft, SAVED, 'barber-c', CAMS)).toEqual(['barber-c', 'barber-r'])
    expect(camerasToRevert(draft, SAVED, 'barber-r', CAMS)).toEqual(['barber-r', 'barber-c'])
  })
  it('leaves another camera whose changes are on other areas', () => {
    let draft = setOwner(SAVED, 'barber-c', 'c-L1', true)
    draft = { ...draft, 'barber-l': [...draft['barber-l'], chair('l-x')] }
    expect(camerasToRevert(draft, SAVED, 'barber-c', CAMS)).toEqual(['barber-c', 'barber-r'])
  })
  it('follows the old pair id of a zone moved to another area', () => {
    const draft: ZonesByCam = {
      ...SAVED,
      'barber-c': [chair('c-L1', { pair: 'L9' }), SAVED['barber-c'][1]],
      'barber-r': [chair('r-L1', { pair: 'L1' })],                     // owner cleared
    }
    expect(camerasToRevert(draft, SAVED, 'barber-c', CAMS)).toEqual(['barber-c', 'barber-r'])
  })
  it('ignores cameras that are not configured', () => {
    const saved = { ...SAVED, 'barber-1': [chair('x', { pair: 'L1', owner: true })] }
    const draft = { ...setOwner(saved, 'barber-c', 'c-L1', true) }
    expect(camerasToRevert(draft, saved, 'barber-c', CAMS)).toEqual(['barber-c', 'barber-r'])
  })
})

describe('revertLinked', () => {
  it('reverts this camera fully and only the linked area on the other camera', () => {
    let draft = setOwner(SAVED, 'barber-c', 'c-L1', true)                 // clears r-L1's owner
    draft = { ...draft, 'barber-c': [...draft['barber-c'], chair('c-new')] }
    // unrelated unsaved edits on the linked camera: a rename and a new zone on another area
    draft = { ...draft, 'barber-r': [...draft['barber-r'], chair('r-R9', { pair: 'R9' }), chair('r-x')] }
    draft = { ...draft, 'barber-r': draft['barber-r'].map(z => (z.id === 'r-L1' ? { ...z, name: 'Front' } : z)) }
    const next = revertLinked(draft, SAVED, 'barber-c', CAMS)
    expect(next['barber-c']).toEqual(SAVED['barber-c'])
    // r-L1 is back as saved (owner restored, so its rename in the same area goes too); the rest survives
    expect(next['barber-r']).toEqual([SAVED['barber-r'][0], chair('r-R9', { pair: 'R9' }), chair('r-x')])
    expect(next['barber-l']).toBe(draft['barber-l'])
  })
  it('keeps an unrelated rename on the linked camera', () => {
    const saved: ZonesByCam = { ...SAVED, 'barber-r': [...SAVED['barber-r'], chair('r-L2', { pair: 'L2', owner: true })] }
    let draft = setOwner(saved, 'barber-c', 'c-L1', true)
    draft = { ...draft, 'barber-r': draft['barber-r'].map(z => (z.id === 'r-L2' ? { ...z, name: 'Back' } : z)) }
    const next = revertLinked(draft, saved, 'barber-c', CAMS)
    expect(next['barber-r']).toEqual([saved['barber-r'][0], { ...saved['barber-r'][1], name: 'Back' }])
  })
  it('puts back a deleted linked zone in its saved place and drops an added one', () => {
    const saved: ZonesByCam = { 'barber-c': [chair('c-L1', { pair: 'L1' })], 'barber-r': [chair('r-a'), chair('r-L1', { pair: 'L1', owner: true }), chair('r-b')] }
    const draft: ZonesByCam = {
      'barber-c': [chair('c-L1', { pair: 'L1', owner: true })],
      'barber-r': [chair('r-a'), chair('r-b', { name: 'B' }), chair('r-L1b', { pair: 'L1' })],
    }
    const next = revertLinked(draft, saved, 'barber-c', CAMS)
    expect(next['barber-r']).toEqual([chair('r-a'), saved['barber-r'][1], chair('r-b', { name: 'B' })])
  })
})

describe('revertLinked with seats', () => {
  const seat = (id: string, of: string): Zone => ({ id, name: id, type: 'seat', points: [[0, 0], [1, 0], [1, 1]], seat_of: of })
  const saved: ZonesByCam = {
    'barber-c': [chair('c-L1', { pair: 'L1' }), chair('c-L2', { pair: 'L2' })],
    'barber-r': [chair('r-L1', { pair: 'L1', owner: true }), seat('r-L1-seat', 'r-L1'),
      chair('r-L2', { pair: 'L2', owner: true }), seat('r-L2-seat', 'r-L2')],
  }
  const cams = ['barber-c', 'barber-r']
  // the current camera's own change on L1 is what links barber-r
  const withCentreEdit = (d: ZonesByCam): ZonesByCam => ({ ...d, 'barber-c': [chair('c-L1', { pair: 'L1', name: 'x' }), saved['barber-c'][1]] })

  it('restores a linked chair together with the seat its delete took', () => {
    const draft = withCentreEdit({ ...saved, 'barber-r': deleteZone(saved['barber-r'], 'r-L1') })
    const next = revertLinked(draft, saved, 'barber-c', cams)
    expect(next['barber-r']).toEqual(saved['barber-r'])
    expect(zoneProblems(next, cams)).toEqual(zoneProblems(saved, cams))
  })
  it('drops a redrawn chair and puts its re-pointed seat back on the old chair', () => {
    const redrawn = deleteZone(saved['barber-r'], 'r-L1').filter(z => z.id !== 'r-L1-seat')
    const draft = withCentreEdit({ ...saved, 'barber-r': [chair('r-L1b', { pair: 'L1', owner: true }), seat('r-L1-seat', 'r-L1b'), ...redrawn] })
    const next = revertLinked(draft, saved, 'barber-c', cams)
    expect(next['barber-r']).toEqual(saved['barber-r'])
    expect(zoneProblems(next, cams)).toEqual(zoneProblems(saved, cams))
  })
  it('leaves a seat of an out-of-scope chair as drafted', () => {
    const draft = withCentreEdit({ ...saved, 'barber-r': saved['barber-r'].map(z => (z.id === 'r-L2-seat' ? { ...z, name: 'moved' } : z.id === 'r-L1' ? { ...z, name: 'y' } : z)) })
    const next = revertLinked(draft, saved, 'barber-c', cams)
    expect(next['barber-r']).toEqual(saved['barber-r'].map(z => (z.id === 'r-L2-seat' ? { ...z, name: 'moved' } : z)))
  })
})
