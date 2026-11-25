import { describe, expect, it } from 'vitest'
import { makeDraftZone, zoneInvalidReason, zoneSummary, type DraftZone } from './zoneDraft'

const ppeZone = (items?: string[]): DraftZone => ({
  ...makeDraftZone('Bay'), points: [{ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 1, y: 1 }], ringClosed: true, closed: true,
  intrusion: { mode: 'off', classes: ['person'] }, ppe: { enabled: true, ...(items ? { items } : {}) },
})

describe('a PPE zone and the equipment it checks', () => {
  it('cannot be saved checking nothing', () => {
    expect(zoneInvalidReason(ppeZone([]))).toBe('Pick the equipment PPE checks in Bay')
    expect(zoneInvalidReason(ppeZone(['helmet']))).toBeNull()
    expect(zoneInvalidReason(ppeZone())).toBeNull()                // no list: everything the PPE module checks
  })
  it('says what it checks in the zone list', () => {
    expect(zoneSummary(ppeZone(['helmet']))).toBe('PPE: hard hat')
    expect(zoneSummary(ppeZone(['helmet', 'vest']))).toBe('PPE: hard hat + vest')
    expect(zoneSummary(ppeZone())).toBe('PPE')
  })
})
