import { describe, expect, it } from 'vitest'
import { coverage, narrowedFeatures, setupWarnings, unusedFeatures } from './coverage'
import type { SafetyCamera, SafetyZone } from './types'

const zone = (name: string, over: Partial<SafetyZone>): SafetyZone => ({
  id: name, name, points: [], intrusion: { mode: 'off', classes: ['person'] },
  overcrowding: { enabled: false, threshold: 5, classes: ['person'] }, ...over,
})

describe('what each feature covers on a camera', () => {
  it('PPE, fall and fire cover the whole frame until a zone carries them', () => {
    expect(coverage('fire', [])).toEqual({ kind: 'whole' })
    expect(coverage('fire', [zone('Bay', { ppe: { enabled: true } })])).toEqual({ kind: 'whole' })
    expect(coverage('fire', [zone('Charging bay', { fire: { enabled: true } })]))
      .toEqual({ kind: 'zones', zones: ['Charging bay'] })
  })
  it('proximity and zones do nothing without a zone, and say so', () => {
    expect(coverage('proximity', [])).toEqual({ kind: 'none' })
    expect(coverage('zones', [zone('Dock', { overcrowding: { enabled: true, threshold: 4, classes: ['person'] } })]))
      .toEqual({ kind: 'zones', zones: ['Dock'] })
    expect(coverage('proximity', [zone('Lane', { proximity: { enabled: true } as SafetyZone['proximity'] })]))
      .toEqual({ kind: 'zones', zones: ['Lane'] })
  })
})

describe('switching a section on narrows a whole-frame feature', () => {
  it('names the features that went from the whole frame to some zones', () => {
    const saved = [zone('Bay', {})]
    const draft = [zone('Bay', { fire: { enabled: true } }), zone('Yard', { ppe: { enabled: false } })]
    expect(narrowedFeatures(saved, draft)).toEqual(['fire'])
    expect(narrowedFeatures(draft, draft)).toEqual([])
  })
})

const cam = (id: string, features: SafetyCamera['features']): SafetyCamera => ({ id, name: id, features, streams: [] })

describe('features switched on that cannot raise anything yet', () => {
  it('proximity with no proximity zone, and zones with no intrusion or crowding zone', () => {
    expect(unusedFeatures(['proximity', 'zones', 'ppe'], [])).toEqual(['proximity', 'zones'])
    expect(unusedFeatures(['proximity', 'zones'], [zone('Lane', { proximity: { enabled: true } })])).toEqual(['zones'])
    expect(unusedFeatures(['ppe', 'fire', 'fall'], [])).toEqual([])
  })
  it('across the site: one warning per camera and feature, only once its zones are known', () => {
    const cameras = [cam('bay-a', ['proximity', 'ppe']), cam('office', ['zones']), cam('store', ['fire']),
      cam('yard', ['proximity'])]
    const zones = { 'bay-a': [], office: [zone('Desk', { intrusion: { mode: 'restricted', classes: ['person'] } })],
      store: [] }
    expect(setupWarnings(cameras, zones).map(w => [w.camera.id, w.feature])).toEqual([['bay-a', 'proximity']])
  })
})

