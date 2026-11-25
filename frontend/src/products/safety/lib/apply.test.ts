import { describe, expect, it } from 'vitest'
import { pendingChanges } from './apply'
import type { SafetyCamera } from './types'

const cam = (id: string, features: string[], running: string[]): SafetyCamera => ({
  id, name: id, features: features as SafetyCamera['features'],
  streams: running.map(f => ({ id: `${id}@${f}`, feature: f as SafetyCamera['features'][number], running: true })),
})

describe('what Apply will change', () => {
  it('lists the features to start and to stop, per camera and overall', () => {
    const changes = pendingChanges([cam('bay-a', ['proximity', 'ppe'], ['proximity', 'zones']), cam('store', ['fire'], ['fire'])])
    expect(changes.cameras).toEqual([{ id: 'bay-a', start: ['ppe'], stop: ['zones'] }])
    expect(changes.features).toEqual(['ppe', 'zones'])
  })
  it('is empty when what runs is what is saved', () => {
    expect(pendingChanges([cam('store', ['fire'], ['fire'])])).toEqual({ cameras: [], features: [] })
  })
  it('counts a configured stream that sends nothing as applied (Health reports it; Apply would not fix it)', () => {
    const offline = cam('store', ['fire'], ['fire'])
    offline.streams[0].running = false
    expect(pendingChanges([offline])).toEqual({ cameras: [], features: [] })
  })
})
