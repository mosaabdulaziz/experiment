import { describe, expect, it } from 'vitest'
import { shownZones } from './zoneDisplay'

const z = (id: string, type: string) => ({ id, name: id, type, points: [[0, 0], [1, 0], [1, 1]] }) as never

describe('shownZones', () => {
  it('drops the switched-off types and keeps unknown ones', () => {
    const out = shownZones([z('a', 'seat'), z('b', 'chair'), z('c', 'door_area')], { seat: false, chair: true })
    expect(out.map((x: { id: string }) => x.id)).toEqual(['b', 'c'])
  })
  it('shows everything without switches', () => {
    expect(shownZones([z('a', 'seat')], undefined)).toHaveLength(1)
  })
})
