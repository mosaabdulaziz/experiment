import { describe, expect, it } from 'vitest'
import type { IrisEvent } from '@/types/event'
import { matchesIncidentFilters, NO_FILTERS } from './incidentFilters'

const alert = { id: 1, camera: 'cam-1', event_type: 'alert', tags: ['proximity'], details: {} } as unknown as IrisEvent
const held = { id: 2, camera: 'cam-1', event_type: 'alert', tags: ['proximity', 'depth-suppressed'],
  details: { suppressed: 'depth' } } as unknown as IrisEvent

describe('the Incidents page filters (Ticket 6: pairs the depth check held back)', () => {
  it('hides the held-back ones unless asked for', () => {
    expect([alert, held].filter(e => matchesIncidentFilters(e, NO_FILTERS, 0)).map(e => e.id)).toEqual([1])
  })
  it('shows only the held-back ones when asked for, the other filters still applying', () => {
    const f = { ...NO_FILTERS, heldBack: true }
    expect([alert, held].filter(e => matchesIncidentFilters(e, f, 0)).map(e => e.id)).toEqual([2])
    expect(matchesIncidentFilters(held, { ...f, cameras: ['cam-2'] }, 0)).toBe(false)
  })
})
