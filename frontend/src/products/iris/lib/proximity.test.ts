import { describe, expect, it } from 'vitest'
import type { IrisEvent } from '@/types/event'
import { forLiveView, isDepthSuppressed, proximityFacts } from './proximity'

const ev = (details: Record<string, string>) => ({ id: 9, details }) as unknown as IrisEvent

describe('proximityFacts', () => {
  it('reads the depth check’s gap, threshold and image', () => {
    const f = proximityFacts(ev({ depth_gap: '0.050', depth_max_gap: '0.20', depth_image: '1', confirmation: 'depth' }))
    expect([f.depthGap, f.depthMaxGap, f.depthImage]).toEqual([0.05, 0.2, true])
  })
  it('has none of them for an event without the depth check', () => {
    const f = proximityFacts(ev({ confirmation: 'image' }))
    expect([f.depthGap, f.depthMaxGap, f.depthImage]).toEqual([null, null, false])
  })
})

// Ticket 6: a pair the depth check held back is a quiet event: info, tagged, with what it would have been
describe('a pair the depth check held back', () => {
  const held = { id: 4, tags: ['proximity', 'depth-suppressed'], severity: 'info',
    details: { suppressed: 'depth', would_be: 'critical', depth_gap: '0.500', depth_max_gap: '0.25' } } as unknown as IrisEvent
  it('reads as held back, with the severity it would have had and the reading against the cut-off', () => {
    const f = proximityFacts(held)
    expect([f.heldBack, f.wouldBe, f.depthGap, f.depthMaxGap]).toEqual([true, 'critical', 0.5, 0.25])
  })
  it('is told apart from an alert by its tag or its details', () => {
    expect(isDepthSuppressed(held)).toBe(true)
    expect(isDepthSuppressed({ ...held, tags: ['proximity'] } as IrisEvent)).toBe(true)
    expect(isDepthSuppressed({ id: 5, tags: ['proximity'], details: {} } as unknown as IrisEvent)).toBe(false)
  })
  it('never reaches the live view: the history the WebSocket sends on connect drops it', () => {
    const alert = { id: 5, event_type: 'alert', tags: ['proximity'], details: {} } as unknown as IrisEvent
    expect(forLiveView([{ ...held, event_type: 'alert' } as IrisEvent, alert]).map(e => e.id)).toEqual([5])
  })
})
