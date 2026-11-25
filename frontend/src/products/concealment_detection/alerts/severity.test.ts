import { describe, expect, it } from 'vitest'
import { channels, effectiveSeverity } from './severity'

describe('severity -> alert channels', () => {
  it('surfaces unreviewed critical and warning candidates in toast, alarm card, and rail badge', () => {
    expect(channels({ severity: 'critical', status: 'candidate' })).toEqual({
      toast: true,
      card: true,
      badge: true,
    })
    expect(channels({ severity: 'warning', status: 'candidate' })).toEqual({
      toast: true,
      card: true,
      badge: true,
    })
  })

  it('surfaces unreviewed info candidates in toast and badge only (no alarm card)', () => {
    expect(channels({ severity: 'info', status: 'candidate' })).toEqual({
      toast: true,
      card: false,
      badge: true,
    })
  })

  it('suppresses alert channels once a candidate has been reviewed', () => {
    expect(channels({ severity: 'critical', status: 'acknowledged' })).toEqual({
      toast: false,
      card: false,
      badge: false,
    })
    expect(channels({ severity: 'warning', status: 'acknowledged' })).toEqual({
      toast: false,
      card: false,
      badge: false,
    })
    expect(channels({ severity: 'warning', status: 'concern_noted' })).toEqual({
      toast: false,
      card: false,
      badge: false,
    })
    expect(channels({ severity: 'warning', status: 'dismissed' })).toEqual({
      toast: false,
      card: false,
      badge: false,
    })
  })

  it('maps high tier candidates to critical UI severity', () => {
    expect(effectiveSeverity({ severity: 'critical', tier: 'high' })).toBe('critical')
    expect(effectiveSeverity({ severity: 'warning', tier: 'high' })).toBe('critical')
    expect(effectiveSeverity({ severity: 'warning', tier: 'medium' })).toBe('warning')
    expect(effectiveSeverity({ severity: 'info', tier: 'low' })).toBe('info')
  })
})
