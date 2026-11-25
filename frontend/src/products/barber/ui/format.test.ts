import { describe, expect, it } from 'vitest'
import { fmtTimer } from './format'

describe('fmtTimer', () => {
  it('shows mm:ss under an hour', () => {
    expect(fmtTimer(0)).toBe('00:00')
    expect(fmtTimer(125.9)).toBe('02:05')
    expect(fmtTimer(3599)).toBe('59:59')
  })
  it('keeps the seconds from an hour on, so it reads as h:mm:ss and keeps ticking', () => {
    expect(fmtTimer(3600)).toBe('1:00:00')
    expect(fmtTimer(3661)).toBe('1:01:01')
    expect(fmtTimer(5340 + 25)).toBe('1:29:25')
  })
  it('shows a dash when unknown', () => {
    expect(fmtTimer(null)).toBe('—')
    expect(fmtTimer(undefined)).toBe('—')
    expect(fmtTimer(NaN)).toBe('—')
  })
})
