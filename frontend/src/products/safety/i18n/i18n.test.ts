import { describe, expect, it } from 'vitest'
import { ar } from './ar'
import { en } from './en'
import { formatNumber, formatTime, translate } from './index'

describe('translations', () => {
  it('has every English key in Arabic', () => {
    expect(Object.keys(ar).sort()).toEqual(Object.keys(en).sort())
  })
  it('fills in values', () => {
    expect(translate('en', 'banner.count', { n: 3 })).toContain('3')
  })
})

describe('numbers and times in Arabic', () => {
  it('writes the numbers it fills in with Arabic-Indic digits, and leaves English ones as they are', () => {
    expect(translate('ar', 'banner.count', { n: 218 })).toContain('٢١٨')
    expect(translate('en', 'banner.count', { n: 218 })).toContain('218')
    expect(translate('ar', 'health.fps', { n: 4.9 })).toContain('٤٫٩')
    expect(formatNumber('ar', 0.25, 2)).toBe('٠٫٢٥')
  })
  it('formats times with Arabic-Indic digits', () => {
    expect(formatTime('ar', '2026-11-09T18:20:00Z')).toMatch(/^[^0-9]*$/)
    expect(formatTime('ar', '2026-11-09T18:20:00Z')).toMatch(/[٠-٩]/)
    expect(formatTime('en', '2026-11-09T18:20:00Z')).toMatch(/[0-9]/)
  })
})
