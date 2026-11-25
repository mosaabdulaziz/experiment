import { describe, expect, it } from 'vitest'
import { en } from './en'
import { ar } from './ar'
import { fmtLongDate, fmtNumber, translate } from './index'

const ARABIC_INDIC = /[٠-٩۰-۹]/

describe('i18n', () => {
  it('has an Arabic text for every English key and nothing extra', () => {
    expect(Object.keys(ar).sort()).toEqual(Object.keys(en).sort())
    for (const k of Object.keys(ar) as (keyof typeof ar)[]) expect(ar[k].trim()).not.toBe('')
  })
  it('fills variables', () => {
    expect(translate('en', 'staff.moved', { name: 'Ahmed', chair: 'R1' })).toBe('Moved Ahmed from R1')
  })
  it('formats with Western digits', () => {
    expect(fmtNumber('ar', 1234.5, 1)).not.toMatch(ARABIC_INDIC)
    expect(fmtLongDate('ar', '2026-09-25')).not.toMatch(ARABIC_INDIC)
    expect(fmtLongDate('ar', '2026-09-25')).toContain('25')
    expect(fmtLongDate('en', '2026-09-25')).toMatch(/^Friday,? 25 September 2026$/)
  })
})
