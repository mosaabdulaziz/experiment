import { describe, expect, it } from 'vitest'
import { reportUrl } from '../api/client'

describe('reportUrl', () => {
  it('builds the report endpoint URL', () => {
    expect(reportUrl({ kind: 'week', start: '2026-10-11', lang: 'ar', format: 'pdf' }))
      .toBe('/barber/api/barber/report?kind=week&start=2026-10-11&lang=ar&format=pdf')
  })
  it('turns a month picker value into its first day', () => {
    expect(reportUrl({ kind: 'month', start: '2026-10', lang: 'en', format: 'docx' }))
      .toBe('/barber/api/barber/report?kind=month&start=2026-10-01&lang=en&format=docx')
  })
})

import { reportStart } from '../api/client'

describe('reportStart', () => {
  it("follows the page's day until the user picks one", () => {
    expect(reportStart('day', null, '2026-10-15')).toBe('2026-10-15')
    expect(reportStart('month', null, '2026-10-15')).toBe('2026-10')
    expect(reportStart('week', '2026-10-11', '2026-10-15')).toBe('2026-10-11')
    expect(reportStart('day', '2026-09', '2026-10-15')).toBe('2026-09-01')
    expect(reportStart('month', '2026-09-20', '2026-10-15')).toBe('2026-09')
  })
})

describe('reportUrl download', () => {
  it('asks for the HTML as a file', () => {
    expect(reportUrl({ kind: 'day', start: '2026-10-15', lang: 'en', format: 'html', download: true }))
      .toBe('/barber/api/barber/report?kind=day&start=2026-10-15&lang=en&format=html&download=1')
  })
})
