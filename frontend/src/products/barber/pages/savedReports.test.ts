import { describe, expect, it } from 'vitest'
import { savedReportUrl, type SavedReport } from '../api/client'
import { periodLabel, reportLines, savedTime } from './savedReports'

const ARABIC_INDIC = /[٠-٩۰-۹]/
const day: SavedReport = {
  id: 'day/2026-10-24', kind: 'day', first: '2026-10-24', last: '2026-10-24', lap_ended: null, label: null,
  saved_at: '2026-10-25T05:00:41',
  files: [
    { report: 'analysis', lang: 'en', format: 'pdf', name: '24oct_a_en.pdf' },
    { report: 'analysis', lang: 'en', format: 'html', name: '24oct_a_en.html' },
    { report: 'analysis', lang: 'ar', format: 'pdf', name: '24oct_a_ar.pdf' },
    { report: 'photos', lang: 'ar', format: 'html', name: '24oct_p_ar.html' },
  ],
}

describe('savedReportUrl', () => {
  it('names the folder and the file', () => {
    expect(savedReportUrl('day/2026-10-16/lap-2026-10-22_1313', '16oct_x_en.pdf'))
      .toBe('/barber/api/barber/reports/saved/file?id=day%2F2026-10-16%2Flap-2026-10-22_1313&name=16oct_x_en.pdf')
    expect(savedReportUrl('day/2026-10-24', 'a.html', true))
      .toBe('/barber/api/barber/reports/saved/file?id=day%2F2026-10-24&name=a.html&download=1')
  })
})

describe('periodLabel', () => {
  it('labels a day, a week (across a month too) and a month', () => {
    expect(periodLabel('en', day)).toMatch(/^Sat,? 24 Oct 2026$/)
    expect(periodLabel('en', { kind: 'week', first: '2026-10-17', last: '2026-10-23' })).toMatch(/^Sat 17 – Fri,? 23 Oct 2026$/)
    expect(periodLabel('en', { kind: 'week', first: '2026-10-31', last: '2026-11-06' })).toMatch(/^Sat,? 31 Oct – Fri,? 6 Nov 2026$/)
    expect(periodLabel('en', { kind: 'month', first: '2026-10-01', last: '2026-10-31' })).toBe('October 2026')
    expect(periodLabel('ar', { kind: 'month', first: '2026-10-01', last: '2026-10-31' })).toBe('أكتوبر 2026')
    expect(periodLabel('ar', day)).not.toMatch(ARABIC_INDIC)
  })
})

describe('savedTime', () => {
  it('gives the day and the 24-hour time in Western digits', () => {
    expect(savedTime('en', '2026-10-25T05:00:41')).toMatch(/^25 Oct,? 05:00$/)
    expect(savedTime('ar', '2026-10-25T05:00:41')).not.toMatch(ARABIC_INDIC)
  })
})

describe('reportLines', () => {
  it('gives each report and language its links, only for the files saved', () => {
    const lines = reportLines(day)
    expect(lines.map(l => l.report)).toEqual(['analysis', 'photos'])
    expect(lines[0].langs.map(l => [l.lang, l.links.map(k => k.kind)])).toEqual([['en', ['view', 'pdf', 'html']], ['ar', ['pdf']]])
    expect(lines[1].langs.map(l => [l.lang, l.links.map(k => k.kind)])).toEqual([['ar', ['view', 'html']]])
    expect(lines[0].langs[0].links[0].href).toBe(savedReportUrl('day/2026-10-24', '24oct_a_en.html'))
    expect(lines[0].langs[0].links[2].href).toBe(savedReportUrl('day/2026-10-24', '24oct_a_en.html', true))
  })
  it('a week or month has the analysis line only, Word too when saved', () => {
    const week: SavedReport = { ...day, id: 'week/2026-10-17', kind: 'week', first: '2026-10-17', last: '2026-10-23',
      files: [{ report: 'analysis', lang: 'en', format: 'docx', name: 'w_en.docx' }] }
    expect(reportLines(week)).toEqual([{ report: 'analysis', langs: [{ lang: 'en', links: [
      { kind: 'docx', href: savedReportUrl('week/2026-10-17', 'w_en.docx') }] }] }])
  })
})
