/** The Reports page's helpers (2026-10-25; spec 2026-10-25-barber-saved-reports-design.md section 3.2): a saved
 * report's period and time labels, and its links per report and language. */
import { savedReportUrl, type ReportFormat, type SavedReport } from '../api/client'
import { LOCALE, type Lang } from '../i18n'

export type LinkKind = 'view' | 'pdf' | 'html' | 'docx'
export interface Line { report: 'analysis' | 'photos'; langs: { lang: Lang; links: { kind: LinkKind; href: string }[] }[] }

const fmt = (lang: Lang, o: Intl.DateTimeFormatOptions, d: Date) =>
  new Intl.DateTimeFormat(LOCALE[lang], { numberingSystem: 'latn', ...o }).format(d)
const noon = (iso: string) => new Date(`${iso.slice(0, 10)}T12:00:00`)
const FULL: Intl.DateTimeFormatOptions = { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' }

/** "Sat, 24 Oct 2026"; a week "Sat 17 – Fri, 23 Oct 2026"; a month "October 2026". */
export function periodLabel(lang: Lang, r: Pick<SavedReport, 'kind' | 'first' | 'last'>): string {
  if (r.kind === 'month') return fmt(lang, { month: 'long', year: 'numeric' }, noon(r.first))
  if (r.kind === 'day') return fmt(lang, FULL, noon(r.first))
  const head: Intl.DateTimeFormatOptions = r.first.slice(0, 7) === r.last.slice(0, 7)
    ? { weekday: 'short', day: 'numeric' } : { weekday: 'short', day: 'numeric', month: 'short' }
  return `${fmt(lang, head, noon(r.first))} – ${fmt(lang, FULL, noon(r.last))}`
}

/** "25 Oct, 05:00": when a folder was saved or a lap ended (the service's clock). */
export function savedTime(lang: Lang, iso: string): string {
  return fmt(lang, { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit', hourCycle: 'h23' }, new Date(iso))
}

/** Per report (analysis, then photos) and language (en, then ar): View (the HTML in a tab), PDF, HTML (as a file),
 * Word -- only for the files the folder holds. */
export function reportLines(r: SavedReport): Line[] {
  const lines: Line[] = []
  for (const report of ['analysis', 'photos'] as const) {
    const langs = (['en', 'ar'] as const).map(lang => {
      const file = (format: ReportFormat) => r.files.find(f => f.report === report && f.lang === lang && f.format === format)
      const html = file('html'), pdf = file('pdf'), docx = file('docx')
      const links: Line['langs'][number]['links'] = []
      if (html) links.push({ kind: 'view', href: savedReportUrl(r.id, html.name) })
      if (pdf) links.push({ kind: 'pdf', href: savedReportUrl(r.id, pdf.name) })
      if (html) links.push({ kind: 'html', href: savedReportUrl(r.id, html.name, true) })
      if (docx) links.push({ kind: 'docx', href: savedReportUrl(r.id, docx.name) })
      return { lang, links }
    }).filter(l => l.links.length > 0)
    if (langs.length) lines.push({ report, langs })
  }
  return lines
}
