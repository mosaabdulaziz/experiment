/** The Report panel's haircut photo links (2026-10-22): a daily report only; disabled off the shop network (or
 * while that is not known yet) -- the photos are of customers. */
import { photoReportUrl, type ReportKind } from '../api/client'

export interface PhotoLinks { show: boolean; enabled: boolean; pdf: string; html: string }

export function photoLinks(kind: ReportKind, start: string, lang: 'en' | 'ar', allowed: boolean | null): PhotoLinks {
  return { show: kind === 'day', enabled: allowed === true,
           pdf: photoReportUrl({ start, lang, format: 'pdf' }), html: photoReportUrl({ start, lang, format: 'html' }) }
}
