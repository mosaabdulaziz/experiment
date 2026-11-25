/** The Customers register's pure bits. */
import { fmtNumber, translate, type Key, type Lang } from '../i18n'

export const CUSTOMER_PAGE = 24
export const MAX_FACES = 5
export const MAX_BODIES = 3

/** Retention in whole months, as the privacy line says it (365 days -> 12). */
export const retentionMonths = (days: number): number => Math.round(days / 30.4)

/** Which "{n} visits" text fits n (Arabic: one, two, 3-10, 11+; English
 * singular only for 1 -- its other three keys read the same). */
export function visitCountKey(n: number): Key {
  if (n === 1) return 'cu.visits1'
  if (n === 2) return 'cu.visits2'
  if (n >= 3 && n <= 10) return 'cu.visitsFew'
  return 'cu.visitsMany'
}

/** "1 visit" / "12 visits" / "زيارة واحدة" / "زيارتان" / "5 زيارات" / "12 زيارة", Western digits. */
export const visitCount = (lang: Lang, n: number): string =>
  translate(lang, visitCountKey(n), { n: fmtNumber(lang, n) })
