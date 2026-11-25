/** The Customers panel's door entries: date and time per row, in the order the API gives (newest first). */
import type { CustomerVisitBrief, CustomerVisitServed } from '../api/types'
import { fmtShortDate, type Lang } from '../i18n'
import { fmtClock } from '../ui/format'

export interface CustomerEntry { id: string; at: string }

export const entryRows = (entries: CustomerEntry[], lang: Lang): { id: string; date: string; time: string }[] =>
  entries.map(e => ({ id: e.id, date: fmtShortDate(lang, e.at), time: fmtClock(e.at) }))

const unique = (xs: string[]) => xs.filter((x, i) => xs.indexOf(x) === i)

/** 2026-10-18 (user): the panel's visits, newest first -- date, time and the chairs he was served at, by the shop's
 * names (a resumed haircut can move to a second chair; none: not served). */
export const visitRows = (visits: CustomerVisitBrief[] | undefined, lang: Lang, chairName: (id: string) => string):
    { id: string; date: string; time: string; chairs: string[] }[] =>
  [...(visits ?? [])].sort((a, b) => b.entered_at.localeCompare(a.entered_at)).map(v => ({
    id: v.id, date: fmtShortDate(lang, v.entered_at), time: fmtClock(v.entered_at),
    chairs: unique(v.haircut_sessions.map(s => chairName(s.chair_id))),
  }))

/** The card's chairs, one per served visit, newest first ("R2 + L7" for a haircut that moved). */
export const servedChairs = (served: CustomerVisitServed[] | undefined, chairName: (id: string) => string): string[] =>
  (served ?? []).filter(v => v.chairs.length).map(v => unique(v.chairs.map(chairName)).join(' + '))
