import { snapshotUrl } from '../api/client'
import type { Alert } from '../api/types'
import type { Key } from '../i18n'
import { elapsedSince, secondsBetween } from '../ui/format'

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

export interface CropView { widthPct: number; leftPct: number; topPct: number }

/** Where to place the full frame, as an <img> (width in % of the card, height auto) inside an overflow-hidden
 * card of aspect `cardAspect`, so the subject box -- grown by `grow` of its size on each side -- fills the card
 * with uniform scaling: the grown region is widened in its short dimension to the card's aspect, kept centred on
 * the subject and clamped into the frame (showing as much as fits). left/top are negative % offsets of the card. */
export function cropView(box: [number, number, number, number], frameW: number, frameH: number, cardAspect = 4 / 3, grow = 0.3): CropView {
  if (!(frameW > 0) || !(frameH > 0)) return { widthPct: 100, leftPct: 0, topPct: 0 }
  const [x, y] = box
  const w = Math.max(box[2], 0.02), h = Math.max(box[3], 0.02)
  let pw = w * (1 + 2 * grow) * frameW
  let ph = h * (1 + 2 * grow) * frameH
  if (pw / ph < cardAspect) pw = ph * cardAspect
  else ph = pw / cardAspect
  if (pw > frameW) { pw = frameW; ph = pw / cardAspect }
  if (ph > frameH) { ph = frameH; pw = ph * cardAspect }
  const rw = pw / frameW, rh = ph / frameH
  const left = clamp(x + box[2] / 2 - rw / 2, 0, 1 - rw)
  const top = clamp(y + box[3] / 2 - rh / 2, 0, 1 - rh)
  return { widthPct: 100 / rw, leftPct: -(left / rw) * 100, topPct: -(top / rh) * 100 }
}

export interface AlertCardModel {
  id: string
  kindKey: Key
  open: boolean
  startedAt: string
  durationS: number | null
  chair: string | null
  barber: string | null
  ticket: number | null
  imageUrl: string | null
  box: [number, number, number, number] | null
}

export function alertCard(a: Alert, now: number, names: Record<string, string>, chairName: (id: string) => string | null): AlertCardModel {
  return {
    id: a.id,
    kindKey: a.kind === 'excessive_service' ? 'alert.excessive_service' : 'alert.excessive_wait',
    open: a.open,
    startedAt: a.started_at,
    durationS: a.open ? elapsedSince(a.started_at, now) : secondsBetween(a.started_at, a.ended_at),
    chair: a.chair_id ? chairName(a.chair_id) ?? a.chair_id : null,
    barber: a.barber_id ? names[a.barber_id] ?? null : null,
    ticket: a.ticket ?? null,
    imageUrl: a.has_snapshot ? snapshotUrl(a.id) : null,
    box: a.subject_box ?? null,
  }
}

/** The Alerts page refetches when this changes: an alert opened or cleared, or an open alert's picture got
 * saved (the live heartbeat carries has_snapshot). */
export function openAlertsKey(open: Pick<Alert, 'id' | 'has_snapshot'>[] | undefined): string {
  return (open ?? []).map(a => a.id + (a.has_snapshot ? '*' : '')).join(',')
}

/** The warnings page's order (2026-10-18, user): every open alert first, newest first; then the closed ones under a
 * heading per shop day (a day starts at dayStartHour; e.g. 02:00 is the previous day's), newest day first and newest
 * first within it. */
export type AlertSection = { kind: 'open'; cards: AlertCardModel[] } | { kind: 'day'; day: string; cards: AlertCardModel[] }

const newestFirst = (a: AlertCardModel, b: AlertCardModel) => b.startedAt.localeCompare(a.startedAt)

/** The shop day of a local ISO time: its date once dayStartHour is taken off. */
function shopDayOf(iso: string, dayStartHour: number): string {
  const d = new Date(iso)
  d.setHours(d.getHours() - dayStartHour)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`
}

export function alertSections(cards: AlertCardModel[], dayStartHour: number): AlertSection[] {
  const out: AlertSection[] = []
  const open = cards.filter(c => c.open).sort(newestFirst)
  if (open.length) out.push({ kind: 'open', cards: open })
  const days = new Map<string, AlertCardModel[]>()
  for (const c of cards.filter(c => !c.open).sort(newestFirst)) {
    const day = shopDayOf(c.startedAt, dayStartHour)
    days.set(day, [...(days.get(day) ?? []), c])
  }
  for (const day of [...days.keys()].sort().reverse()) out.push({ kind: 'day', day, cards: days.get(day)! })
  return out
}

/** One page of size cards over the sections in order; a section the page cuts keeps its heading on both pages. */
export function pageSections(sections: AlertSection[], page: number, size: number): AlertSection[] {
  const lo = page * size, hi = lo + size
  const out: AlertSection[] = []
  let at = 0
  for (const s of sections) {
    const cards = s.cards.slice(Math.max(0, lo - at), Math.max(0, hi - at))
    if (cards.length) out.push({ ...s, cards })
    at += s.cards.length
  }
  return out
}
