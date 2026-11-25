import type { ChairOcc, ResumeDecision } from '../api/types'
import type { GroupKey } from '../floor/areas'
import type { Board, ChairCard } from './selectors'

/** A chair as the Cameras tab's status strip shows it. From the board (an
 * older backend): busy (a haircut), washing (counts as busy), or free. From
 * the chair engine's states when it sends them: starting (sitting down),
 * busy (sitting: the haircut), checking (sitting, re-ID still deciding: the
 * previous customer back or a new one), away (stepped out, until resumeUntil),
 * free (empty), or no_view (the owning camera can't see the chair). */
export type ChipKind = 'busy' | 'washing' | 'free' | 'starting' | 'checking' | 'away' | 'no_view'
export interface ChairChip {
  area: string
  name: string
  barber: ChairCard['barber']
  kind: ChipKind
  ticket: number | null
  since: string | null
  /** an ongoing haircut's start, shown on its card (2026-10-18); for away it is not `since` (the away start) */
  startedAt: string | null
  /** away only: the haircut resumes if the customer is back by then */
  resumeUntil: string | null
  /** busy only: how the re-ID check decided this sitting (resumed / new customer, the score), for RESUME_SHOW_MS */
  decision: ResumeDecision | null
  /** a pedicure chair (2026-11-10): no barber line, no haircut target on its timer */
  pedicure: boolean
}
/** busy = starting / sitting / checking / away (and an older backend's washing);
 * free = empty; a chair with no view is neither, counted in noView. */
export interface ChipGroup { side: GroupKey; chips: ChairChip[]; busy: number; free: number; noView: number }
export interface ChairStripModel { groups: ChipGroup[]; total: number; busy: number; free: number; noView: number }

const STATE_KIND = new Map<string, ChipKind>([['empty', 'free'], ['starting', 'starting'], ['sitting', 'busy'], ['away', 'away'], ['no_view', 'no_view']])

/** How long a chair must stay "starting" before the dashboard shows it. */
export const MIN_STARTING_MS = 3000
/** How long a card says how the re-ID check decided (2026-10-18). */
export const RESUME_SHOW_MS = 5 * 60_000

/** The check's answer while it is fresh (no clock: always). */
export function freshDecision(occ: ChairOcc | undefined, nowMs?: number): ResumeDecision | null {
  const d = occ?.resume
  if (!d) return null
  const at = Date.parse(d.at)
  return nowMs == null || !Number.isFinite(at) || nowMs - at < RESUME_SHOW_MS ? d : null
}

/** The chair's state as shown: "starting" flickers in the engine, so it shows
 * only after `since` is at least minStartingMs old (against the server-aligned
 * now); before that (or with no usable `since`) the chair shows as empty. The
 * other states pass through. Without nowMs, nothing is debounced. */
export function effectiveState(occ: ChairOcc | undefined, nowMs?: number, minStartingMs = MIN_STARTING_MS): ChairOcc['state'] {
  const st = occ?.state
  if (st !== 'starting' || nowMs == null) return st
  const since = occ!.since ? Date.parse(occ!.since) : NaN
  return Number.isFinite(since) && nowMs - since >= minStartingMs ? 'starting' : 'empty'
}

const chip = (c: ChairCard, occ: ChairOcc | undefined, nowMs?: number): ChairChip => {
  const cut = (ch: Omit<ChairChip, 'startedAt' | 'decision'>): ChairChip => ({
    ...ch,
    startedAt: ch.kind === 'busy' || (ch.kind === 'washing' && !occ?.haircut_since) ? ch.since
      : ch.kind === 'away' || ch.kind === 'washing' ? occ?.haircut_since ?? null : null,
    decision: ch.kind === 'busy' ? freshDecision(occ, nowMs) : null,
  })
  const s = c.status
  const base = { area: c.area, name: c.name, barber: c.barber, resumeUntil: null, pedicure: occ?.kind === 'pedicure' }
  const ticket = s.kind === 'free' ? null : s.ticket
  const state = effectiveState(occ, nowMs)
  const kind = state ? STATE_KIND.get(state) : undefined
  if (kind) {
    // the haircut timer runs from the board's haircut start (as before), else from the state's start
    if (kind === 'busy' && occ!.checking) return cut({ ...base, kind: 'checking', ticket: occ!.check_ticket ?? null, since: occ!.checking })
    // a pedicure chair (2026-11-10): its timer runs from the session's start, like a haircut's
    if (kind === 'busy') return cut({ ...base, kind, ticket,
      since: s.kind === 'haircut' ? s.since : occ!.kind === 'pedicure' ? occ!.haircut_since ?? occ!.since ?? null : occ!.since ?? null })
    if (kind === 'away') return cut({ ...base, kind, ticket, since: occ!.since ?? null, resumeUntil: occ!.resume_until ?? null })
    if (kind === 'free') return cut({ ...base, kind, ticket: null, since: null })
    return cut({ ...base, kind, ticket, since: occ!.since ?? null })
  }
  if (s.kind === 'free') return cut({ ...base, kind: 'free', ticket: null, since: null })
  return cut({ ...base, kind: s.kind === 'washing' ? 'washing' : 'busy', ticket: s.ticket, since: s.since })
}

const group = (side: ChipGroup['side'], cards: ChairCard[], chairs: Record<string, ChairOcc>, nowMs?: number): ChipGroup => {
  const chips = cards.map(c => chip(c, Object.prototype.hasOwnProperty.call(chairs, c.area) ? chairs[c.area] : undefined, nowMs))
  const free = chips.filter(c => c.kind === 'free').length
  const noView = chips.filter(c => c.kind === 'no_view').length
  return { side, chips, busy: chips.length - free - noView, free, noView }
}

/** The Shop board's chair groups, in the same (physical) order, as strip
 * chips; each chair's engine state (`chairs`, pushed with the scene) wins
 * over the board's customer-derived status when it has one. */
export function chairStrip(board: Pick<Board, 'groups'>, chairs?: Record<string, ChairOcc> | null, nowMs?: number): ChairStripModel {
  const occ = chairs ?? {}
  const groups = board.groups.map(g => group(g.key, g.cards, occ, nowMs))
  const sum = (f: (g: ChipGroup) => number) => groups.reduce((n, g) => n + f(g), 0)
  const busy = sum(g => g.busy), free = sum(g => g.free), noView = sum(g => g.noView)
  return { groups, total: busy + free + noView, busy, free, noView }
}
