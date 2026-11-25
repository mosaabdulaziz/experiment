import type { IrisEvent } from '@/types/event'
import type { AlertKind } from '@/lib/clips'
import { alertKind, clipState } from '@/lib/clips'
import { isDepthSuppressed } from '@/lib/proximity'

/** The Incidents page's filters. An empty camera or kind selection means "no filter"; within a group the chips are
 *  OR'd, the groups AND'd. Pairs the depth check held back (Ticket 6) are shown only when asked for, and then only
 *  they are: they are not alerts, but what a zone's depth cut-off kept back, to check the cut-off against. */
export interface IncidentFilterState {
  cameras: string[]
  kinds: AlertKind[]
  footageOnly: boolean
  heldBack: boolean
}

export const NO_FILTERS: IncidentFilterState = { cameras: [], kinds: [], footageOnly: false, heldBack: false }

export function matchesIncidentFilters(ev: IrisEvent, f: IncidentFilterState, ttlSeconds: number): boolean {
  if (isDepthSuppressed(ev) !== f.heldBack) return false
  if (f.cameras.length && (!ev.camera || !f.cameras.includes(ev.camera))) return false
  if (f.kinds.length && !f.kinds.includes(alertKind(ev))) return false
  if (f.footageOnly && clipState(ev, ttlSeconds) !== 'ready') return false
  return true
}
