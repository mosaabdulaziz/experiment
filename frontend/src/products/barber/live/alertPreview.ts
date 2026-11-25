import type { Alert, LiveCustomer, Scene, SceneEntry, ZonesByCam } from '../api/types'
import { areaOf, type AreaTable } from '../floor/areas'

/** Where an alert's preview looks: a camera and a box on it -- the customer's
 * (a long wait), or the chair zone's bounds with its outline (a long haircut). */
export interface PreviewTarget { cam: string; box: SceneEntry['box']; outline?: number[][] }

/** A long haircut: the chair, not the person (2026-10-07, user: a sitter's
 * track id swaps and drops out; the chair stays put) -- its owner camera and
 * that camera's chair zone, for as long as the alert is open. Null when the
 * chair has no owner camera or no zone there. */
function chairTarget(chair: string | null | undefined, areas: AreaTable, zones: ZonesByCam): PreviewTarget | null {
  const cam = chair ? areas.byId[chair]?.owner : null
  const z = cam ? (zones[cam] ?? []).find(z => z.type === 'chair' && areaOf(z) === chair && z.points.length >= 3) : undefined
  if (!cam || !z) return null
  const xs = z.points.map(p => p[0])
  const ys = z.points.map(p => p[1])
  const [x0, y0] = [Math.min(...xs), Math.min(...ys)]
  return { cam, box: [x0, y0, Math.max(...xs) - x0, Math.max(...ys) - y0], outline: z.points }
}

/** A long haircut: its chair (chairTarget). A long wait: the customer as the
 * live scene sees him right now, on the camera that owns the waiting area
 * (the centre camera, whether or not he stands inside the zone this moment).
 * Only that camera -- the side cameras overlap the waiting area but don't
 * own it. Null while the owner doesn't see him (not tracked, or re-identified
 * as someone else) -- the Preview button is disabled then. */
export function previewTarget(alert: Pick<Alert, 'kind' | 'visit_id' | 'chair_id'>, customers: LiveCustomer[],
                              scene: Scene, areas: AreaTable, zones: ZonesByCam = {}): PreviewTarget | null {
  const c = customers.find(x => x.visit_id === alert.visit_id)
  if (alert.kind === 'excessive_service') return chairTarget(alert.chair_id ?? c?.chair_id, areas, zones)
  if (!c) return null
  let owners: (string | null | undefined)[] = areas.areas.filter(a => a.type === 'waiting').map(a => a.owner)
  if (!owners.length) owners = [areas.centre]
  for (const cam of new Set(owners)) {
    const e = cam ? scene[cam]?.find(x => x.person_id === c.person_id && x.box) : undefined
    if (cam && e) return { cam, box: e.box }
  }
  return null
}

/** Where the customer was last seen, and when (browser ms). */
export interface Seen { target: PreviewTarget; at: number }

/** Shadow tracking: the live scene drops a customer for a moment all the time
 * (measured 2026-09-28 over 90 s: the median customer is in only 59% of scene
 * updates; 65% of the dropouts last <= 1 s, 97% <= 10 s). Keep his last
 * position for `holdMs` after he drops out, so Preview doesn't flicker. */
export function shadow(prev: Seen | null | undefined, current: PreviewTarget | null, now: number, holdMs: number): Seen | null {
  if (current) return { target: current, at: now }
  return prev && now - prev.at <= holdMs ? prev : null
}

/** The picture the server saved when the alert opened (2026-10-18): what the sidebar's Preview shows when the live
 * scene has no target -- a wait whose customer lost his track on the wait camera, or a trackless extra wait. */
export interface StoredPicture { cam: string; box: [number, number, number, number]; openedAt: string }

export function storedPicture(a: Pick<Alert, 'has_snapshot' | 'subject_cam' | 'subject_box' | 'started_at'>): StoredPicture | null {
  return a.has_snapshot && a.subject_cam && a.subject_box ? { cam: a.subject_cam, box: a.subject_box, openedAt: a.started_at } : null
}

/** Preview works with a live sighting, or else with the stored picture. */
export const canPreview = (seen: Seen | null, stored: StoredPicture | null): boolean => !!seen || !!stored
