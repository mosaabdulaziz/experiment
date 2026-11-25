import type { Zone, ZonesByCam, ZoneProblem } from '../api/types'
import { centroid } from '../ui/geometry'

const AREA_TYPES = new Set(['chair', 'washing', 'waiting'])
export const areaOf = (z: Zone) => z.pair ?? z.id

export interface Area { id: string; type: 'chair' | 'washing' | 'waiting'; name: string; owner: string | null; drawnIn: string[]; hasSeat: boolean
  /** a chair's kind from its owner zone (2026-11-10) */
  kind: 'haircut' | 'pedicure' }
export type GroupKey = 'left' | 'right' | 'vip' | 'kids' | 'other' | 'pedicure'
const GROUP_ORDER: GroupKey[] = ['right', 'left', 'vip', 'kids', 'other']

/** A chair's group from its name (shops without a door camera, 2026-10-14): R1.. right, L5.. left, VIP.., Kids/K.. */
export function groupOfName(name: string): GroupKey {
  if (/^R\d/i.test(name)) return 'right'
  if (/^L\d/i.test(name)) return 'left'
  if (/^VIP/i.test(name)) return 'vip'
  if (/^(kid|K\d|K$)/i.test(name)) return 'kids'
  return 'other'
}

const byNumber = (a: string, b: string) => a.localeCompare(b, undefined, { numeric: true })

export interface AreaTable {
  centre: string | null; areas: Area[]; byId: Record<string, Area>; rows: { left: string[]; right: string[] }
  groups: { key: GroupKey; ids: string[] }[]
}

/** Zones of every camera -> one row per shop area (a pair id, or an unpaired
 * zone's id), and the Shop board's two chair rows as the centre camera sees
 * them (the camera with the door line). Mirrors barber-service zones.py.
 * `cameras`, if given, restricts everything to that set -- a stale camera
 * still in zones.json (e.g. one no longer wired up) but not in the live
 * camera list is ignored entirely, including for centre detection. */
export function buildAreas(zones: ZonesByCam, cameras?: string[]): AreaTable {
  const allowed = cameras ? new Set(cameras) : null
  const cams = Object.keys(zones).filter(c => !allowed || allowed.has(c))
  const pairCount = (c: string) => (zones[c] ?? []).filter(z => !!z.pair).length
  const doorCams = cams.filter(c => (zones[c] ?? []).some(z => z.type === 'entrance_line'))
  // The centre camera is the door-line camera with the most paired zones
  // (a stale camera's door line rarely has any); ties go to the lower id.
  const centre = doorCams.reduce<string | null>((best, c) => {
    if (best === null) return c
    const diff = pairCount(c) - pairCount(best)
    if (diff > 0) return c
    if (diff < 0) return best
    return c < best ? c : best
  }, null)
  const byId: Record<string, Area> = {}
  const pos: Record<string, [number, number]> = {}          // centroid in the centre view
  for (const cam of cams) {
    const list = zones[cam] ?? []
    const seated = new Set(list.filter(z => z.type === 'seat').map(z => z.seat_of))
    for (const z of list) {
      if (!AREA_TYPES.has(z.type)) continue
      const id = areaOf(z)
      const a = byId[id] ??= { id, type: z.type as Area['type'], name: z.name || id, owner: null, drawnIn: [], hasSeat: false, kind: 'haircut' }
      if (!a.drawnIn.includes(cam)) a.drawnIn.push(cam)
      const owns = !!z.owner || !z.pair
      if (owns && a.owner === null) { a.owner = cam; a.name = z.name || id; a.kind = z.kind === 'pedicure' ? 'pedicure' : 'haircut' }
      if (owns && seated.has(z.id)) a.hasSeat = true
      if (cam === centre) pos[id] = centroid(z.points)
    }
  }
  const areas = Object.values(byId).sort((a, b) => a.id.localeCompare(b.id))
  const chairs = areas.filter(a => a.type === 'chair' && a.kind !== 'pedicure')
  // 2026-11-10: pedicure chairs are their own group, after the others, never in the rows
  const pedicure = areas.filter(a => a.type === 'chair' && a.kind === 'pedicure')
                        .sort((a, b) => byNumber(a.name, b.name)).map(a => a.id)
  const left: string[] = [], right: string[] = []
  const placed = chairs.filter(a => pos[a.id]).sort((a, b) => pos[a.id][1] - pos[b.id][1])
  for (const a of placed) (pos[a.id][0] < 0.5 ? left : right).push(a.id)
  for (const a of chairs.filter(c => !pos[c.id])) {
    const mates = (row: string[]) => row.filter(id => byId[id].owner === a.owner).length
    ;(mates(left) >= mates(right) ? left : right).push(a.id)
  }
  const groups = centre
    ? [{ key: 'left' as GroupKey, ids: left }, { key: 'right' as GroupKey, ids: right }]
    : GROUP_ORDER.map(key => ({ key, ids: chairs.filter(a => groupOfName(a.name) === key)
                                               .sort((a, b) => byNumber(a.name, b.name)).map(a => a.id) }))
                 .filter(g => g.ids.length)
  if (pedicure.length) groups.push({ key: 'pedicure', ids: pedicure })
  return { centre, areas, byId, rows: { left, right }, groups }
}

/** Port of zones.zone_set_problems: same checks, same order. `cameras`, if
 * given, drops every other camera's zones first, so a stale camera's own
 * problems (and its half of a pair) don't show up. */
export function zoneProblems(zones: ZonesByCam, cameras?: string[]): ZoneProblem[] {
  const allowed = cameras ? new Set(cameras) : null
  const entries = Object.entries(zones).filter(([cam]) => !allowed || allowed.has(cam))
  const problems: ZoneProblem[] = []
  const members: Record<string, [string, Zone][]> = {}
  for (const [cam, list] of entries) for (const z of list) if (z.pair) (members[z.pair] ??= []).push([cam, z])
  for (const pair of Object.keys(members).sort()) {
    const zs = members[pair]
    const cams = [...new Set(zs.map(([c]) => c))].sort()
    if (cams.length < 2) problems.push({ code: 'pair_single_camera', pair, camera: cams[0] })
    if (zs.length !== cams.length) problems.push({ code: 'pair_twice_in_camera', pair })
    const owners = zs.filter(([, z]) => z.owner).length
    if (owners !== 1) problems.push({ code: 'pair_owner_count', pair, count: owners })
    if (new Set(zs.filter(([, z]) => z.type === 'chair').map(([, z]) => z.kind ?? null)).size > 1)
      problems.push({ code: 'pair_kind_mismatch', pair })
  }
  for (const [cam, list] of entries) {
    const seated = new Set(list.filter(z => z.type === 'seat').map(z => z.seat_of))
    for (const z of list) if (z.type === 'chair' && (z.owner || !z.pair) && !seated.has(z.id)) problems.push({ code: 'chair_without_seat', camera: cam, zone: z.name })
  }
  const chairAreas = new Set<string>()
  for (const [, list] of entries) for (const z of list) if (z.type === 'chair') chairAreas.add(areaOf(z))
  for (const [cam, list] of entries) {
    for (const z of list) if (z.type === 'face_area' && !chairAreas.has(z.chair ?? '')) problems.push({ code: 'face_area_unknown_chair', camera: cam, zone: z.name })
  }
  return problems
}
