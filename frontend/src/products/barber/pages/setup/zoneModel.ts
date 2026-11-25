/** Pure editing operations on zones -- the zone editor (Task 14) calls these
 * and re-renders from what they return. Nothing here mutates its inputs;
 * every function returns new objects/arrays. Kept compatible with
 * barber-service/zones.py's `validate_zones`: an entrance line must carry an
 * `inside` point once it has its 2 points, a seat needs `seat_of`, a chair
 * or washing zone needs >= 3 points to close. */
import type { Zone, ZonesByCam, ZoneType } from '../../api/types'
import { clamp01 } from '../../ui/geometry'

type Pt = [number, number]

const clampPt = (p: readonly [number, number]): Pt => [clamp01(p[0]), clamp01(p[1])]

/** validate_zones's own check, mirrored: how far `pt` sits off the infinite
 * line through p1, p2, as a fraction of the line's length. */
function offLine(pt: number[], p1: number[], p2: number[]): number {
  const dx = p2[0] - p1[0]
  const dy = p2[1] - p1[1]
  const len = Math.hypot(dx, dy) || 1e-9
  const cross = dx * (pt[1] - p1[1]) - dy * (pt[0] - p1[0])
  return Math.abs(cross) / len
}

const ON_LINE_THRESHOLD = 1e-3

/** Default `inside` for a freshly-drawn door line: the midpoint, nudged off
 * the line along its normal so it clears validate_zones's "not on the line"
 * check. A line flush with a frame edge has its whole length on that edge,
 * so nudging outward would just get clamped straight back onto the line --
 * try the centre-ward side first, then the other, and keep whichever
 * clamped point actually clears the threshold. The user can still flip it
 * to the other side with `flipInside`. */
function defaultInside(p1: number[], p2: number[]): Pt {
  const dx = p2[0] - p1[0]
  const dy = p2[1] - p1[1]
  const len = Math.hypot(dx, dy) || 1e-6
  const nx = -dy / len
  const ny = dx / len
  const offset = 0.05
  const mx = (p1[0] + p2[0]) / 2
  const my = (p1[1] + p2[1]) / 2
  const towardCentre = nx * (0.5 - mx) + ny * (0.5 - my) >= 0
  const candidates: Pt[] = towardCentre
    ? [clampPt([mx + nx * offset, my + ny * offset]), clampPt([mx - nx * offset, my - ny * offset])]
    : [clampPt([mx - nx * offset, my - ny * offset]), clampPt([mx + nx * offset, my + ny * offset])]
  return candidates.find(c => offLine(c, p1, p2) >= ON_LINE_THRESHOLD) ?? candidates[0]
}

/** Reflection of `pt` across the infinite line through p1, p2. */
function reflectAcross(pt: number[], p1: number[], p2: number[]): Pt {
  const dx = p2[0] - p1[0]
  const dy = p2[1] - p1[1]
  const len2 = dx * dx + dy * dy || 1e-9
  const vx = pt[0] - p1[0]
  const vy = pt[1] - p1[1]
  const t = (vx * dx + vy * dy) / len2
  const cx = p1[0] + t * dx
  const cy = p1[1] + t * dy
  return clampPt([2 * cx - pt[0], 2 * cy - pt[1]])
}

export function newZone(type: ZoneType, cam: string, existing: Zone[]): Zone {
  const letter = cam.split('-').pop()![0]
  const ids = new Set(existing.map(z => z.id))
  let n = 1
  while (ids.has(`${letter}-${type}-${n}`)) n++
  return { id: `${letter}-${type}-${n}`, name: '', type, points: [] }
}

export function addPoint(z: Zone, p: [number, number]): Zone {
  const point = clampPt(p)
  if (z.type === 'entrance_line') {
    if (z.points.length >= 2) return { ...z, points: z.points.map(pt => [...pt]) }
    const points = [...z.points.map(pt => [...pt]), point]
    if (points.length === 2 && z.inside === undefined) {
      return { ...z, points, inside: defaultInside(points[0], points[1]) }
    }
    return { ...z, points }
  }
  return { ...z, points: [...z.points.map(pt => [...pt]), point] }
}

export function isClosable(z: Zone): boolean {
  return z.type === 'entrance_line' ? z.points.length === 2 : z.points.length >= 3
}

export function movePoint(z: Zone, i: number, p: [number, number]): Zone {
  const point = clampPt(p)
  return { ...z, points: z.points.map((pt, idx) => (idx === i ? point : [...pt])) }
}

export function deleteZone(zones: Zone[], id: string): Zone[] {
  return zones.filter(z => z.id !== id && z.seat_of !== id)
}

export function setPair(all: ZonesByCam, cam: string, id: string, pair: string | null): ZonesByCam {
  const list = all[cam] ?? []
  const next = list.map(z => {
    if (z.id !== id) return z
    if (pair === null) {
      const { pair: _pair, owner: _owner, ...rest } = z
      return rest as Zone
    }
    return { ...z, pair }
  })
  return { ...all, [cam]: next }
}

export function setOwner(all: ZonesByCam, cam: string, id: string, owner: boolean): ZonesByCam {
  const pair = all[cam]?.find(z => z.id === id)?.pair
  const out: ZonesByCam = {}
  for (const [c, list] of Object.entries(all)) {
    out[c] = list.map(z => {
      if (c === cam && z.id === id) return { ...z, owner }
      if (owner && pair && z.pair === pair) {
        const { owner: _owner, ...rest } = z
        return rest as Zone
      }
      return z
    })
  }
  return out
}

export function setSeatOf(zones: Zone[], seatId: string, chairId: string): Zone[] {
  return zones.map(z => (z.id === seatId ? { ...z, seat_of: chairId } : z))
}

export function setFaceChair(zones: Zone[], id: string, chair: string): Zone[] {
  return zones.map(z => (z.id === id ? { ...z, chair } : z))
}

export function flipInside(z: Zone): Zone {
  if (!z.inside || z.points.length < 2) return { ...z }
  return { ...z, inside: reflectAcross(z.inside, z.points[0], z.points[1]) }
}

export function rename(zones: Zone[], id: string, name: string): Zone[] {
  return zones.map(z => (z.id === id ? { ...z, name } : z))
}

export function pairChoices(all: ZonesByCam, type: 'chair' | 'washing' | 'waiting'): string[] {
  const ids = new Set<string>()
  for (const list of Object.values(all)) {
    for (const z of list) if (z.type === type && z.pair) ids.add(z.pair)
  }
  return [...ids].sort()
}

/** A chair zone's kind (2026-11-10): 'pedicure' or none (a barber chair). */
export function setKind(zones: Zone[], id: string, kind: 'pedicure' | null): Zone[] {
  return zones.map(z => {
    if (z.id !== id) return z
    if (kind) return { ...z, kind }
    const { kind: _k, ...rest } = z
    return rest as Zone
  })
}
