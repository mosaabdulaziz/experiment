import { ZONE_FEATURES } from '@/types/zone'
import { EDITOR_EN, type EditorStrings } from './editorStrings'
import type { ZonePoint, ZoneIntrusion, ZoneOvercrowding, ZoneProximity, IntrusionMode, Zone, ZonePpe } from '@/types/zone'

/** A zone being edited in the layers panel. `key` is a client-only stable identity for
 *  React lists + selection (the backend assigns canonical ids on save). Drawing state
 *  (`ringClosed`/`closed`) lives per-zone so each zone tracks its own polygon progress. */
export interface DraftZone {
  key: string
  name: string
  points: ZonePoint[]
  ringClosed: boolean
  closed: boolean
  intrusion: ZoneIntrusion
  overcrowding: ZoneOvercrowding
  /** Absent until the zone has proximity settings (the section starts from the pipeline's defaults). */
  proximity?: ZoneProximity
  /** The safety product's per-zone switches; absent unless set (iris never shows them). */
  ppe?: ZonePpe
  fall?: { enabled: boolean }
  fire?: { enabled: boolean }
}

let counter = 0
const newKey = (): string => `z-${Date.now().toString(36)}-${counter++}`

export function makeDraftZone(name: string): DraftZone {
  return {
    key: newKey(),
    name,
    points: [],
    ringClosed: false,
    closed: false,
    intrusion: { mode: 'off', classes: ['person'] },
    overcrowding: { enabled: false, threshold: 5, classes: ['person'] },
  }
}

export function draftFromZone(z: Zone): DraftZone {
  const closed = z.points.length >= 3
  return {
    key: newKey(),
    name: z.name,
    points: z.points,
    ringClosed: closed,
    closed,
    intrusion: z.intrusion,
    overcrowding: z.overcrowding,
    proximity: z.proximity,
    ...(z.ppe ? { ppe: z.ppe } : {}),
    ...(z.fall ? { fall: z.fall } : {}),
    ...(z.fire ? { fire: z.fire } : {}),
  }
}

/** The zone's purpose color: restricted → accent, valid → green, overcrowding-only → yellow.
 *  One language across the rail swatch, the canvas polygon, the config accent, and the feed. */
export function zoneColor(mode: IntrusionMode): string {
  const colorVar = mode === 'valid' ? '--green' : mode === 'off' ? '--yellow' : '--accent'
  const fallback = mode === 'valid' ? '#6bf2a8' : mode === 'off' ? '#ffd166' : '#ff8a4c'
  return getComputedStyle(document.documentElement).getPropertyValue(colorVar).trim() || fallback
}

export function zoneSummary(z: DraftZone, s: EditorStrings = EDITOR_EN): string {
  const parts: string[] = []
  if (z.intrusion.mode === 'restricted') parts.push(s.sumRestricted)
  else if (z.intrusion.mode === 'valid') parts.push(s.sumValid)
  if (z.overcrowding.enabled) parts.push(s.sumCrowd(z.overcrowding.threshold))
  if (z.proximity?.enabled) parts.push(s.sumProximity)
  if (z.ppe?.enabled) parts.push(s.sumPpe(z.ppe.items))
  if (z.fall?.enabled) parts.push(s.sumFalls)
  if (z.fire?.enabled) parts.push(s.sumFire)
  return parts.length ? parts.join(s.separator) : s.sumNone
}

/** Why this zone can't be saved yet (drawing unfinished or config incomplete), or null. */
export function zoneInvalidReason(z: DraftZone, s: EditorStrings = EDITOR_EN): string | null {
  if (!z.closed || z.points.length < 3) return s.finishDrawing(z.name)
  if (z.intrusion.mode === 'off' && !z.overcrowding.enabled && !z.proximity?.enabled
      && !ZONE_FEATURES.some(f => z[f]?.enabled))
    return s.turnOn(z.name)
  if (z.intrusion.mode !== 'off' && z.intrusion.classes.length === 0)
    return s.pickIntrusion(z.name)
  if (z.overcrowding.enabled && z.overcrowding.classes.length === 0)
    return s.pickOvercrowding(z.name)
  if (z.proximity?.enabled && z.proximity.vehicle_classes.length === 0)
    return s.pickVehicle(z.name)
  if (z.ppe?.enabled && z.ppe.items?.length === 0)
    return s.pickPpe(z.name)
  return null
}
