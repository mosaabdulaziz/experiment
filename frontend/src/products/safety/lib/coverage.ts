/** Where on a camera each feature looks, from the camera's zones (products/safety/README.md: the "zones" column of
 *  the roles table).
 *
 *  PPE, fall and fire cover the whole frame until some zone switches their section on; from then on, only those
 *  zones. Proximity, intrusion and overcrowding are zone rules: without a zone they do nothing. */
import type { CameraFeature, SafetyCamera } from './types'

/** What coverage reads of a zone: a saved one (SafetyZone) or one in the zone editor (iris's DraftZone). */
export interface ZoneLike {
  name: string
  intrusion: { mode: string }
  overcrowding: { enabled: boolean }
  proximity?: { enabled: boolean }
  ppe?: { enabled: boolean }
  fall?: { enabled: boolean }
  fire?: { enabled: boolean }
}

export type Coverage = { kind: 'whole' } | { kind: 'zones'; zones: string[] } | { kind: 'none' }

const BLOCK_FEATURES = ['ppe', 'fall', 'fire'] as const

function uses(feature: CameraFeature, z: ZoneLike): boolean {
  switch (feature) {
    case 'proximity': return Boolean(z.proximity?.enabled)
    case 'zones': return z.intrusion.mode !== 'off' || z.overcrowding.enabled
    default: return Boolean(z[feature]?.enabled)
  }
}

export function coverage(feature: CameraFeature, zones: ZoneLike[]): Coverage {
  const named = zones.filter(z => uses(feature, z)).map(z => z.name)
  if (named.length) return { kind: 'zones', zones: named }
  return (BLOCK_FEATURES as readonly string[]).includes(feature) ? { kind: 'whole' } : { kind: 'none' }
}

/** The PPE / fall / fire features a change takes from the whole frame down to some zones (worth a warning before
 *  saving: everything outside those zones stops being checked). */
export function narrowedFeatures(saved: ZoneLike[], draft: ZoneLike[]): CameraFeature[] {
  return BLOCK_FEATURES.filter(f => coverage(f, saved).kind === 'whole' && coverage(f, draft).kind === 'zones')
}

/** The features switched on that cannot raise anything yet: proximity and zone rules need a zone. */
export function unusedFeatures(features: CameraFeature[], zones: ZoneLike[]): CameraFeature[] {
  return features.filter(f => coverage(f, zones).kind === 'none')
}

export interface SetupWarning {
  camera: SafetyCamera
  feature: CameraFeature
}

/** What still needs setting up across the site: each camera's features that cannot raise anything yet. A camera whose
 *  zones have not loaded is left out rather than guessed at. */
export function setupWarnings(cameras: SafetyCamera[], zones: Record<string, ZoneLike[] | undefined>): SetupWarning[] {
  return cameras.flatMap(camera => {
    const own = zones[camera.id]
    return own ? unusedFeatures(camera.features, own).map(feature => ({ camera, feature })) : []
  })
}

