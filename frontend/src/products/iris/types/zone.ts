export interface ZonePoint {
  x: number
  y: number
}

/** Intrusion detection mode for a zone. 'restricted' alerts on entry, 'valid' alerts on
 *  exit (the polygon is the only permitted area), 'off' disables intrusion entirely
 *  (e.g. an overcrowding-only zone). */
export type IntrusionMode = 'restricted' | 'valid' | 'off'

export interface ZoneIntrusion {
  mode: IntrusionMode
  /** Object classes that trigger an intrusion alert when in/out of the zone. */
  classes: string[]
}

export interface ZoneOvercrowding {
  enabled: boolean
  /** Alert when the count of the chosen classes inside the zone reaches this. */
  threshold: number
  classes: string[]
}

/** One configured zone on a stream. `id` is the backend-assigned stable label
 *  (z0, z1, …); `name` is the operator-facing label. */
/** What a proximity zone does with a person whose feet are hidden and no depth check confirms them. */
export type HiddenFeet = 'unconfirmed' | 'suppress'

/** The proximity module's per-zone settings (perception/modules/proximity/README.md). Distances are in person
 *  heights: there is no metric calibration. */
export interface ZoneProximity {
  enabled: boolean
  vehicle_classes: string[]
  max_gap: number
  persist_s: number
  grace_s: number
  cooldown_s: number
  stationary_window_s: number
  stationary_motion: number
  depth_check: boolean
  depth_max_gap: number         // the depth check's cut-off: relative depth gap at or under which it is "the same depth"
  hidden_feet: HiddenFeet
  person_classes?: string[]
}

/** Present on GET /zones when the running pipeline has the proximity module. */
export interface ProximityInfo {
  available: boolean
  defaults: Omit<ZoneProximity, 'enabled'>
  depth_model: boolean
}

/** The safety product's features that a zone switches on or off (app/features.py ZONE_FEATURES). A camera with no
 *  zone carrying a feature's block checks it on the whole frame. */
export type ZoneFeature = 'ppe' | 'fall' | 'fire'
export const ZONE_FEATURES: ZoneFeature[] = ['ppe', 'fall', 'fire']

/** PPE's block: on or off, and the equipment checked in the zone (some of the module's items, PpeInfo); without
 *  `items`, everything the module checks. */
export interface ZonePpe {
  enabled: boolean
  items?: string[]
}

/** Present on GET /zones when the running pipeline has the PPE module: the equipment a zone can ask it to check. */
export interface PpeInfo {
  items: string[]
}

/** How the editor names the PPE module's items (perception/modules/ppe: helmet, vest). */
export const PPE_ITEM_LABEL: Record<string, string> = { helmet: 'hard hat', vest: 'vest' }
export const ppeItemName = (item: string): string => PPE_ITEM_LABEL[item] ?? item

export interface Zone {
  id: string
  name: string
  points: ZonePoint[]
  intrusion: ZoneIntrusion
  overcrowding: ZoneOvercrowding
  /** Present when the pipeline runs the proximity module, or the zone has proximity settings saved. */
  proximity?: ZoneProximity
  ppe?: ZonePpe
  fall?: { enabled: boolean }
  fire?: { enabled: boolean }
}

/** GET/POST /api/streams/{id}/zones — a stream owns a list of zones. */
export interface ZonesPayload {
  zones: Zone[]
  /** Generation counter announced to the sidecar after a successful save;
   *  absent in legacy mode or when the config regeneration failed. */
  zones_version?: number
  /** The proximity settings' generation announced to the pipeline after a save (when it runs the module). */
  proximity_version?: number
  proximity?: ProximityInfo
  ppe?: PpeInfo
}

/** Zone-config apply status derived from `GET /api/cameras` (DeepStream only). */
export interface ZonesStatus {
  enabled: boolean
  version: number
  appliedVersion: number
  /** Per camera, the version its zones process applied (absent for an older sidecar: use appliedVersion). */
  appliedByCamera?: Record<string, number>
  /** The proximity module's reload acknowledgement, when the pipeline runs it. */
  proximity?: { version: number; appliedVersion: number } | null
}
