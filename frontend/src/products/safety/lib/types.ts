/** What the safety product reads from the IRIS app (products/safety/README.md, "The app side"). */

/** The five detection features, each a role sidecar, plus camera health (app/features.py FEATURES). */
export type Feature = 'proximity' | 'zones' | 'ppe' | 'fall' | 'fire' | 'health'
export const FEATURES: Feature[] = ['proximity', 'zones', 'ppe', 'fall', 'fire', 'health']
/** The ones a camera can be toggled to run (products/safety/safety.yaml features). */
export type CameraFeature = Exclude<Feature, 'health'>
export const CAMERA_FEATURES: CameraFeature[] = ['proximity', 'zones', 'ppe', 'fall', 'fire']

export type Severity = 'info' | 'warning' | 'critical'

/** One event from /api/events and the /ws stream; `camera` is the stream (`<camera>@<role>`). */
export interface SafetyEvent {
  id: number
  timestamp: string
  event_type: string
  message: string
  tags: string[]
  camera: string | null
  camera_id: string | null
  feature: Feature | null
  severity: Severity | string | null
  incident_status: 'ongoing' | 'resolved' | 'unknown' | null
  incident_started_at: string | null
  incident_duration_seconds: number | null
  clip_id: string | null
  clip_status: string | null
  clips: { clip_id: string; status: string }[]
  acknowledged_at: string | null
  details: Record<string, unknown> | null
  /** The alert clip holding the incident's own frame (proximity incidents), when it has one. */
  snapshot_id?: string | null
}

export interface FeatureSettings {
  ack_required: boolean
  notify: boolean
  min_severity: Severity
}

/** GET /api/safety/settings */
export interface Settings {
  features: Record<Feature, FeatureSettings>
  /** `enabled`: the switch over all of Telegram (off by default); each feature's `notify` applies only when it is on. */
  telegram: { enabled: boolean; chat_id: string; token_set: boolean }
  language: 'en' | 'ar'
}

/** GET /api/safety/cameras */
export interface SafetyCamera {
  id: string
  name: string
  features: CameraFeature[]
  /** `fps`: the frames a second camera health last measured on that stream (null until it has a reading). */
  streams: { id: string; feature: CameraFeature | null; running: boolean; fps?: number | null }[]
}

export interface ApplyStatus {
  version: number
  applied_version: number
  applying: boolean
}

/** A zone as GET /api/streams/{camera}/zones returns it, with the safety features' blocks. */
export interface SafetyZone {
  id: string
  name: string
  points: { x: number; y: number }[]
  intrusion: { mode: 'restricted' | 'valid' | 'off'; classes: string[] }
  overcrowding: { enabled: boolean; threshold: number; classes: string[] }
  proximity?: { enabled: boolean } & Record<string, unknown>
  ppe?: { enabled: boolean }
  fall?: { enabled: boolean }
  fire?: { enabled: boolean }
}

/** One borderline fall from the fall service's review queue (GET /api/safety/fall/candidates). */
export interface FallCandidate {
  id: number
  camera: string
  pts: number | null
  kind: string | null
  evaluation: string
  combined: number
  status: 'candidate' | 'acknowledged' | 'confirmed' | 'dismissed'
  created_at: string
  note?: string | null
}
