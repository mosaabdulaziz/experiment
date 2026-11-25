export type EventType = 'comment' | 'alert'

export type Severity = 'critical' | 'warning' | 'info'
export type IncidentStatus = 'ongoing' | 'resolved' | 'unknown'
export interface EventClip { clip_id: string; status: string }

/** Absent/'none' = no VLM description for this event (legacy backend, or a non-VLM alert type). */
export type DescriptionStatus = 'pending' | 'ready' | 'none'

/**
 * Lifecycle of an alert's preview clip. Saved as 'pending' when the alert fires,
 * resolved by the backend's AlertClipService to 'ready' or 'unavailable'. Absent
 * when alert clips are off or the backend is the local (non-DeepStream) one.
 */
export type ClipStatus = 'pending' | 'ready' | 'unavailable'

export interface IrisEvent {
  id: number
  timestamp: string
  event_type: EventType
  message: string
  tags: string[]
  stream_offset_start: number | null
  stream_offset_end: number | null
  camera: string | null
  description_status?: DescriptionStatus
  description?: string | null
  clip_id?: string | null
  clip_status?: ClipStatus | null
  severity?: Severity | null
  category?: string | null
  incident_status?: IncidentStatus | null
  incident_started_at?: string | null
  incident_duration_seconds?: number | null
  /** Every clip of an incident, oldest first; clips[0] is clip_id. */
  clips?: EventClip[]
  /** A module alert's own fields, as strings (proximity: gap, track ids, confirmation, ...). */
  details?: Record<string, string> | null
  /** The alert clip the incident's snapshot is stored with (GET /api/alerts/snapshot/{id}/{camera}/{event id}). */
  snapshot_id?: string | null
  /** When an operator acknowledged it (POST /api/events/{id}/acknowledge); null until then. */
  acknowledged_at?: string | null
}
