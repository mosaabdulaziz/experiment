import type { ClipStatus, DescriptionStatus, EventClip, IncidentStatus, IrisEvent } from './event'
import type { Mode } from './config'

export type WsStatus = 'idle' | 'analyzing' | 'paused' | 'error'

/**
 * The backend emits `event_update` in two shapes from two independent enrichment
 * paths — descriptions (stream_manager) and preview clips (alert_clips). They are
 * NOT interchangeable: handling one shape as if it were the other overwrites the
 * other path's fields with undefined. Incidents (module_alerts / alert_clips) send the
 * whole record, their clip list, or their status.
 */
export type EventUpdate =
  | { type: 'event_update'; id: number; description_status: DescriptionStatus; description?: string | null }
  | { type: 'event_update'; id: number; clip_status: ClipStatus; clip_id?: string | null; clips?: EventClip[] }
  | { type: 'event_update'; id: number; event: IrisEvent; camera?: string }
  | { type: 'event_update'; id: number; clips: EventClip[] }
  | { type: 'event_update'; id: number; incident_status: IncidentStatus }
  | { type: 'event_update'; id: number; acknowledged_at: string }

export type WsMessage =
  | { type: 'history'; events: IrisEvent[]; mode?: Mode }
  | { type: 'event'; event: IrisEvent }
  | EventUpdate
  | { type: 'status'; status: WsStatus }
  | { type: 'mode'; mode: Mode }
  | { type: 'error'; message?: string }
  | { type: 'cleared' }
