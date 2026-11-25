import { apiGet, apiPost } from './client'

/**
 * Raw shape of `GET /api/cameras`. The endpoint serves two backends:
 *   - DeepStream (multi-stream): `{ cameras: StreamInfo[], pipeline: string }` — no `current` key.
 *   - Local webcam (legacy):     `{ cameras: CameraInfo[], current: ... }`.
 * The presence of `current` is the reliable discriminator (it holds even when
 * `cameras` is empty), so callers branch on that rather than on array contents.
 */
export interface CamerasResponse {
  cameras: unknown[]
  current?: number | string | null
  /** DeepStream only: desired pipeline state as set via iris:ctrl:pipeline */
  pipeline?: string
  /** DeepStream only: zone-config apply handshake (version written by the app,
   *  applied_version acked by the sidecar after hot-reloading the ROI config). */
  zones?: {
    enabled: boolean
    version: number
    applied_version: number
    /** Per camera: the version the zones process running it applied. */
    applied_by_camera?: Record<string, number>
    proximity?: { version: number; applied_version: number } | null
  }
}

/** `GET`/`POST /api/proximity/preview`: what the proximity pipeline draws on every camera's live view, one
 *  setting for all viewers. `available` only when the pipeline runs the proximity module. */
export interface ProximityPreview {
  available: boolean
  masks: boolean
  lines: boolean
  boxes: boolean
  /** the latest depth map, blended under the drawings (cameras with a depth-check zone) */
  depth: boolean
  /** person heights: a person and a moving vehicle this far apart still get a line (close pairs always do) */
  line_range: number
  /** how alert clips and snapshots draw objects (the live view has its own switches) */
  clip_style: ProximityClipStyle
}

export type ProximityClipStyle = 'masks' | 'boxes'

export type ProximityPreviewLayer = 'boxes' | 'masks' | 'lines' | 'depth'

export type ProximityPreviewChange = Partial<Record<ProximityPreviewLayer, boolean>> & {
  line_range?: number
  clip_style?: ProximityClipStyle
}

export const streamsApi = {
  list: () => apiGet<CamerasResponse>('/api/cameras'),
  startPipeline: () => apiPost<{ status: string; pipeline: string }>('/api/streams/start'),
  stopPipeline: () => apiPost<{ status: string; pipeline: string }>('/api/streams/stop'),
  proximityPreview: () => apiGet<ProximityPreview>('/api/proximity/preview'),
  setProximityPreview: (change: ProximityPreviewChange) =>
    apiPost<ProximityPreview>('/api/proximity/preview', change),
}
