import { apiGet } from './client'

/**
 * `GET /api/alerts/clip/{id}` — what ClipStore persisted at build time. `range` is
 * measured off the built file (not the requested window), and `anchor` is the pts
 * the alert actually fired at, so the two together locate the trigger inside the
 * clip even when the recorder lagged and the cut came out short.
 */
export interface AlertClipMeta {
  cameras: string[]
  meta?: {
    camera?: string
    anchor?: number
    range?: [number, number]
  }
}

export const alertClipsApi = {
  meta: (clipId: string) => apiGet<AlertClipMeta>(`/api/alerts/clip/${clipId}`),
}
