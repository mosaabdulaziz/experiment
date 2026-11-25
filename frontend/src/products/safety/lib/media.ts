/** What an incident shows as its picture -- on the incident cards and in the banner: its snapshot (the alert's own
 *  frame, which proximity incidents have), else its clip shown at the alert moment, else why there is none. The same
 *  rules as IRIS's incident cards (products/iris/lib/clips.ts clipState). Pure. */
import type { SafetyEvent } from './types'

/** How long the app keeps alert clips: IRIS's DEFAULT_CLIP_TTL_SECONDS (products/iris/lib/clips.ts). */
export const CLIP_TTL_SECONDS = 21600

export type Media =
  | { kind: 'snapshot'; src: string; clip: string | null }
  | { kind: 'clip'; clipId: string; clip: string }
  | { kind: 'pending' | 'unavailable' | 'expired' | 'none' }

const enc = encodeURIComponent

export function mediaOf(ev: SafetyEvent, now: number, ttlSeconds = CLIP_TTL_SECONDS): Media {
  // an incident whose first clip failed but whose escalation clip was stored still shows the clip it has
  const clipId = ev.clip_id ?? ev.clips?.find(c => c.status === 'ready')?.clip_id ?? null
  const status = ev.clip_id ? ev.clip_status : clipId ? 'ready' : ev.clip_status
  const clip = clipId && ev.camera ? `/api/alerts/clip/${enc(clipId)}/${enc(ev.camera)}` : null
  const started = Date.parse(ev.timestamp)
  const expired = !Number.isNaN(started) && (now - started) / 1000 > ttlSeconds
  if (ev.snapshot_id && ev.camera) {
    if (expired) return { kind: 'expired' }            // the snapshot is its clip's still: it goes with the clip
    return { kind: 'snapshot', src: `/api/alerts/snapshot/${enc(ev.snapshot_id)}/${enc(ev.camera)}/${ev.id}`, clip }
  }
  if (ev.event_type !== 'alert' || !status) return { kind: 'none' }
  if (status === 'pending') return { kind: 'pending' }
  if (status !== 'ready' || !clip || !clipId) return { kind: 'unavailable' }
  if (expired) return { kind: 'expired' }
  return { kind: 'clip', clipId, clip }
}
