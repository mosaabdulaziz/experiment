/** The IRIS app's endpoints this console uses: /api/safety/* (products/safety/api.py) and the shared event, stream
 *  and zone routes. */
import { jsonBody, jsonFetch } from '../../shared/api/http'
import type { ApplyStatus, FallCandidate, SafetyCamera, SafetyEvent, SafetyZone, Settings } from './lib/types'

/** One role sidecar's stream as GET /api/cameras lists it (app/stream_manager.py list_streams). */
export interface StreamInfo {
  id: string
  camera: string
  feature: string | null
  running: boolean
  offset: number | null
}

/** A chat the Telegram bot was added to or written in (GET /api/safety/telegram/chats). */
export interface TelegramChat {
  id: string
  title: string
  type: string
}

const enc = encodeURIComponent

/** jsonFetch for /api/safety/*, whose failures explain themselves in `error` (products/safety/api.py) where the
 *  shared helper reads FastAPI's `detail`: without it the operator saw "HTTP 400" instead of the reason. */
async function safetyFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init)
  if (!res.ok) {
    const body = await res.json().catch(() => null)
    const reason = body?.error ?? body?.detail
    throw new Error(typeof reason === 'string' ? reason : `${path}: HTTP ${res.status}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  settings: () => safetyFetch<Settings>('/api/safety/settings'),
  saveSettings: (change: Partial<Settings> | Record<string, unknown>) =>
    safetyFetch<Settings>('/api/safety/settings', jsonBody('PUT', change)),
  cameras: () => safetyFetch<{ cameras: SafetyCamera[] }>('/api/safety/cameras').then(r => r.cameras),
  saveCamera: (id: string, change: { name?: string; features?: string[] }) =>
    safetyFetch<SafetyCamera>(`/api/safety/cameras/${enc(id)}`, jsonBody('PUT', change)),
  apply: () => safetyFetch<ApplyStatus>('/api/safety/apply', { method: 'POST' }),
  applyStatus: () => safetyFetch<ApplyStatus>('/api/safety/apply'),
  unacknowledged: () => safetyFetch<{ events: SafetyEvent[] }>('/api/safety/unacknowledged').then(r => r.events),
  telegramTest: () => safetyFetch<{ ok: boolean }>('/api/safety/telegram/test', { method: 'POST' }),
  telegramChats: () => safetyFetch<{ chats: TelegramChat[] }>('/api/safety/telegram/chats').then(r => r.chats),
  events: (limit = 300) => jsonFetch<SafetyEvent[]>(`/api/events?limit=${limit}`),
  acknowledge: (id: number) => jsonFetch<{ id: number; acknowledged_at: string }>(
    `/api/events/${id}/acknowledge`, { method: 'POST' }),
  fallCandidates: () => safetyFetch<FallCandidate[]>('/api/safety/fall/candidates?status=candidate'),
  reviewFall: (id: number, status: 'confirmed' | 'dismissed') =>
    safetyFetch<{ id: number; status: string }>(`/api/safety/fall/review/${id}`, jsonBody('POST', { status })),
  streams: () => jsonFetch<{ cameras: StreamInfo[] }>('/api/cameras').then(r => r.cameras),
  zones: (camera: string) => jsonFetch<{ zones: SafetyZone[] }>(`/api/streams/${enc(camera)}/zones`)
    .then(r => r.zones),
}

/** A still of a camera straight from go2rtc: its picture when no feature runs on it. */
export const stillUrl = (camera: string): string => `/api/safety/cameras/${enc(camera)}/snapshot`

/** The alert clip of an event, played from the stream that recorded it. */
export const clipUrl = (ev: SafetyEvent): string | null =>
  ev.clip_id && ev.camera && ev.clip_status === 'ready' ? `/api/alerts/clip/${enc(ev.clip_id)}/${enc(ev.camera)}` : null
