/** The product's own service, under its own prefix.
 *
 * Nothing here reads Redis or opens a socket: the tab asks the product's five REST routes
 * and the platform's published frames (live/FrameLayer). The confirmed fall also reaches the
 * platform on its own alert path; this is the operator's own surface.
 */
import type { FallEvent } from './alerts/severity'

const API = '/fall_detection/api'

export type Status = {
  device: string | null
  weights_verified: boolean
  classifier_ready: boolean
  paths_live: string[]
}

export type Review = {
  id: number
  camera: string
  pts: number | null
  track: number | null
  kind: string | null
  evaluation: string
  combined: number
  path_scores: Record<string, number | null>
  clip_window: [number, number] | null
  status: string
  note: string | null
  created_at: string
}

/** The service's own words, when it sent any: a 400 says what the verdict must be. */
async function failure(base: string, res: Response): Promise<string> {
  try {
    const j = await res.json()
    const detail = (j as { detail?: unknown } | null)?.detail
    if (typeof detail === 'string') return `${base} — ${detail}`
  } catch { /* not json */ }
  return base
}

/** The wait was torn down: the server may already have committed the write. */
function isAbort(err: unknown): boolean {
  if (typeof err !== 'object' || err === null) return false
  const name = (err as { name?: unknown }).name
  return name === 'TimeoutError' || name === 'AbortError'
}

async function read<T>(path: string): Promise<T> {
  const res = await fetch(`${API}${path}`, { signal: AbortSignal.timeout(4000) })
  if (!res.ok) throw new Error(await failure(`${path}: ${res.status}`, res))
  return res.json() as Promise<T>
}

export const status = () => read<Status>('/status')
export const cameras = () => read<string[]>('/cameras')
export const candidates = () => read<Review[]>('/candidates')

/** The history since the cursor -- only what arrived while the operator was away. */
export const events = (since: number, camera?: string) =>
  read<FallEvent[]>(`/events?since=${since}${camera ? `&camera=${encodeURIComponent(camera)}` : ''}`)

/** The operator's verdict on a queued candidate. */
export async function review(id: number, verdict: string): Promise<void> {
  let res: Response
  try {
    res = await fetch(`${API}/review/${id}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status: verdict }),
      signal: AbortSignal.timeout(4000),
    })
  } catch (err) {
    // The request reached the server before the wait was torn down: the verdict may be committed.
    if (isAbort(err)) throw new Error(`review ${id}: the verdict may have landed — reload to confirm`)
    throw err
  }
  if (!res.ok) throw new Error(await failure(`review ${id}: ${res.status}`, res))
}
