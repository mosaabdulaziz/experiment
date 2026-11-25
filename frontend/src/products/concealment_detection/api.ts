/** The concealment detection sidecar service, under its own prefix.
 *
 * Nothing here reads Redis or opens a socket: the tab polls the sidecar's REST routes
 * and renders the platform's published frames via live/FrameLayer.
 */
import type { StageEntry } from './format'

const API = '/concealment_detection/api'

export type CameraHealth = {
  camera_ok: boolean
  reason: string
}

export type Capabilities = {
  cart_detection: boolean
  bag_context_requires_boxed_detections: boolean
  durable_clips: boolean
}

export type Status = {
  running: boolean
  ready: boolean
  cameras: Record<string, CameraHealth>
  capabilities: Capabilities
}

export type CameraInfo = {
  id: string
  location: string
}

export type Tier = 'high' | 'medium' | 'low'
export type Severity = 'critical' | 'warning' | 'info'
export type TargetKind = 'clothing' | 'personal_bag'
export type ReviewVerdict = 'acknowledged' | 'concern_noted' | 'dismissed'
export type ReviewStatus = 'candidate' | ReviewVerdict

export const ALLOWED_VERDICTS: readonly ReviewVerdict[] = [
  'acknowledged',
  'concern_noted',
  'dismissed',
] as const

export type Candidate = {
  id: number
  camera: string
  location: string
  track_id: number
  pts: number
  display_time: string
  tier: Tier
  severity: Severity
  category: string
  kind: string
  score: number
  zone_id: string
  target_kind: TargetKind
  body_zone_name: string
  terminal_stage: string | null
  hold_duration_s: number | null
  dwell_duration_s: number | null
  description: string
  stage_timeline: StageEntry[]
  clip_window: [number, number] | number[]
  keyframes_pts: number[]
  clip_path: string | null
  thumbnail_path: string | null
  requires_human_verification: boolean
  status: ReviewStatus
  reviewer: string | null
  reviewed_at: string | null
  note: string
  created_at: string
}

export type Assessment = {
  id: number
  camera: string
  track_id: number
  pts: number
  state: string
  reason: string
  created_at: string
}

async function failure(base: string, res: Response): Promise<string> {
  try {
    const j = await res.json()
    const detail = (j as { detail?: unknown } | null)?.detail
    if (typeof detail === 'string') return `${base} - ${detail}`
  } catch { /* not json */ }
  return base
}

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

export async function cameras(): Promise<CameraInfo[]> {
  const raw = await read<Array<CameraInfo | string> | { cameras?: Array<CameraInfo | string> }>('/cameras')
  const list = Array.isArray(raw) ? raw : (raw.cameras ?? [])
  return list.map(c => (typeof c === 'string' ? { id: c, location: c } : c))
}

export type CandidatesQuery = {
  camera?: string
  review_status?: ReviewStatus
  since?: number
  limit?: number
  before?: number
}

export async function candidates(q?: CandidatesQuery): Promise<Candidate[]> {
  const params = new URLSearchParams()
  if (q?.camera) params.set('camera', q.camera)
  if (q?.review_status) params.set('review_status', q.review_status)
  if (q?.since !== undefined) params.set('since', String(q.since))
  if (q?.limit !== undefined) params.set('limit', String(q.limit))
  if (q?.before !== undefined) params.set('before', String(q.before))
  const qs = params.toString()
  const raw = await read<Candidate[] | { candidates?: Candidate[] }>(`/candidates${qs ? `?${qs}` : ''}`)
  return Array.isArray(raw) ? raw : (raw.candidates ?? [])
}

export const candidate = (id: number) => read<Candidate>(`/candidates/${id}`)

export async function assessments(q?: { camera?: string; limit?: number }): Promise<Assessment[]> {
  const params = new URLSearchParams()
  if (q?.camera) params.set('camera', q.camera)
  if (q?.limit !== undefined) params.set('limit', String(q.limit))
  const qs = params.toString()
  const raw = await read<Assessment[] | { assessments?: Assessment[] }>(`/assessments${qs ? `?${qs}` : ''}`)
  return Array.isArray(raw) ? raw : (raw.assessments ?? [])
}

export async function review(
  id: number,
  verdict: ReviewVerdict,
  opts?: { note?: string; reviewer?: string },
): Promise<Candidate> {
  let res: Response
  try {
    res = await fetch(`${API}/review/${id}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        status: verdict,
        note: opts?.note ?? '',
        ...(opts?.reviewer ? { reviewer: opts.reviewer } : {}),
      }),
      signal: AbortSignal.timeout(4000),
    })
  } catch (err) {
    if (isAbort(err)) throw new Error(`review ${id}: the verdict may have landed - reload to confirm`)
    throw err
  }
  if (!res.ok) throw new Error(await failure(`review ${id}: ${res.status}`, res))
  return res.json() as Promise<Candidate>
}
