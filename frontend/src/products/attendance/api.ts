import { jsonFetch } from '../../shared/api/http'
/** Client for attendance-service (the app's own BFF, proxied under
 * /attendance) plus the two read-only iris endpoints that carry camera
 * pixels. Shapes mirror iris-frontend's api/attendance.ts, extended with the
 * dashboard fields added 2026-09-01. */

export type AttendanceStatus = 'present' | 'absent' | 'leave'

export interface AttendanceRecord {
  employee_id: string
  name: string
  photo_url: string
  day: string
  first_arrival: string | null
  last_departure: string | null
  present: boolean
  code: string | null
  department: string | null
  display_name_ar: string | null
  on_leave: boolean
  status: AttendanceStatus
}

export interface RecordingInfo {
  cam_id: string
  recording_start: string
  // null for a real live camera with no fixed duration to loop (2026-09-12,
  // issue 1) -- unused by the tile clock (see components.tsx's useWallClock)
  duration_seconds: number | null
  stream_started_at: string | null
  // Camera clock (issue 1 Phase 2): milliseconds to add to server time to get
  // this camera's own clock (from its RTCP Sender Reports), or null when the
  // camera provides none -- the tile then shows server time.
  clock_offset_ms: number | null
  clock_source: 'camera' | 'server'
}

export interface CameraInfo {
  id: string
  running: boolean
  is_file: boolean
  source: string
  offset: number | null
  roles: string[]
  /** Optional human-readable name/zone from sidecar.yaml; null when unset,
   * same fallback convention as roles (missing means "not configured", not
   * "every camera"). */
  label: string | null
  zone: string | null
}

// The shared mechanism; the paths below are this product's own.
const apiGet = <T,>(path: string): Promise<T> => jsonFetch<T>(path)

export const api = {
  daily: (day: string) => apiGet<AttendanceRecord[]>(`/attendance/api/attendance?day=${day}`),
  listRecordings: () => apiGet<RecordingInfo[]>('/attendance/api/recordings'),
  serverTime: () => apiGet<{ epoch_ms: number }>('/attendance/api/time'),
  listCameras: () =>
    apiGet<{ cameras: CameraInfo[] }>('/api/cameras').then(r => r.cameras),
  employeePhotoUrl: (employeeId: string) => `/attendance/api/employees/${employeeId}/photo`,
  setLeave: async (employeeId: string, onLeave: boolean) => {
    const res = await fetch(`/attendance/api/employees/${employeeId}/leave`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ on_leave: onLeave }),
    })
    if (!res.ok) throw new Error(`leave toggle failed: HTTP ${res.status}`)
  },
}

/* ── productivity monitoring (2026-09-01) ─────────────────────────────── */

export type DeskState = 'at_desk' | 'away' | 'absent'

export interface MonitoringRecord {
  employee_id: string
  name: string
  display_name_ar: string | null
  department: string | null
  code: string | null
  state: DeskState
  at_desk_seconds: number
  phone_seconds: number
  away_seconds: number
  presence_seconds: number
  working_seconds: number
  not_working_seconds: number
}

export interface SceneEntry {
  track_id: number
  employee_id: string | null
  name: string | null
  zone_id: string | null
  /** box center travelled noticeably in the last few seconds (a passer-by, not a worker) */
  moving: boolean
  /** employee credited for the seat this body occupies (identity anchored to the seat) */
  credited_as: string | null
  credited_name: string | null
  /** squat box = seated-shaped (see policy seated_max_aspect) */
  seated: boolean
  /** normalized [left, top, w, h] or null */
  box: number[] | null
}

export type ZoneType = 'seat' | 'restricted'

export interface Zone {
  id: string
  name: string
  /** normalized [[x, y], ...] */
  points: number[][]
  /** absent on a zone saved before this field existed -- treat as 'seat' (see zones.py) */
  type?: ZoneType
  /** only meaningful when type === 'restricted' */
  allowed_roles?: string[]
}

export interface Policy {
  debounce_seconds: number
  away_grace_seconds: number
  phone_sustained_seconds: number
  seated_max_aspect: number
  working_formula: string
}

export type ViolationType = 'role_not_allowed' | 'unidentified'

export interface Violation {
  id: string
  employee_id: string | null
  name: string | null
  camera: string
  zone_id: string
  violation_type: ViolationType
  started_at: string
  ended_at: string
  open: boolean
}

export interface MonitoringSnapshot {
  type: 'monitoring_snapshot'
  day: string
  records: MonitoringRecord[]
  scene: Record<string, SceneEntry[]>
  zones: Record<string, Zone[]>
  policy: Policy
  violations: Violation[]
}

export const monitoringApi = {
  snapshot: () => apiGet<MonitoringSnapshot>('/attendance/api/monitoring'),
  savePolicy: async (p: Partial<Policy>) => {
    const res = await fetch('/attendance/api/monitoring/policy', {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(p),
    })
    if (!res.ok) throw new Error(`policy save failed: HTTP ${res.status}`)
    return res.json() as Promise<Policy>
  },
  saveZones: async (camId: string, zones: Zone[]) => {
    const res = await fetch(`/attendance/api/monitoring/zones/${camId}`, {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(zones),
    })
    if (!res.ok) throw new Error(`zones save failed: HTTP ${res.status}`)
    return res.json() as Promise<Zone[]>
  },
  // Deletes one day's restricted-zone violations only (2026-09-13).
  clearViolations: async (day: string) => {
    const res = await fetch(`/attendance/api/monitoring/violations?day=${day}`, { method: 'DELETE' })
    if (!res.ok) throw new Error(`clear violations failed: HTTP ${res.status}`)
    return res.json() as Promise<{ day: string; removed: number }>
  },
}

/* ── attendance v2: schedule/gap policy (separate from Policy above --
 * that one belongs to desk/phone monitoring, this one to attendance; see
 * sessions.py's module docstring) ───────────────────────────────────── */

export type CameraType = 'office' | 'break_room'

export interface AttendancePolicy {
  work_start: string // "HH:MM"
  work_end: string
  // 2026-09-14: today's departure column stays empty until this "HH:MM"
  departure_reveal_time: string
  late_grace_minutes: number
  early_leave_grace_minutes: number
  break_min_seconds: number
  exit_gap_seconds: number
  violation_debounce_seconds: number
  camera_types: Record<string, CameraType>
  /** cameras checked for who may be seen on them: inside their drawn restricted zones, else the whole stream */
  access_control?: Record<string, { allowed_roles: string[] }>
}

export const attendancePolicyApi = {
  get: () => apiGet<AttendancePolicy>('/attendance/api/attendance/policy'),
  save: async (p: Partial<AttendancePolicy>) => {
    const res = await fetch('/attendance/api/attendance/policy', {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(p),
    })
    if (!res.ok) throw new Error(`attendance policy save failed: HTTP ${res.status}`)
    return res.json() as Promise<AttendancePolicy>
  },
}

/* ── attendance v2: admin-driven enrollment ──────────────────────────── */

export interface AdminEmployee {
  id: string
  name: string
  role: string | null
  department: string | null
  email: string | null
  code: string | null
  gallery_rec_id: string
  active: boolean
  on_leave: boolean  // 2026-09-14: leave toggle lives on the Employees page
  photo_url: string
}

export interface PhotoUploadResult {
  filename: string
  ok: boolean
  face_id?: string
  embeddings?: number
  mode?: 'primary' | 'extra'
  reason?: 'no_face' | 'multiple_faces' | 'not_distinct' | 'unreadable_photo' | 'gallery_write_failed' | 'timeout' | 'subprocess_failed'
  conflicts_with_name?: string
  similarity?: number
  count?: number
}

export const adminApi = {
  list: () => apiGet<AdminEmployee[]>('/attendance/api/admin/employees'),
  create: async (payload: { name: string; role?: string; department?: string; email?: string; code?: string }) => {
    const res = await fetch('/attendance/api/admin/employees', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
    })
    if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail ?? `HTTP ${res.status}`)
    return res.json() as Promise<AdminEmployee>
  },
  update: async (id: string, payload: Partial<Pick<AdminEmployee, 'name' | 'role' | 'department' | 'active' | 'email' | 'code'>>) => {
    const res = await fetch(`/attendance/api/admin/employees/${id}`, {
      method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
    })
    if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail ?? `HTTP ${res.status}`)
    return res.json() as Promise<AdminEmployee>
  },
  uploadPhotos: async (id: string, files: File[]) => {
    const body = new FormData()
    files.forEach(f => body.append('files', f))
    const res = await fetch(`/attendance/api/admin/employees/${id}/photos`, { method: 'POST', body })
    if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail ?? `HTTP ${res.status}`)
    return res.json() as Promise<{ employee_id: string; results: PhotoUploadResult[] }>
  },
  remove: async (id: string) => {
    const res = await fetch(`/attendance/api/admin/employees/${id}`, { method: 'DELETE' })
    if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail ?? `HTTP ${res.status}`)
  },
}

/* ── attendance v2: per-employee history page ────────────────────────── */

export interface DaySession {
  entry: string
  exit: string | null
  breaks: { start: string; end: string }[]
}

export interface DayRow {
  day: string
  sessions: DaySession[]
  first_arrival: string | null   // 2026-09-14: first entry of the day
  last_departure: string | null  // last exit; null while the last session is still open
  span_seconds: number           // first arrival -> last departure, gaps included
  attendance_seconds: number
  break_seconds: number
  net_seconds: number
  overtime_seconds: number
  early_arrival: boolean  // first entry before work_start (2026-09-13)
  late: boolean
  early_leave: boolean
  absent: boolean
}

export interface EmployeeHistory {
  employee_id: string
  name: string
  display_name_ar: string | null
  month: string // "YYYY-MM"
  days: DayRow[]
  summary: {
    days_present: number
    times_late: number
    times_early_leave: number
    total_overtime_seconds: number
    total_net_seconds: number
  }
}

export const historyApi = {
  get: (employeeId: string, month?: string) =>
    apiGet<EmployeeHistory>(`/attendance/api/employees/${employeeId}/history${month ? `?month=${month}` : ''}`),
}

/* ── fire/smoke detection (2026-09-04) ───────────────────────────────────
 * Iris's own alert log (app/events.py), not attendance-service's -- fire/smoke
 * alerts are a full-frame sidecar behavior with no attendance involvement.
 * Every alert kind shares this one table; the "fire_smoke" tag is what
 * narrows it down (see app/stream_manager.py's count_alert_message). */

export type EventStatus = 'pending' | 'ready' | 'unavailable' | null

// Fire/smoke only (app/events.py); null on every other alert kind.
// "ended_at" is deliberately not a field: the backend derives it from
// started_at + duration_seconds (or, while ongoing, from started_at + elapsed
// wall time) rather than storing a value that could drift from those two.
export type IncidentStatus = 'ongoing' | 'resolved' | 'unknown' | null

export interface ClipRef {
  clip_id: string
  status: string
}

export interface IrisEvent {
  id: number
  timestamp: string
  event_type: string
  message: string
  tags: string[]
  camera: string | null
  description: string | null
  description_status: EventStatus
  clip_id: string | null
  clip_status: EventStatus
  incident_status: IncidentStatus
  incident_started_at: string | null
  incident_duration_seconds: number | null
  // Every clip attached to this incident, oldest first; clips[0] is the same
  // clip clip_id/clip_status point at. Empty on events saved before this field
  // existed even when clip_id is set -- see effectiveClips() in useFireAlerts.
  clips: ClipRef[]
  acknowledged_at: string | null
}

export const alertClipUrl = (clipId: string, camera: string): string =>
  `/api/alerts/clip/${clipId}/${camera}`

export const eventsApi = {
  list: (limit = 500) => apiGet<IrisEvent[]>(`/api/events?limit=${limit}`),
  acknowledge: async (id: number): Promise<void> => {
    const res = await fetch(`/api/events/${id}/acknowledge`, { method: 'POST' })
    if (!res.ok) throw new Error(`acknowledge failed: HTTP ${res.status}`)
  },
  /** Scoped clear (2026-09-06): deletes only events carrying `tag`, e.g.
   * 'fire_smoke' for the Fire/Smoke page's Clear button -- NOT the same as
   * iris-frontend's "clear everything" button, which would also wipe
   * unrelated alert kinds this dashboard never shows. */
  clearByTag: async (tag: string): Promise<number> => {
    const res = await fetch(`/api/events/tag/${encodeURIComponent(tag)}`, { method: 'DELETE' })
    if (!res.ok) throw new Error(`clear failed: HTTP ${res.status}`)
    return (await res.json()).deleted as number
  },
}
