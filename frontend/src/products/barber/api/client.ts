import type { CameraConf, CustomerDetail, CustomerRow, InsightsFilters, Policy, RosterConfirmation, RosterDetail,
  Staff, Stats, Zone, ZonesByCam, ZoneProblem, Alert, LiveSnapshot, KidsCarState } from './types'

const BASE = '/barber/api'

export class ApiError extends Error {
  constructor(public status: number, public detail: string) { super(detail) }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, init && { ...init, headers: { 'Content-Type': 'application/json', ...init.headers } })
  if (!res.ok) {
    let detail = res.statusText
    try { const j = await res.json(); if (typeof j.detail === 'string') detail = j.detail } catch { /* not json */ }
    throw new ApiError(res.status, detail)
  }
  return res.status === 204 ? (undefined as T) : res.json()
}
const q = (params: Record<string, string | number | boolean | undefined | null>) => {
  const s = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== '') s.set(k, String(v))
  const text = s.toString()
  return text ? `?${text}` : ''
}
const json = (method: string, body: unknown): RequestInit => ({ method, body: JSON.stringify(body) })

/** The report endpoint (2026-10-15): a day / 7 days / month as html, pdf or docx; a month picker's YYYY-MM -> its 1st. */
export type ReportKind = 'day' | 'week' | 'month'
export type ReportFormat = 'html' | 'pdf' | 'docx'
/** The panel's start: the user's pick, else the page's day (followed until the user picks; final review I2), shaped
 * for the kind: YYYY-MM for a month, YYYY-MM-DD otherwise. */
export function reportStart(kind: ReportKind, picked: string | null, day: string): string {
  const v = picked ?? day
  if (kind === 'month') return v.slice(0, 7)
  return v.length === 7 ? `${v}-01` : v
}
export function reportUrl(r: { kind: ReportKind; start: string; lang: 'en' | 'ar'; format: ReportFormat; download?: boolean }): string {
  const start = /^\d{4}-\d{2}$/.test(r.start) ? `${r.start}-01` : r.start
  return `${BASE}/barber/report${q({ kind: r.kind, start, lang: r.lang, format: r.format, download: r.download ? 1 : undefined })}`
}
/** The haircut photo report (2026-10-22): one shop day as html or pdf; the shop network only. */
export function photoReportUrl(r: { start: string; lang: 'en' | 'ar'; format: 'html' | 'pdf'; download?: boolean }): string {
  return `${BASE}/barber/report/photos${q({ start: r.start, lang: r.lang, format: r.format, download: r.download ? 1 : undefined })}`
}
/** Saved reports (2026-10-25): the folders the service wrote when a period ended; the files on the shop network only. */
export interface SavedFile { report: 'analysis' | 'photos'; lang: 'en' | 'ar'; format: ReportFormat; name: string }
export interface SavedReport {
  id: string; kind: ReportKind; first: string; last: string; lap_ended: string | null; label: string | null
  saved_at: string; files: SavedFile[]
}
export interface SavedReports { enabled: boolean; allowed: boolean; items: SavedReport[] }
export function savedReportUrl(id: string, name: string, download = false): string {
  return `${BASE}/barber/reports/saved/file${q({ id, name, download: download ? 1 : undefined })}`
}

export const api = {
  live: () => call<LiveSnapshot>('/barber/live'),
  photoReportAllowed: () => call<{ allowed: boolean }>('/barber/report/photos/allowed'),
  savedReports: (kind: ReportKind) => call<SavedReports>(`/barber/reports/saved${q({ kind })}`),
  cameras: () => call<CameraConf[]>('/barber/cameras'),
  stats: (f: Pick<InsightsFilters, 'from' | 'to' | 'barber'> & { bucket?: string }) =>
    call<Stats>(`/barber/stats${q({ from: f.from, to: f.to, bucket: f.bucket, barber: f.barber })}`),
  alerts: (p: { from?: string; to?: string; open?: boolean; visit?: string; limit?: number }) => call<Alert[]>(`/barber/alerts${q(p)}`),
  customers: (limit: number, offset: number, sort: 'last_seen' | 'visit_count') =>
    call<{ rows: CustomerRow[]; total: number }>(`/barber/customers${q({ limit, offset, sort })}`),
  customer: (id: string) => call<CustomerDetail>(`/barber/customers/${encodeURIComponent(id)}`),
  eraseCustomer: (id: string) => call<unknown>(`/barber/customers/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  staff: () => call<Staff[]>('/staff'),
  addStaff: (name: string) => call<Staff>('/staff', json('POST', { name })),
  patchStaff: (id: string, patch: Partial<Pick<Staff, 'name' | 'active'>>) => call<Staff>(`/staff/${id}`, json('PATCH', patch)),
  roster: () => call<RosterDetail>('/barber/roster?detail=1'),
  assignChair: (chair: string, staffId: string | null) => call<Record<string, string>>(`/barber/roster/${encodeURIComponent(chair)}`, json('PUT', { staff_id: staffId })),
  rosterConfirmation: () => call<RosterConfirmation>('/barber/roster/confirmation'),
  confirmRoster: () => call<RosterConfirmation>('/barber/roster/confirmation', { method: 'POST' }),
  zones: () => call<ZonesByCam>('/monitoring/zones'),
  saveZones: (cam: string, zones: Zone[]) => call<Zone[]>(`/monitoring/zones/${encodeURIComponent(cam)}`, json('PUT', zones)),
  zoneProblems: () => call<ZoneProblem[]>('/monitoring/zones/problems'),
  policy: () => call<Policy>('/barber/policy'),
  savePolicy: (changes: Policy) => call<Policy>('/barber/policy', json('PUT', changes)),
  kidsCar: () => call<KidsCarState>('/barber/kids-car'),
  pinKidsCar: (camera: string, x: number, y: number) => call<KidsCarState>('/barber/kids-car', json('PUT', { camera, x, y })),
  clearKidsCar: () => call<KidsCarState>('/barber/kids-car', { method: 'DELETE' }),
}

/** Server crop paths are "/api/barber/crops/..."; the browser reaches them under /barber. */
export const cropUrl = (path: string) => `/barber${path}`
export const snapshotUrl = (alertId: string) => cropUrl(`/api/barber/alerts/${encodeURIComponent(alertId)}/snapshot`)
/** The shop's logo (the site's `logo`; the live snapshot's site.logo says it has one). */
export const logoUrl = cropUrl('/api/barber/logo')
export const videoWsUrl = (stream: string) =>
  `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/go2rtc/api/ws?src=${encodeURIComponent(stream)}`
export const stillUrl = (stream: string, bust: number) => `/go2rtc/api/frame.jpeg?src=${encodeURIComponent(stream)}&t=${bust}`
