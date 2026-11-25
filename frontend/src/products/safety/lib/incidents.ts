/** Incidents as the operator works them: which ones a filter shows, which still wait for an acknowledgement, how
 *  long each has lasted, in what order. Pure: the pages and the banner build on it. */
import { FEATURES, type Feature, type SafetyEvent, type Settings, type Severity } from './types'

export interface Filters {
  /** Empty: every feature. */
  features: Feature[]
  camera: string | null
  /** The zone's name, as the alert recorded it (details.zone_name). */
  zone: string | null
  severity: Severity | null
  /** open: still going; unacked: waiting for an acknowledgement, ended or not. */
  status: 'all' | 'open' | 'unacked'
}

export const NO_FILTERS: Filters = { features: [], camera: null, zone: null, severity: null, status: 'all' }

/** The colour a card takes (its data-tone): red for every fall incident, sure or for review, as for fire; otherwise its
 *  severity's -- red critical, amber warning, grey info or none. The severity chip still says the severity. */
export function toneOf(ev: Pick<SafetyEvent, 'feature' | 'severity'>): Severity {
  if (ev.feature === 'fall') return 'critical'
  return (SEVERITIES as string[]).includes(ev.severity ?? '') ? (ev.severity as Severity) : 'info'
}

const STATUSES: Filters['status'][] = ['all', 'open', 'unacked']
const SEVERITIES: Severity[] = ['info', 'warning', 'critical']

/** The filters a link opens the list with (?status=unacked from the banner, ?feature=fire,ppe, ?camera=, ?zone=,
 *  ?severity=); anything unknown is ignored. */
export function filtersFromQuery(q: Record<string, string>): Filters {
  const features = (q.feature ?? '').split(',').filter((f): f is Feature => (FEATURES as string[]).includes(f))
  return {
    features,
    camera: q.camera || null,
    zone: q.zone || null,
    severity: (SEVERITIES as string[]).includes(q.severity) ? (q.severity as Severity) : null,
    status: (STATUSES as string[]).includes(q.status) ? (q.status as Filters['status']) : 'all',
  }
}

/** The zone an incident fired in, by the name it had then; null for one with no zone (the whole frame). */
export function zoneOf(ev: SafetyEvent): string | null {
  const name = ev.details?.zone_name
  return typeof name === 'string' && name ? name : null
}

/** The zones these incidents fired in, each once, sorted: the zone filter's choices. */
export function zonesOf(events: SafetyEvent[]): string[] {
  return [...new Set(events.map(zoneOf).filter((z): z is string => z !== null))].sort()
}

export function isOpen(ev: SafetyEvent): boolean {
  return ev.incident_status === 'ongoing'
}

/** Its feature requires an acknowledgement and nobody has given one -- ended incidents too: someone must see each. */
export function needsAck(ev: SafetyEvent, settings: Settings | null): boolean {
  if (!ev.feature || ev.acknowledged_at || !settings) return false
  return settings.features[ev.feature]?.ack_required ?? false
}

export function matches(ev: SafetyEvent, f: Filters, settings: Settings | null): boolean {
  if (f.features.length && (!ev.feature || !f.features.includes(ev.feature))) return false
  if (f.camera && ev.camera_id !== f.camera) return false
  if (f.zone && zoneOf(ev) !== f.zone) return false
  if (f.severity && ev.severity !== f.severity) return false
  if (f.status === 'open' && !isOpen(ev)) return false
  if (f.status === 'unacked' && !needsAck(ev, settings)) return false
  return true
}

/** Seconds it has lasted: an ongoing incident until `now`, an ended one its recorded length; null for a plain alert. */
export function durationSeconds(ev: SafetyEvent, now: number): number | null {
  if (ev.incident_status === 'ongoing' && ev.incident_started_at) {
    return Math.max(0, Math.round((now - Date.parse(ev.incident_started_at)) / 1000))
  }
  return ev.incident_duration_seconds ?? null
}

/** Open incidents first, then the newest. */
export function sortIncidents(events: SafetyEvent[]): SafetyEvent[] {
  return [...events].sort((a, b) => Number(isOpen(b)) - Number(isOpen(a)) || b.id - a.id)
}

/** Alerts only (not VLM comments), newest version of each event kept. */
export function alertsOnly(events: SafetyEvent[]): SafetyEvent[] {
  return events.filter(e => e.event_type === 'alert')
}

/** What the banner lists: the server's unacknowledged incidents (it filters by feature, so frequent alerts of other
 *  features cannot push one out) plus live ones that arrived since, each in its newest version, minus those
 *  acknowledged from this screen a moment ago. Newest first. */
export function waitingList(server: SafetyEvent[], live: SafetyEvent[], settings: Settings | null,
                            justAcked: Set<number>): SafetyEvent[] {
  const byId = new Map(server.map(e => [e.id, e]))
  for (const e of live) byId.set(e.id, e)
  return [...byId.values()]
    .filter(e => !justAcked.has(e.id) && needsAck(e, settings))
    .sort((a, b) => b.id - a.id)
}

/** Local calendar day as YYYY-MM-DD. */
function dayKey(t: Date): string {
  const p = (n: number) => String(n).padStart(2, '0')
  return `${t.getFullYear()}-${p(t.getMonth() + 1)}-${p(t.getDate())}`
}

/** Incidents in runs of the local day they started on, in the order given; today's and yesterday's runs are named. */
export function byDay(events: SafetyEvent[], now: number): { day: string; events: SafetyEvent[] }[] {
  const today = dayKey(new Date(now))
  const yesterday = dayKey(new Date(new Date(now).setDate(new Date(now).getDate() - 1)))
  const out: { day: string; events: SafetyEvent[] }[] = []
  for (const e of events) {
    const key = dayKey(new Date(e.incident_started_at ?? e.timestamp))
    const day = key === today ? 'today' : key === yesterday ? 'yesterday' : key
    const last = out[out.length - 1]
    if (last && last.day === day) last.events.push(e)
    else out.push({ day, events: [e] })
  }
  return out
}

/** Whether a tile flashes now: when the newest incident going on is newer than any it has seen -- not on first sight
 *  (a tile appearing, or moving to another row, finds incidents already going on), and not when the newest one ends
 *  while an older one goes on. `seen` is the newest incident id the tile has seen. */
export function newIncident(seen: number | null, newest: number | null): { flash: boolean; seen: number | null } {
  if (newest === null) return { flash: false, seen }
  if (seen === null) return { flash: false, seen: newest }
  return newest > seen ? { flash: true, seen: newest } : { flash: false, seen }
}
