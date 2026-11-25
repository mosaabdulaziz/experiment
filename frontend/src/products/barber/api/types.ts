/** barber-service's JSON shapes, checked against the live server (8003).
 * Times are naive local ISO strings ("2026-09-25T13:41:05"). */

export type ZoneType = 'entrance_line' | 'waiting' | 'chair' | 'seat' | 'washing' | 'row' | 'door_area' | 'face_area'
export interface Zone {
  id: string
  name: string
  type: ZoneType
  points: number[][]
  pair?: string
  owner?: boolean
  seat_of?: string
  /** a chair zone's kind (2026-11-10): 'pedicure' = a pedicure chair; none = a barber chair */
  kind?: 'pedicure'
  chair?: string
  inside?: number[]
  [extra: string]: unknown
}
export type ZonesByCam = Record<string, Zone[]>
export type ZoneProblem =
  | { code: 'pair_single_camera'; pair: string; camera: string }
  | { code: 'pair_twice_in_camera'; pair: string }
  | { code: 'pair_owner_count'; pair: string; count: number }
  | { code: 'pair_kind_mismatch'; pair: string }
  | { code: 'chair_without_seat'; camera: string; zone: string }
  | { code: 'door_area_ignored'; camera: string }
  | { code: 'face_area_unknown_chair'; camera: string; zone: string }

export type CustomerState = 'waiting' | 'haircut' | 'washing' | 'served' | 'finished'
export type Outcome = 'completed' | 'walked_out' | 'lost' | 'void_staff' | 'interrupted'
export type Bucket = 'hour' | 'day' | 'week' | 'month'

export interface LiveCustomer {
  visit_id: string
  person_id: string
  ticket: number | null
  state: CustomerState
  state_since: string | null
  entered_at: string
  entry_observed: boolean
  waiting_started_at: string | null
  service_started_at: string | null
  wait_target_s: number | null
  chair_id: string | null
  session_id: string | null
  haircut_started_at: string | null
  barber_id: string | null
  cameras: string[]
}
/** A chair's live state (chair engine). An older backend sends only
 * person_id/session_id/sitting, so every newer field is optional. */
export type ChairState = 'empty' | 'starting' | 'sitting' | 'away' | 'no_view'
export interface ChairOcc {
  person_id: string | null
  session_id: string | null
  sitting: number
  state?: ChairState
  /** start of the state; for sitting, the start of this sitting */
  since?: string | null
  /** the chair's haircut start (confirmed haircut only): a resumed haircut's clock runs from it */
  haircut_since?: string | null
  /** the camera that owns (decides) the chair */
  camera?: string
  /** the chair's kind (2026-11-10): a pedicure chair's sessions are never customers */
  kind?: 'haircut' | 'pedicure'
  occupant?: number | null
  cover?: boolean
  /** away only: the haircut resumes if the customer is back by then */
  resume_until?: string | null
  /** sitting, and re-ID is still deciding whether the sitter is the chair's
   * previous customer back or a new one: since when (ce_resume_check) */
  checking?: string | null
  /** while checking: the previous customer's ticket */
  check_ticket?: number | null
  /** how the re-ID check decided, while the sitting it decided runs (2026-10-18) */
  resume?: ResumeDecision | null
}
/** A re-ID check's answer: the sitter is the chair's previous customer back (resumed) or a new one. sim null: no
 * re-ID on one side, the time rule decided. threshold: what sim had to reach (lower within the wash window). */
export interface ResumeDecision {
  resumed: boolean
  sim: number | null
  threshold: number | null
  /** how long the chair had been left */
  gap_s: number
  at: string
  /** the previous customer's ticket */
  ticket: number | null
  /** return: the chair taken again after its customer left; reseat: its seat seen taken again after cover only */
  kind?: 'return' | 'reseat'
}
export type OccupancyWhy = 'counted' | 'not_still' | 'not_owner' | 'cannot_sit' | 'in_chair_zone'
export interface Occupancy { chair: string | null; why: OccupancyWhy | null }
export type RuleWhy = 'occupied' | 'staff_cover' | 'staff' | 'low_conf' | 'no_mask' | 'below_min' | 'other_chair' | 'no_chair' | 'other_view'
/** The chair engine's verdict on one person (cameras owning chairs only):
 * the chair tested, the share of the seat the lower body covers, and why. */
export interface Rule { chair: string | null; share: number | null; why: RuleWhy }
export interface SceneEntry {
  track_id: number
  box: [number, number, number, number]
  person_id: string | null
  ticket: number | null
  /** 'entered': seen at the door, not yet a confirmed visit (no ticket) */
  state: CustomerState | 'entered' | null
  staff: boolean | null
  areas: string[]
  seat: string | null
  still: boolean
  /** Occupancy test point (chair-decision anchor) and standing point, both
   * normalized 0..1; omitted by an older backend, so overlays must draw
   * nothing rather than throw. */
  anchor?: [number, number]
  foot?: [number, number]
  seat_line?: [[number, number], [number, number]]
  occupancy?: Occupancy
  /** Chair engine (newer backend): the person's mask polygon as flat
   * normalized [x1, y1, x2, y2, ...], the y of the top of the lower 35 %
   * (the cut line), and the chair rule's verdict. */
  outline?: number[] | null
  lower_y?: number | null
  rule?: Rule | null
}
export type Scene = Record<string, SceneEntry[]>
export interface Alert {
  id: string
  kind: 'excessive_wait' | 'excessive_service'
  camera: string
  visit_id: string
  session_id: string | null
  started_at: string
  ended_at: string | null
  open: boolean
  chair_id: string | null
  barber_id: string | null
  /** the visit's own entered_at, so a link can find its shop day */
  visit_entered_at?: string | null
  has_snapshot?: boolean
  /** the ticket of the visit (haircut alert) or the customer wait (wait alert) */
  ticket?: number | null
  /** the subject's box in the snapshot, [x, y, w, h] normalised */
  subject_box?: [number, number, number, number] | null
  /** the camera the snapshot was taken on */
  subject_cam?: string | null
}
/** The kids car's place (sub-project 4): found by the locator (auto) or set by hand (manual). */
export interface KidsCarPlace { camera: string; box: [number, number, number, number]; source: 'auto' | 'manual'; conf: number | null; since: number }
export interface KidsCarState { place: KidsCarPlace | null; enabled: boolean; locator: boolean; sizes: Record<string, [number, number]> }
export interface PerBarber { barber_id: string | null; name: string | null; haircuts: number; average_haircut_s: number | null }
export interface Share { count: number; percent: number | null }
export interface Today {
  served_customers: number
  completed_haircuts: number
  visitors: number
  average_wait_s: number | null
  average_haircut_s: number | null
  haircuts_per_barber: PerBarber[]
}
export interface LiveSnapshot {
  type: 'live_snapshot'
  server_time: string
  /** the shop's name, and whether GET /api/barber/logo has its logo (2026-10-18; an older backend sends none) */
  site?: { place: string; branch: string | null; logo?: boolean }
  /** a run on recordings (2026-10-18): the footage's shop day and how far it has played -- its reports' clock */
  footage?: { day: string; start: string; end: string; now: string } | null
  shop_day: string
  /** the hour the shop day starts (policy), for placing a timestamp on its day */
  day_start_hour?: number
  customers_waiting: number | null
  customers: LiveCustomer[]
  chairs: Record<string, ChairOcc>
  scene: Scene
  clips: Record<string, { started_ms: number; duration_s: number }>
  zones: ZonesByCam
  zone_display?: Record<string, boolean>
  zone_problems: ZoneProblem[]
  roster: Record<string, string>
  roster_confirmed: boolean
  open_alerts: Alert[]
  recent_alerts: Alert[]
  targets: { wait_target_s: number; service_target_s: number }
  today: Today
}
export interface LongestWait { seconds: number; visit_id: string; waiting_started_at: string; service_started_at: string }
export interface LongestHaircut {
  seconds: number
  session_id: string
  visit_id: string
  chair_id: string
  barber_id: string | null
  started_at: string
  ended_at: string | null
}
export interface Summary {
  served_customers: number
  visitors: number
  customer_visits: number
  walk_outs: number
  lost: number
  completed_haircuts: number
  average_wait_s: number | null
  longest_wait: LongestWait | null
  excessive_wait: Share
  average_haircut_s: number | null
  longest_haircut: LongestHaircut | null
  excessive_haircut: Share
  returning_visits: Share
}
export interface Stats extends Summary {
  from: string
  to: string
  haircuts_per_barber: PerBarber[]
  avg_days_between_visits: number | null
  customers_waiting: number
  bucket?: Bucket
  series?: ({ bucket: string } & Summary)[]
}
export interface Session {
  id: string
  visit_id: string
  chair_id: string
  started_at: string
  ended_at: string | null
  service_s: number | null
  wash_s: number
  barber_id: string | null
  service_target_s: number | null
  excessive_service: boolean
  open: boolean
}
export interface Staff { id: string; name: string; active: boolean; created_at: string }
export interface CameraConf { id: string; stream: string; name?: string | null }
export type Policy = Record<string, number>
export type RosterDetail = Record<string, { staff_id: string; since: string }>
export interface RosterConfirmation { day: string; confirmed: boolean; confirmed_at: string | null }
export interface CustomerRow {
  id: string
  first_seen: string
  last_seen: string
  visit_count: number
  last_visit_id: string | null
  face_count: number
  thumb: string | null
  thumb_kind?: 'face' | 'body' | null   // 2026-09-29: a customer without a face shows his best body crop
  /** 2026-10-18: each visit's chairs (ids), newest visit first; empty chairs = not served */
  served?: CustomerVisitServed[]
}
export interface CustomerVisitServed { at: string; chairs: string[] }
/** the part of a visit the panel shows */
export interface CustomerVisitBrief { id: string; entered_at: string; haircut_sessions: { chair_id: string }[] }
export interface CropRef { id: number; quality: number; taken_at: string; visit_id: string | null; url: string }
export interface CustomerDetail extends CustomerRow {
  faces: CropRef[]; bodies: CropRef[]; entries: { id: string; at: string }[]; visits?: CustomerVisitBrief[]
}
export interface InsightsFilters { from: string; to: string; bucket: Bucket; barber: string }
