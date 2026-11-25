/** Pure logic behind the Cameras view's occupancy and tracks overlays, kept
 * out of the SVG components so it can be unit-tested without a DOM: what to
 * draw for each scene entry's occupancy test point, the colour a bound
 * track is drawn in (the same one on every camera), and a small per-camera
 * ring buffer of recent anchors for the track trails. An older backend that
 * hasn't grown the new scene fields yet must make these draw nothing, never
 * throw. */
import { effectiveState, freshDecision } from './chairStrip'
import type { ChairOcc, ChairState, LiveCustomer, OccupancyWhy, ResumeDecision, SceneEntry, Zone } from '../api/types'
import type { Rect } from '../ui/Feed'

export type OccupancyReason = Exclude<OccupancyWhy, 'counted'>
export interface OccupancyMark {
  track_id: number
  anchor: [number, number]
  foot?: [number, number]
  seatLine?: [[number, number], [number, number]]
  counted: boolean
  /** null when counted, or when the backend didn't give a reason */
  reason: OccupancyReason | null
}

/** The marks the OccupancyOverlay draws: one per entry that carries both an
 * anchor and an occupancy verdict. Anything short of that (an older
 * backend, or a person never evaluated for a chair) draws nothing. */
export function occupancyMarks(entries: SceneEntry[]): OccupancyMark[] {
  const marks: OccupancyMark[] = []
  for (const e of entries) {
    if (!e.anchor || !e.occupancy) continue
    const why = e.occupancy.why
    marks.push({
      track_id: e.track_id,
      anchor: e.anchor,
      foot: e.foot,
      seatLine: e.seat_line,
      counted: why === 'counted',
      reason: why && why !== 'counted' ? why : null,
    })
  }
  return marks
}

function hash(s: string): number {
  let h = 0
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0
  return Math.abs(h)
}

/** The same hsl colour for the same person on every camera; white for an
 * unbound track (no person_id yet). */
export function personColor(personId: string | null): string {
  if (!personId) return '#ffffff'
  return `hsl(${hash(personId) % 360}, 70%, 55%)`
}

/** `#<track_id>`, or `#<track_id> → <first 6 chars of person_id>` once bound. */
export function trackLabel(trackId: number, personId: string | null): string {
  return personId ? `#${trackId} → ${personId.slice(0, 6)}` : `#${trackId}`
}

export interface TrailPoint { x: number; y: number; t: number }
/** Recent anchors, per camera, per track_id. */
export type TrailBuf = Record<string, Record<number, TrailPoint[]>>

/** Appends this tick's anchor (per track_id, for one camera) and drops
 * anything older than `keepMs`, on every track_id already buffered for that
 * camera too -- so a track that vanishes from the scene still ages out
 * instead of leaving a stale trail forever. */
export function pushTrail(buf: TrailBuf, cam: string, entries: SceneEntry[], nowMs: number, keepMs = 10000): TrailBuf {
  const camBuf: Record<number, TrailPoint[]> = { ...(buf[cam] ?? {}) }
  for (const e of entries) {
    if (!e.anchor) continue
    const prev = camBuf[e.track_id] ?? []
    camBuf[e.track_id] = [...prev, { x: e.anchor[0], y: e.anchor[1], t: nowMs }]
  }
  const cutoff = nowMs - keepMs
  for (const key of Object.keys(camBuf)) {
    const trackId = Number(key)
    const pruned = camBuf[trackId].filter(p => p.t >= cutoff)
    if (pruned.length) camBuf[trackId] = pruned
    else delete camBuf[trackId]
  }
  return { ...buf, [cam]: camBuf }
}

// ── chair engine (newer backend) ────────────────────────────────────────────

type Pt = [number, number]
const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)

/** A flat [x1, y1, x2, y2, ...] outline as points; null unless it is a
 * polygon (at least 3 points, all numbers). An odd trailing value is dropped. */
export function outlinePoints(outline: unknown): Pt[] | null {
  if (!Array.isArray(outline)) return null
  const pts: Pt[] = []
  for (let i = 0; i + 1 < outline.length; i += 2) {
    const x = outline[i], y = outline[i + 1]
    if (!finite(x) || !finite(y)) return null
    pts.push([x, y])
  }
  return pts.length >= 3 ? pts : null
}

/** The part of a polygon at or below the line y = `y` (image y grows
 * downwards), clipped Sutherland-Hodgman style; empty when nothing is
 * below it. Drawn as the shaded lower body without an SVG clipPath (whose
 * ids would clash between a tile and its enlarged copy). */
export function clipBelow(points: Pt[], y: number): Pt[] {
  const out: Pt[] = []
  for (let i = 0; i < points.length; i++) {
    const a = points[i]
    const b = points[(i + 1) % points.length]
    const aIn = a[1] >= y
    const bIn = b[1] >= y
    if (aIn) out.push(a)
    if (aIn !== bIn) {
      const t = (y - a[1]) / (b[1] - a[1])
      out.push([a[0] + t * (b[0] - a[0]), y])
    }
  }
  return out.length >= 3 ? out : []
}

export const RULE_COLOR = { occupied: '#1b9e5f', staff_cover: '#3b7fd6', staff: '#8a968f', other: '#d98b1f' } as const
/** Why-values with their own translated label (cam.rule.<why>); the rest of
 * the "not counted" verdicts. */
export const RULE_REASONS = ['low_conf', 'no_mask', 'below_min', 'other_chair', 'no_chair', 'other_view'] as const
export type RuleReason = typeof RULE_REASONS[number]

/** Colour and (for a person who doesn't count) a short reason, by rule.why.
 * An unknown or missing why is drawn orange with no reason. */
export function ruleStyle(why: unknown): { color: string; reason: RuleReason | null } {
  if (why === 'occupied' || why === 'staff_cover' || why === 'staff') return { color: RULE_COLOR[why], reason: null }
  return { color: RULE_COLOR.other, reason: (RULE_REASONS as readonly unknown[]).includes(why) ? why as RuleReason : null }
}

/** "R3 · 23%" when both the chair and the share are known, else null. */
export function ruleLabel(rule: SceneEntry['rule']): string | null {
  if (!rule || !rule.chair || !finite(rule.share)) return null
  return `${rule.chair} · ${Math.round(Math.max(0, Math.min(1, rule.share)) * 100)}%`
}

export interface MaskMark {
  track_id: number
  points: Pt[]
  /** the shaded lower body (empty without a cut line, or nothing below it) */
  lower: Pt[]
  /** the cut line's y, null without one */
  lowerY: number | null
  color: string
  reason: RuleReason | null
  label: string | null
}

/** One mark per scene entry with a usable outline. Without a rule (a camera
 * owning no chair) the mask is drawn plain white, unlabelled. */
export function maskMarks(entries: SceneEntry[]): MaskMark[] {
  const marks: MaskMark[] = []
  for (const e of entries) {
    const points = outlinePoints(e.outline)
    if (!points) continue
    const lowerY = finite(e.lower_y) ? e.lower_y : null
    const style = e.rule ? ruleStyle(e.rule.why) : { color: '#ffffff', reason: null }
    marks.push({ track_id: e.track_id, points, lower: lowerY == null ? [] : clipBelow(points, lowerY), lowerY,
                 color: style.color, reason: style.reason, label: ruleLabel(e.rule) })
  }
  return marks
}

/** What the occupancy layer draws: the mask outlines, and the old anchor
 * marks only for entries that have no outline (an older backend). */
export function occupancyLayer(entries: SceneEntry[]): { masks: MaskMark[]; legacy: OccupancyMark[] } {
  return { masks: maskMarks(entries), legacy: occupancyMarks(entries.filter(e => !outlinePoints(e.outline))) }
}

export const CHAIR_STATE_COLOR: Record<ChairState, string> = {
  empty: '#8a968f', starting: '#e0b422', sitting: '#1b9e5f', away: '#8e5bd0', no_view: '#d64550',
}
const CHAIR_STATES = Object.keys(CHAIR_STATE_COLOR) as ChairState[]

export interface SeatMark {
  id: string
  /** the chair zone the seat belongs to */
  chair: string
  /** the chair's label (its pair, else its name) */
  name: string
  points: number[][]
  state: ChairState
  color: string
  dashed: boolean
  /** seconds in the state (counting up; for away, how long it has been away);
   * null when the backend gave no time */
  seconds: number | null
  /** sitting: how the re-ID check decided this sitting, while fresh (2026-10-18) */
  decision: ResumeDecision | null
  /** sitting: re-ID is deciding (the previous customer back, or a new one) */
  checking: boolean
}

/** The seats of one camera coloured by their chair's state, only on the
 * camera that owns the chair (chairs[chair].camera; without it, the camera
 * whose chair zone owns the pair). Chairs without a state (an older
 * backend) draw nothing. */
export function seatMarks(zones: Zone[], cam: string, chairs: Record<string, ChairOcc> | null | undefined,
                          nowMs: number): SeatMark[] {
  if (!chairs) return []
  const byId = new Map(zones.map(z => [z.id, z]))
  const marks: SeatMark[] = []
  for (const s of zones) {
    if (s.type !== 'seat' || !s.seat_of || !Array.isArray(s.points) || s.points.length < 3) continue
    const cz = byId.get(s.seat_of)
    if (!cz || cz.type !== 'chair') continue
    const key = cz.pair ?? cz.id
    const ch = chairs[key]
    if (!ch || !ch.state || !CHAIR_STATES.includes(ch.state)) continue
    // a "starting" younger than 3 s is shown as it was before: empty
    const state = effectiveState(ch, nowMs) as ChairState
    const shown = state === ch.state ? ch : { ...ch, state, since: null }
    const owner = ch.camera ? ch.camera === cam : (!!cz.owner || !cz.pair)
    if (!owner) continue
    let seconds: number | null = null
    // a haircut in the chair times from its own start (2026-10-18): since restarts with each sitting, so a resumed
    // haircut read 00:00 here while its card went on
    const from = state === 'sitting' ? shown.haircut_since ?? shown.since : shown.since
    if (state !== 'empty' && from) {
      const since = Date.parse(from)
      if (Number.isFinite(since)) seconds = Math.max(0, (nowMs - since) / 1000)
    }
    const checking = state === 'sitting' && !!ch.checking
    marks.push({ id: s.id, chair: cz.id, name: cz.pair || cz.name || cz.id, points: s.points, state: state,
                 color: CHAIR_STATE_COLOR[state], dashed: state === 'no_view', seconds,
                 decision: state === 'sitting' && !checking ? freshDecision(ch, nowMs) : null, checking })
  }
  return marks
}

const ABOVE_GAP = 3

/** Where an h-high tag goes on a zone (2026-10-18, user: "L6 · sitting" in the middle of the chair hid the chair;
 * then: beside it was bad, right on top of the zone): centred over the zone's box, its bottom ABOVE_GAP above the
 * zone; just under the zone when the picture ends above it. x is the tag's centre. */
export function aboveZone(points: number[][], rect: Rect, h: number): { x: number; y: number; anchor: 'middle' } {
  const xs = points.map(p => rect.x + p[0] * rect.w)
  const ys = points.map(p => rect.y + p[1] * rect.h)
  const x = (Math.min(...xs) + Math.max(...xs)) / 2
  const above = Math.min(...ys) - ABOVE_GAP - h
  return { x, y: above >= rect.y + 2 ? above : Math.max(...ys) + ABOVE_GAP, anchor: 'middle' }
}

/** Whether the People layer draws a person's box. With the Occupancy layer
 * on, a person whose mask outline is drawn there loses the box (the outline
 * replaces it, the label stays); without a usable outline, or with
 * Occupancy off, the box stays. */
export function drawBox(entry: Pick<SceneEntry, 'outline'>, occupancyOn: boolean): boolean {
  return !(occupancyOn && outlinePoints(entry.outline) != null)
}

/** When a customer's box label starts counting: a haircut from the haircut's start (2026-10-18: a resume goes on,
 * like its chair card), any other state from that state's start. */
export function labelSince(c: Pick<LiveCustomer, 'state' | 'state_since' | 'haircut_started_at'>): string | null {
  return c.state === 'haircut' ? c.haircut_started_at ?? c.state_since : c.state_since
}
