/** What the engine sees, drawn over a camera picture: the zones and the
 * people it tracks. Labels use the Shop board's words (#ticket, Staff), so a
 * box can be matched to a board card at a glance. All geometry is spatial
 * and stays left-to-right in Arabic too. */
import { useRef, type CSSProperties } from 'react'
import type { ChairOcc, CustomerState, LiveCustomer, ResumeDecision, Scene, Zone } from '../../api/types'
import { useLang } from '../../i18n'
import { aboveZone, drawBox, labelSince, occupancyLayer, personColor, pushTrail, seatMarks, trackLabel, type TrailBuf } from '../../live/overlayModel'
import { boxKind, ticketLabel } from '../../live/selectors'
import { useStore, type Store } from '../../live/store'
import type { Rect } from '../../ui/Feed'
import { elapsedSince, fmtTimer } from '../../ui/format'
import { centroid, insideNormal } from '../../ui/geometry'

export const DOOR = '#14a05e'
const STAFF = '#c3ccc7'
const LABEL_H = 17
const LABEL_FONT = 11.5
/** 2026-10-18 (user): the Staff tags and the zone names in a smaller chip, so they cover less of the picture */
const SMALL_H = 14
const SMALL_FONT = 9.5
/** Tiles narrower than this drop the elapsed times from labels, which overlap there. */
const COMPACT_W = 420

/** Stroke for what a zone is for: the state a customer there is in. */
export const ZONE_STROKE: Record<string, string> = {
  waiting: 'var(--st-waiting)',
  chair: 'var(--st-haircut)',
  washing: 'var(--st-washing)',
  seat: 'var(--ink-mute)',
  row: 'var(--zone-row)',
  door_area: 'var(--zone-door-area)',
  face_area: 'var(--zone-face-area)',
}

const STATE_STROKE: Record<CustomerState, string> = {
  waiting: 'var(--st-waiting)',
  haircut: 'var(--st-haircut)',
  washing: 'var(--st-washing)',
  served: 'var(--st-finished)',
  finished: 'var(--st-finished)',
}

let measureCtx: CanvasRenderingContext2D | null | undefined
/** Width of a label in the overlay font (Arabic glyphs are much wider than
 * a per-character guess); falls back to an estimate without a canvas. */
function textWidth(text: string, font = LABEL_FONT): number {
  if (measureCtx === undefined) {
    measureCtx = typeof document !== 'undefined' ? document.createElement('canvas').getContext('2d') : null
  }
  if (!measureCtx) return text.length * 6.6 * (font / LABEL_FONT)
  measureCtx.font = `700 ${font}px Cairo, sans-serif`
  return measureCtx.measureText(text).width
}

/** A chip's width for this text */
const chipWidth = (text: string, small = false) => Math.ceil(textWidth(text, small ? SMALL_FONT : LABEL_FONT)) + 10

/** A chip: dark text on a translucent white pill, readable over video; kept
 * inside the picture so a label at the edge isn't clipped. */
export function OvLabel({ x, y, text, color, anchor = 'start', bounds, small = false }: {
  x: number; y: number; text: string; color: string; anchor?: 'start' | 'middle' | 'end'; bounds: Rect; small?: boolean
}) {
  const w = chipWidth(text, small)
  const h = small ? SMALL_H : LABEL_H
  const font = small ? SMALL_FONT : LABEL_FONT
  let left = anchor === 'middle' ? x - w / 2 : anchor === 'end' ? x - w : x
  left = Math.max(bounds.x + 2, Math.min(left, bounds.x + bounds.w - w - 2))
  const top = Math.max(bounds.y + 2, Math.min(y, bounds.y + bounds.h - h - 2))
  return (
    <g pointerEvents="none">
      <rect x={left} y={top} width={w} height={h} rx={4} fill="rgba(255,255,255,0.88)" />
      <text x={left + 5} y={top + h - (small ? 4 : 5)} direction="ltr" className="ovLabel"
            style={{ fill: color, fontSize: font }}>{text}</text>
    </g>
  )
}

const poly = (points: number[][], r: Rect) => points.map(([x, y]) => `${r.x + x * r.w},${r.y + y * r.h}`).join(' ')

/** Zones of one camera: a zone owned here solid, a paired copy (the same
 * area as another camera owns it) dashed, seats thin, the door line with an
 * arrow into the shop. Names in a small chip; a chair's right on top of its zone, not
 * on the chair; none for the zones in `unnamed`. */
export function ZoneOverlay({ zones, rect, unnamed }: { zones: Zone[]; rect: Rect; unnamed?: ReadonlySet<string> }) {
  const px = (p: number[]) => [rect.x + p[0] * rect.w, rect.y + p[1] * rect.h] as const
  const seats = zones.filter(z => z.type === 'seat')
  const areas = zones.filter(z => z.type !== 'seat' && z.type !== 'entrance_line')
  const doors = zones.filter(z => z.type === 'entrance_line')
  return (
    <g>
      {[...seats, ...areas].map(z => {
        if (z.points.length < 3) return null
        const stroke = ZONE_STROKE[z.type] ?? '#fff'
        const seat = z.type === 'seat'
        const style: CSSProperties = { stroke, fill: seat ? 'none' : stroke, fillOpacity: 0.1 }
        return (
          <polygon key={z.id} points={poly(z.points, rect)} style={style}
                   strokeWidth={seat ? 1.25 : 2.25} strokeDasharray={z.pair && !z.owner ? '6 4' : undefined} />
        )
      })}
      {doors.map(z => {
        if (z.points.length < 2) return null
        const [a, b] = z.points
        const [ax, ay] = px(a)
        const [bx, by] = px(b)
        const [mx, my] = px([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2])
        let arrow = null
        if (z.inside) {
          const [nx, ny] = insideNormal(a, b, z.inside)
          // the normal is in normalized units: scale to pixels, a fixed 26 px arrow
          const len = Math.hypot(nx * rect.w, ny * rect.h) || 1
          const ux = (nx * rect.w) / len
          const uy = (ny * rect.h) / len
          const tx = mx + ux * 26
          const ty = my + uy * 26
          arrow = (
            <g stroke={DOOR} strokeWidth={2.5} fill="none" strokeLinecap="round">
              <line x1={mx} y1={my} x2={tx} y2={ty} />
              <polyline points={`${tx - ux * 7 - uy * 5},${ty - uy * 7 + ux * 5} ${tx},${ty} ${tx - ux * 7 + uy * 5},${ty - uy * 7 - ux * 5}`} />
            </g>
          )
        }
        return (
          <g key={z.id}>
            <line x1={ax} y1={ay} x2={bx} y2={by} stroke={DOOR} strokeWidth={3} strokeLinecap="round" />
            {arrow}
          </g>
        )
      })}
      {/* names last, so no polygon covers them; seats stay unnamed (they sit on their chair) */}
      {[...areas, ...doors].map(z => {
        if (!z.points.length || !z.name || unnamed?.has(z.id)) return null
        if (z.type === 'chair' && z.points.length >= 3) {
          const at = aboveZone(z.points, rect, SMALL_H)
          return <OvLabel key={`n-${z.id}`} {...at} text={z.name} color="#17251f" bounds={rect} small />
        }
        const [cx, cy] = px(centroid(z.points))
        return <OvLabel key={`n-${z.id}`} x={cx} y={cy - SMALL_H / 2} anchor="middle" text={z.name} color="#17251f" bounds={rect} small />
      })}
    </g>
  )
}

/** The zones with the Chairs layer on: a chair whose state is drawn on this camera loses its zone name, its state tag
 * already names it ("L6 · sitting"). `camZones` unfiltered: the seats decide where a state is drawn. */
export function ZoneOverlayWithChairs({ zones, camZones, rect, chairs, cam, now }: {
  zones: Zone[]; camZones: Zone[]; rect: Rect; chairs: Store<Record<string, ChairOcc>>; cam: string; now: number
}) {
  const marked = new Set(seatMarks(camZones, cam, useStore(chairs), now).map(m => m.chair))
  return <ZoneOverlay zones={zones} rect={rect} unnamed={marked} />
}

/** People on one camera, fed straight from the scene store: the only part
 * of the page that re-renders on each scene message. Customers are boxed in
 * their state colour with #ticket (and the time in that state when known),
 * staff grey with "Staff", anyone not yet known (or not yet a confirmed
 * visit) thin white. */
export function PeopleOverlay({ store, cam, rect, now, customers = [], occupancyOn = false }: {
  store: Store<Scene>
  cam: string
  rect: Rect
  now: number
  customers?: LiveCustomer[]
  /** the Occupancy layer is on: people with a mask outline lose their box */
  occupancyOn?: boolean
}) {
  const { t } = useLang()
  const entries = useStore(store)[cam] ?? []
  const since = new Map(customers.map(c => [c.person_id, labelSince(c)]))
  return (
    <g>
      {entries.map(e => {
        if (!e.box) return null
        const [l, tp, w, h] = e.box
        const x = rect.x + l * rect.w
        const y = rect.y + tp * rect.h
        const bw = w * rect.w
        const bh = h * rect.h
        const customer = boxKind(e) === 'customer'
        const stroke = e.staff ? STAFF : customer ? STATE_STROKE[e.state as CustomerState] : 'rgba(255,255,255,0.8)'
        let label: string | null = null
        let color = '#17251f'
        if (e.staff) { label = t('cam.staff'); color = '#4d5a53' }
        else if (customer) {
          const s = since.get(e.person_id!)
          label = s && rect.w >= COMPACT_W ? `${ticketLabel(e.ticket)} ${fmtTimer(elapsedSince(s, now))}` : ticketLabel(e.ticket)
          color = stroke
        }
        return (
          <g key={e.track_id}>
            {drawBox(e, occupancyOn) && (
              <rect x={x} y={y} width={bw} height={bh} rx={3} fill="none" style={{ stroke }}
                    strokeWidth={e.staff || customer ? 2.5 : 1.25} />
            )}
            {/* just above the box: people boxes are small in the wide views */}
            {label && <OvLabel x={x} y={y - (e.staff ? SMALL_H : LABEL_H) - 2} text={label} color={color} bounds={rect} small={!!e.staff} />}
          </g>
        )
      })}
    </g>
  )
}

const OCC_COUNTED = '#1b9e5f'
const OCC_REJECTED = '#d98b1f'

/** How chair occupancy is decided, drawn on the picture. With the chair
 * engine: each person's mask outline, the lower body (below the cut line)
 * shaded, coloured by the rule's verdict -- green occupied, blue staff
 * cover, grey staff, orange anything else with its reason -- and "R3 · 23%"
 * (the chair tested and how much of its seat the lower body covers).
 * Entries from an older backend (no outline) keep the old drawing: the
 * anchor -- filled when it counts, a hollow ring with the reason otherwise
 * -- the seat line it's tested against, and the foot point. */
export function OccupancyOverlay({ store, cam, rect }: { store: Store<Scene>; cam: string; rect: Rect }) {
  const { t } = useLang()
  const entries = useStore(store)[cam] ?? []
  const { masks, legacy } = occupancyLayer(entries)
  const px = (p: [number, number]) => [rect.x + p[0] * rect.w, rect.y + p[1] * rect.h] as const
  const pts = (points: [number, number][]) => points.map(p => px(p).join(',')).join(' ')
  return (
    <g pointerEvents="none">
      {masks.map(m => {
        const text = [m.label, m.reason && t(`cam.rule.${m.reason}`)].filter(Boolean).join(' · ')
        const left = Math.min(...m.points.map(p => p[0]))
        const top = m.lowerY ?? Math.min(...m.points.map(p => p[1]))
        const [lx, ly] = px([left, top])
        return (
          <g key={m.track_id}>
            <polygon points={pts(m.points)} fill="none" stroke={m.color} strokeWidth={1.75} strokeLinejoin="round" />
            {m.lower.length > 0 && <polygon points={pts(m.lower)} fill={m.color} fillOpacity={0.35} stroke="none" />}
            {m.lowerY != null && m.lower.length > 0 && (() => {
              const xs = m.lower.map(p => p[0])
              const [ax, ay] = px([Math.min(...xs), m.lowerY])
              const [bx] = px([Math.max(...xs), m.lowerY])
              return <line x1={ax} y1={ay} x2={bx} y2={ay} stroke={m.color} strokeWidth={1.5} strokeDasharray="4 3" />
            })()}
            {text && <OvLabel x={lx} y={ly - LABEL_H - 2} text={text}
                              color={m.color === '#ffffff' ? '#4d5a53' : m.color} bounds={rect} />}
          </g>
        )
      })}
      {legacy.map(m => {
        const [x, y] = px(m.anchor)
        const color = m.counted ? OCC_COUNTED : OCC_REJECTED
        const [fx, fy] = m.foot ? px(m.foot) : [null, null]
        return (
          <g key={m.track_id}>
            {m.seatLine && (() => {
              const [ax, ay] = px(m.seatLine[0])
              const [bx, by] = px(m.seatLine[1])
              return <line x1={ax} y1={ay} x2={bx} y2={by} stroke={color} strokeWidth={1.5} strokeDasharray="4 3" />
            })()}
            <circle cx={x} cy={y} r={5} fill={m.counted ? color : 'none'} stroke={color} strokeWidth={2} />
            {fx != null && <circle cx={fx} cy={fy!} r={2} fill="#fff" />}
            {!m.counted && m.reason && <OvLabel x={x + 8} y={y - 8} text={t(`cam.why.${m.reason}`)} color="#8a5a10" bounds={rect} />}
          </g>
        )
      })}
    </g>
  )
}

/** The re-ID check's answer under a chair's tag (2026-10-18): "resumed 0.84" / "same customer 0.95" (after cover) /
 * "new customer 0.41", "(by time)" without a score; on a small tile only the sign and the score. */
function decisionLabel(d: ResumeDecision, compact: boolean, t: ReturnType<typeof useLang>['t']): string {
  const sim = d.sim == null ? null : d.sim.toFixed(2)
  if (compact) return `${d.resumed ? '↺' : '+'}${sim != null ? ` ${sim}` : ''}`
  if (sim == null) return t(d.resumed ? 'cam.chair.resumedTime' : 'cam.chair.newTime')
  return t(!d.resumed ? 'cam.chair.newCustomer' : d.kind === 'reseat' ? 'cam.chair.sameCustomer' : 'cam.chair.resumed', { n: sim })
}

/** Each chair's state on the seat it is decided by, on the camera owning
 * the chair: grey empty, yellow starting, green sitting, purple away (the
 * customer stepped out), red dashed no view; tagged "R3 · sitting 12:34" (time in the
 * state; for away, how long it has been away) right on top of the seat, so the chair stays in sight. Draws nothing with an
 * older backend (no chair state). */
export function ChairsOverlay({ chairs, zones, cam, rect, now }: {
  chairs: Store<Record<string, ChairOcc>>; zones: Zone[]; cam: string; rect: Rect; now: number
}) {
  const { t } = useLang()
  const marks = seatMarks(zones, cam, useStore(chairs), now)
  return (
    <g pointerEvents="none">
      {marks.map(m => {
        const timer = m.seconds == null || rect.w < COMPACT_W ? '' : m.state === 'away' ? ` ${t('strip.awayFor', { n: fmtTimer(m.seconds) })}` : ` ${fmtTimer(m.seconds)}`
        const text = `${m.name} · ${t(`cam.chair.${m.state}`)}${timer}`
        const second = m.checking ? t('cam.chair.checking') : m.decision ? decisionLabel(m.decision, rect.w < COMPACT_W, t) : null
        // the two tags as one block on top of the seat, the second under the first
        const at = aboveZone(m.points, rect, second ? 2 * LABEL_H + 2 : LABEL_H)
        return (
          <g key={m.id}>
            <polygon points={poly(m.points, rect)} fill={m.color} fillOpacity={m.state === 'empty' ? 0.12 : 0.3}
                     stroke={m.color} strokeWidth={2.25} strokeDasharray={m.dashed ? '6 4' : undefined} />
            <OvLabel {...at} bounds={rect} text={text}
                     color={m.state === 'empty' ? '#4d5a53' : m.state === 'starting' ? '#8a6a00' : m.color} />
            {second && (
              <OvLabel {...at} y={at.y + LABEL_H + 2} bounds={rect} text={second}
                       color={m.checking ? '#1f6fa8' : m.decision!.resumed ? '#1d6b3f' : '#1f5fa8'} />
            )}
          </g>
        )
      })}
    </g>
  )
}

/** Each tracked person's last 10 s of anchors, as a trail: the same colour
 * for the same person on every camera (white while unbound), the label
 * `#track → <person>` once matched, else just `#track`. The buffer lives in
 * a ref so it survives across scene messages without extra re-renders of
 * its own. */
export function TracksOverlay({ store, cam, rect }: { store: Store<Scene>; cam: string; rect: Rect }) {
  const entries = useStore(store)[cam] ?? []
  const bufRef = useRef<TrailBuf>({})
  bufRef.current = pushTrail(bufRef.current, cam, entries, Date.now())
  const trails = bufRef.current[cam] ?? {}
  const byTrack = new Map(entries.map(e => [e.track_id, e]))
  const px = (p: { x: number; y: number }) => [rect.x + p.x * rect.w, rect.y + p.y * rect.h] as const
  return (
    <g pointerEvents="none">
      {Object.entries(trails).map(([key, points]) => {
        if (!points.length) return null
        const trackId = Number(key)
        const personId = byTrack.get(trackId)?.person_id ?? null
        const color = personColor(personId)
        // the label sits on a translucent white chip: white trail colour would vanish there
        const labelColor = personId ? color : '#4d5a53'
        const last = px(points[points.length - 1])
        return (
          <g key={trackId}>
            {points.length > 1 && (
              <polyline points={points.map(p => px(p).join(',')).join(' ')} fill="none" stroke={color}
                        strokeWidth={2} opacity={0.8} />
            )}
            <OvLabel x={last[0] + 6} y={last[1] - 6} text={trackLabel(trackId, personId)} color={labelColor} bounds={rect} />
          </g>
        )
      })}
    </g>
  )
}
