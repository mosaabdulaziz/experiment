/** The zone editor's drawing surface: a camera still with this camera's
 * zones drawn over it, the selected zone with its vertex handles, and the
 * zone being drawn. Pointer input only -- keys are handled by ZoneEditor.
 * Spatial: left-to-right in Arabic too (the pointer math relies on it). */
import { useRef, type PointerEvent as ReactPointerEvent, type MouseEvent as ReactMouseEvent } from 'react'
import type { Zone } from '../../api/types'
import { Feed, type Rect } from '../../ui/Feed'
import { clamp01 } from '../../ui/geometry'
import { DOOR, ZONE_STROKE, ZoneOverlay } from '../live/Overlay'

type Pt = [number, number]
export type Handle = number | 'inside'

const GRAB_PX = 11

export function inPolygon(x: number, y: number, pts: number[][]): boolean {
  let inside = false
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i]
    const [xj, yj] = pts[j]
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / ((yj - yi) || 1e-9) + xi) inside = !inside
  }
  return inside
}

function polygonArea(pts: number[][]): number {
  let a = 0
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) a += (pts[j][0] + pts[i][0]) * (pts[j][1] - pts[i][1])
  return Math.abs(a / 2)
}

function segDistPx(p: number[], a: number[], b: number[], r: Rect): number {
  const [px, py] = [p[0] * r.w, p[1] * r.h]
  const [ax, ay] = [a[0] * r.w, a[1] * r.h]
  const [bx, by] = [b[0] * r.w, b[1] * r.h]
  const dx = bx - ax
  const dy = by - ay
  const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / ((dx * dx + dy * dy) || 1)))
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy))
}

/** The zone under a point: a door line near it, else the smallest polygon
 * holding it (a seat sits inside its chair and must stay clickable). */
export function zoneAt(zones: Zone[], p: number[], r: Rect): Zone | null {
  const line = zones.find(z => z.type === 'entrance_line' && z.points.length === 2 && segDistPx(p, z.points[0], z.points[1], r) <= GRAB_PX)
  if (line) return line
  const hits = zones.filter(z => z.type !== 'entrance_line' && z.points.length >= 3 && inPolygon(p[0], p[1], z.points))
  hits.sort((a, b) => polygonArea(a.points) - polygonArea(b.points))
  return hits[0] ?? null
}

export function ZoneSurface({ src, zones, drawing, selectedId, onAddPoint, onFinish, onSelect, onMove }: {
  src: string | null
  zones: Zone[]
  drawing: Zone | null
  selectedId: string | null
  onAddPoint: (p: Pt) => void
  onFinish: () => void
  onSelect: (id: string | null) => void
  onMove: (id: string, handle: Handle, p: Pt) => void
}) {
  const rectRef = useRef<Rect>({ x: 0, y: 0, w: 1, h: 1 })
  const drag = useRef<{ id: string; handle: Handle } | null>(null)
  const selected = zones.find(z => z.id === selectedId) ?? null

  const toNorm = (e: ReactPointerEvent<Element> | ReactMouseEvent<Element>): Pt => {
    const svg = (e.currentTarget as SVGElement).ownerSVGElement ?? (e.currentTarget as SVGSVGElement)
    const b = svg.getBoundingClientRect()
    const r = rectRef.current
    return [clamp01((e.clientX - b.left - r.x) / r.w), clamp01((e.clientY - b.top - r.y) / r.h)]
  }
  const near = (a: number[], b: number[]) => {
    const r = rectRef.current
    return Math.hypot((a[0] - b[0]) * r.w, (a[1] - b[1]) * r.h) <= GRAB_PX
  }
  const handleAt = (p: number[]): { id: string; handle: Handle } | null => {
    if (!selected) return null
    if (selected.type === 'entrance_line' && selected.inside && near(p, selected.inside)) return { id: selected.id, handle: 'inside' }
    const i = selected.points.findIndex(q => near(p, q))
    return i >= 0 ? { id: selected.id, handle: i } : null
  }

  const onPointerDown = (e: ReactPointerEvent<SVGRectElement>) => {
    if (drawing || e.button !== 0) return
    const p = toNorm(e)
    const h = handleAt(p)
    if (h) {
      drag.current = h
      e.currentTarget.setPointerCapture(e.pointerId)
      return
    }
    onSelect(zoneAt(zones, p, rectRef.current)?.id ?? null)
  }
  const onPointerMove = (e: ReactPointerEvent<SVGRectElement>) => {
    if (drag.current) onMove(drag.current.id, drag.current.handle, toNorm(e))
  }
  const onPointerUp = (e: ReactPointerEvent<SVGRectElement>) => {
    if (!drag.current) return
    drag.current = null
    e.currentTarget.releasePointerCapture(e.pointerId)
  }
  const onClick = (e: ReactMouseEvent<SVGRectElement>) => {
    // the second click of a double-click is the finish gesture, not a point
    if (drawing && e.detail < 2) onAddPoint(toNorm(e))
  }
  const onDoubleClick = () => { if (drawing) onFinish() }

  return (
    <Feed src={src} className="feedEditing">
      {rect => {
        rectRef.current = rect
        const px = (p: number[]) => [rect.x + p[0] * rect.w, rect.y + p[1] * rect.h] as const
        const others = zones.filter(z => z.id !== selectedId)
        const stroke = (z: Zone) => (z.type === 'entrance_line' ? DOOR : ZONE_STROKE[z.type] ?? '#fff')
        return (
          <>
            <ZoneOverlay zones={others} rect={rect} />
            {selected && (
              <g pointerEvents="none">
                <ZoneOverlay zones={[selected]} rect={rect} />
                {selected.points.length >= 3 && (
                  <polygon points={selected.points.map(p => px(p).join(',')).join(' ')} fill="none"
                           stroke="#fff" strokeWidth={1} strokeDasharray="3 3" />
                )}
                {selected.points.map((p, i) => {
                  const [x, y] = px(p)
                  return <circle key={i} cx={x} cy={y} r={6} fill="#fff" style={{ stroke: stroke(selected) }} strokeWidth={2.5} />
                })}
                {selected.type === 'entrance_line' && selected.inside && (
                  <circle cx={px(selected.inside)[0]} cy={px(selected.inside)[1]} r={5.5} fill={DOOR} stroke="#fff" strokeWidth={2} />
                )}
              </g>
            )}
            {drawing && drawing.points.length > 0 && (
              <g pointerEvents="none" style={{ stroke: stroke(drawing) }}>
                <polyline points={drawing.points.map(p => px(p).join(',')).join(' ')} fill="none" strokeWidth={2.25} strokeDasharray="6 4" />
                {drawing.points.map((p, i) => {
                  const [x, y] = px(p)
                  return <circle key={i} cx={x} cy={y} r={i === 0 ? 6 : 4.5} fill="#fff" strokeWidth={2} />
                })}
              </g>
            )}
            <rect x={0} y={0} width="100%" height="100%" fill="transparent" className="zoneHit"
                  onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp}
                  onPointerCancel={onPointerUp} onClick={onClick} onDoubleClick={onDoubleClick} />
          </>
        )
      }}
    </Feed>
  )
}
