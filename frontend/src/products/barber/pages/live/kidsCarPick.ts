/** The kids car override (2026-10-15): where a click lands on the drawn picture, in normalised frame coordinates. */
import type { Rect } from '../../ui/Feed'

/** `rect`: the picture inside its box (Feed's content rect); `box`: the box's place on the page. Null = beside it. */
export function pointIn(rect: Rect, clientX: number, clientY: number, box: { left: number; top: number }): [number, number] | null {
  const x = clientX - box.left - rect.x
  const y = clientY - box.top - rect.y
  if (!rect.w || !rect.h || x < 0 || y < 0 || x > rect.w || y > rect.h) return null
  return [x / rect.w, y / rect.h]
}
