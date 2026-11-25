/** The width of a side panel whose inner edge is dragged (ResizeHandle). The panel sits at the inline end of its row --
 *  on the right in a left-to-right page, on the left in a right-to-left one -- so moving the edge toward the inline
 *  start widens it. Pure, so it is tested off the DOM. */

export interface WidthLimits {
  min: number
  max: number
}

export function clampWidth(width: number, { min, max }: WidthLimits): number {
  return Math.round(Math.min(max, Math.max(min, width)))
}

/** The width while the edge, grabbed at `startX` with the panel `start` wide, is at `x`. */
export function dragWidth(start: number, startX: number, x: number, rtl: boolean, limits: WidthLimits): number {
  return clampWidth(start + (rtl ? x - startX : startX - x), limits)
}

/** The width after a key on the edge: the arrows move it a step (wider when it moves away from the panel), Home and
 *  End jump to the narrowest and widest. Null for any other key. */
export function keyWidth(width: number, key: string, rtl: boolean, limits: WidthLimits, step = 24): number | null {
  if (key === 'Home') return limits.min
  if (key === 'End') return limits.max
  if (key !== 'ArrowLeft' && key !== 'ArrowRight') return null
  const wider = (key === 'ArrowLeft') !== rtl
  return clampWidth(width + (wider ? step : -step), limits)
}
