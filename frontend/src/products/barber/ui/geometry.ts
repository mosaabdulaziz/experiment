/** Geometry shared by the live overlay and the zone editor. Everything is in
 * normalized 0..1 frame coordinates, like barber-service's zones and scene. */

export type Pt = [number, number]

/** Where an image drawn with `object-fit: contain` actually sits inside its
 * box -- overlays must use this rect, or a camera whose aspect differs from
 * the box would have its zones drawn off the picture. */
export function contentRect(boxW: number, boxH: number, imgW: number, imgH: number) {
  if (!imgW || !imgH || !boxW || !boxH) return { x: 0, y: 0, w: boxW, h: boxH }
  const scale = Math.min(boxW / imgW, boxH / imgH)
  const w = imgW * scale
  const h = imgH * scale
  return { x: (boxW - w) / 2, y: (boxH - h) / 2, w, h }
}

export function centroid(pts: number[][]): Pt {
  const n = pts.length || 1
  return [pts.reduce((a, p) => a + p[0], 0) / n, pts.reduce((a, p) => a + p[1], 0) / n]
}

/** The polygon scaled about its centroid by (1 + margin) -- the same rule
 * barber-service uses for a chair's proximity area (zones.expand). */
export function expand(pts: number[][], margin: number): number[][] {
  const [cx, cy] = centroid(pts)
  const k = 1 + margin
  return pts.map(([x, y]) => [cx + (x - cx) * k, cy + (y - cy) * k])
}

/** Unit normal of p1->p2 pointing to the side `inside` is on (for the arrow). */
export function insideNormal(p1: number[], p2: number[], inside: number[]): Pt {
  const dx = p2[0] - p1[0]
  const dy = p2[1] - p1[1]
  const len = Math.hypot(dx, dy) || 1
  let nx = -dy / len
  let ny = dx / len
  const side = (inside[0] - p1[0]) * nx + (inside[1] - p1[1]) * ny
  if (side < 0) { nx = -nx; ny = -ny }
  return [nx, ny]
}

export function clamp01(v: number): number {
  return Math.min(1, Math.max(0, v))
}
