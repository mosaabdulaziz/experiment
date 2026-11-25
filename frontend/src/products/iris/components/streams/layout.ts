/**
 * Mosaic layout for the video wall: arrange N tiles to fill a container of the
 * given size with no gaps and no scrolling. Columns are derived from the count
 * *and* the container's aspect ratio, so a wide area gets wider rows and a tall
 * area gets more rows — keeping each cell close to landscape (16:9) video shape.
 */
export function chooseColumns(count: number, width: number, height: number): number {
  if (count <= 1) return 1
  const aspect = width > 0 && height > 0 ? width / height : 16 / 9
  const cols = Math.round(Math.sqrt(count * aspect))
  return Math.max(1, Math.min(count, cols))
}

/** Split items into `rows` rows as evenly as possible (earlier rows take the remainder). */
export function splitIntoRows<T>(items: T[], rows: number): T[][] {
  const r = Math.max(1, rows)
  const base = Math.floor(items.length / r)
  const extra = items.length % r
  const out: T[][] = []
  let i = 0
  for (let row = 0; row < r; row++) {
    const size = base + (row < extra ? 1 : 0)
    out.push(items.slice(i, i + size))
    i += size
  }
  return out
}

/** Rows of tiles sized to fill a `width`×`height` container. */
export function layoutRows<T>(items: T[], width: number, height: number): T[][] {
  const cols = chooseColumns(items.length, width, height)
  const rows = Math.ceil(items.length / cols)
  return splitIntoRows(items, rows)
}

/** Rows of exactly `cols` tiles each (last row may be short) for the scrollable
 *  video wall. Off-screen rows are unmounted, so only the on-screen tiles poll
 *  their snapshot endpoints. */
export function scrollGridRows<T>(items: T[], cols: number): T[][] {
  const c = Math.max(1, cols)
  const out: T[][] = []
  for (let i = 0; i < items.length; i += c) {
    out.push(items.slice(i, i + c))
  }
  return out
}
