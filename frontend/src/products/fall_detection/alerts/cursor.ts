/** The feed's cursor: the newest event id the tab has seen. */

/** The cursor only moves forward, and an empty batch never moves it. */
export function advance(cursor: number, batch: { id: number }[]): number {
  if (!batch.length) return cursor
  return Math.max(cursor, ...batch.map(e => e.id))
}
