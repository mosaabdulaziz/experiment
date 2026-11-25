export function advance(cursor: number, batch: ReadonlyArray<{ id: number }>): number {
  return batch.reduce((m, e) => (e.id > m ? e.id : m), cursor)
}
