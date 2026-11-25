import type { Candidate } from '../api'
import { channels } from './severity'

export function unseenCount(
  rows: ReadonlyArray<Pick<Candidate, 'id' | 'severity' | 'status'>>,
  seenCursor: number,
): number {
  return rows.filter(r => r.id > seenCursor && channels(r).badge).length
}
