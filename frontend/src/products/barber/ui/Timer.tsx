import { elapsedSince, fmtTimer, levelOf } from './format'
import { Num } from './Num'

/** A running mm:ss (h:mm:ss past an hour) since `since`, on the server clock,
 * coloured by level against `target` (none -> always neutral). */
export function Timer({ since, now, target, className = '' }: { since: string; now: number; target: number | null | undefined; className?: string }) {
  const s = elapsedSince(since, now)
  return <Num className={`timer lvl-${levelOf(s, target)} ${className}`.trim()}>{fmtTimer(s)}</Num>
}
