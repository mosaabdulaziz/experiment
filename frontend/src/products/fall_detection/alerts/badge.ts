/** The rail badge: the alert-worthy events the operator has not looked at yet.
 *
 * A client-side counter over the feed's ids -- the product has no operator session, and a
 * server-side unread flag would be a second source of truth (phase-5 §3).
 */
import { channels, type FallEvent } from './severity'

export function unseenCount(events: FallEvent[], seenCursor: number): number {
  return events.filter(e => e.id > seenCursor && channels(e).badge).length
}
