/** Time formats and the target-status rule shared by every screen.
 *
 * Times from barber-service are naive local ISO strings ("2026-09-14T13:41:05").
 * Clock times are shown straight from the string (no timezone conversion);
 * elapsed times are measured against the server's clock (see live/store). */

/** Elapsed time: `mm:ss` under an hour, `h:mm:ss` from an hour on (2026-10-11, user: `h:mm` read as
 * minutes:seconds, a 1 h haircut's "1:01" looked like a timer stuck at a minute). */
export function fmtTimer(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return '—'
  const s = Math.max(0, Math.floor(seconds))
  const p = (n: number) => String(n).padStart(2, '0')
  if (s >= 3600) return `${Math.floor(s / 3600)}:${p(Math.floor((s % 3600) / 60))}:${p(s % 60)}`
  return `${p(Math.floor(s / 60))}:${p(s % 60)}`
}

/** "13:41" from an ISO datetime. */
export function fmtClock(iso: string | null | undefined): string {
  return iso ? iso.slice(11, 16) : '—'
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

/** "14 Sep" from an ISO date or datetime. */
export function fmtDay(iso: string | null | undefined): string {
  if (!iso) return '—'
  return `${Number(iso.slice(8, 10))} ${MONTHS[Number(iso.slice(5, 7)) - 1]}`
}

/** Seconds between an ISO datetime and a server-clock instant (ms). */
export function elapsedSince(iso: string, serverNowMs: number): number {
  return Math.max(0, (serverNowMs - Date.parse(iso)) / 1000)
}

export function secondsBetween(fromIso: string | null, toIso: string | null): number | null {
  if (!fromIso || !toIso) return null
  return Math.max(0, (Date.parse(toIso) - Date.parse(fromIso)) / 1000)
}

/** Neutral when normal, amber from WARN_AT of the target, red once over it. */
export type Level = 'ok' | 'warn' | 'over'
export const WARN_AT = 0.85

export function levelOf(seconds: number | null | undefined, target: number | null | undefined): Level {
  if (seconds === null || seconds === undefined || !target) return 'ok'
  if (seconds > target) return 'over'
  return seconds >= WARN_AT * target ? 'warn' : 'ok'
}

/** "14:32:08" -- the server clock in the top bar. */
export function fmtClockSeconds(ms: number): string {
  const d = new Date(ms)
  const p = (n: number) => String(n).padStart(2, '0')
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
}

/** A position in a video: "12:34", or "1:02:03" past an hour. */
export function fmtVideoTime(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds))
  const p = (n: number) => String(n).padStart(2, '0')
  return s >= 3600 ? `${Math.floor(s / 3600)}:${p(Math.floor((s % 3600) / 60))}:${p(s % 60)}` : `${Math.floor(s / 60)}:${p(s % 60)}`
}

/** YYYY-MM-DD for <input type="date">. */
export function isoDate(d: Date): string {
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`
}

export function addDays(isoDay: string, days: number): string {
  const d = new Date(`${isoDay}T00:00:00`)
  d.setDate(d.getDate() + days)
  return isoDate(d)
}

/** mm:ss text <-> seconds, for the target inputs. */
export function parseTimer(text: string): number | null {
  const m = /^\s*(\d{1,3}):([0-5]\d)\s*$/.exec(text)
  return m ? Number(m[1]) * 60 + Number(m[2]) : null
}

export function fmtTarget(seconds: number): string {
  const s = Math.round(seconds)
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`
}

/** "1–25" kept left-to-right inside Arabic text (an LTR isolate), for plain-string slots. */
export function rangeText(a: string, b: string): string {
  return `⁦${a}–${b}⁩`
}
