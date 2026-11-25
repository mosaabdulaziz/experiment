/** House formatting helpers for Concealment Review. */

export type StageEntry = {
  stage: string
  entered_pts: number
  detail?: string
}

/** Format relative stream PTS (`+MM:SS` or `+HH:MM:SS`) or Unix epoch seconds (`YYYY-MM-DD HH:MM:SS`). */
export const seconds = (t: number | null | undefined): string => {
  if (t == null || !Number.isFinite(t)) return '—'
  if (t >= 0 && t < 1_000_000_000) {
    const total = Math.floor(t)
    const h = Math.floor(total / 3600)
    const m = Math.floor((total % 3600) / 60)
    const s = total % 60
    const pad = (n: number) => String(n).padStart(2, '0')
    return h > 0 ? `+${pad(h)}:${pad(m)}:${pad(s)}` : `+${pad(m)}:${pad(s)}`
  }
  return new Date(t * 1000).toISOString().replace('T', ' ').slice(0, 19)
}

/** Format an ISO-8601 timestamp or fallback PTS cleanly as `YYYY-MM-DD HH:MM:SS` (never raw microseconds or 1970 epoch). */
export const timestamp = (iso: string | null | undefined, fallbackPts?: number | null): string => {
  if (iso) {
    const ms = Date.parse(iso)
    if (Number.isFinite(ms) && ms >= 1_000_000_000_000) {
      return new Date(ms).toISOString().replace('T', ' ').slice(0, 19)
    }
  }
  return seconds(fallbackPts)
}

/** Compact `HH:MM:SS` clock portion from `timestamp()` for dense side lists. */
export const shortTimestamp = (iso: string | null | undefined, fallbackPts?: number | null): string => {
  const full = timestamp(iso, fallbackPts)
  return full.length === 19 && full[10] === ' ' ? full.slice(11) : full
}

/** Human-readable formatting for snake_case / comma-joined quality reasons. */
export const assessmentReasonLabel = (reason: string | null | undefined): string => {
  if (!reason) return 'insufficient evidence'
  const parts = reason
    .split(',')
    .map(p => p.trim().replace(/_/g, ' '))
    .filter(Boolean)
  return parts.length > 0 ? parts.join(', ') : 'insufficient evidence'
}

/** Duration in seconds to one decimal place. */
export const durationSeconds = (v: number | null | undefined): string =>
  v == null || !Number.isFinite(v) ? '—' : `${v.toFixed(1)}s`

/** Score to two decimals (house style). */
export const pct = (v: number | null | undefined): string =>
  v == null || !Number.isFinite(v) ? '—' : v.toFixed(2)

/** House chip class for a candidate severity (`critical` | `warning` | `info`). */
export const severityChip = (severity: string | null | undefined): 'chipRed' | 'chipAmber' | 'chipGreen' =>
  severity === 'critical' ? 'chipRed' : severity === 'warning' ? 'chipAmber' : 'chipGreen'

/** House chip class for a candidate tier (`high` | `medium` | `low`). */
export const tierChip = (tier: string | null | undefined): string =>
  tier === 'high' ? 'chip chipRed' : tier === 'medium' ? 'chip chipAmber' : 'chip'

/** House chip class for a non-accusatory review status. */
export const statusChip = (status: string): string =>
  status === 'candidate'
    ? 'chip chipAmber'
    : status === 'concern_noted'
      ? 'chip chipRed'
      : status === 'acknowledged'
        ? 'chip chipGreen'
        : 'chip'

/** Readable target label (`personal bag` or `clothing (waist)`). */
export const targetLabel = (targetKind: string | null | undefined, bodyZoneName?: string | null): string => {
  if (targetKind === 'personal_bag') return 'personal bag'
  if (targetKind === 'clothing') {
    const zone = bodyZoneName ? bodyZoneName.replace(/_/g, ' ') : 'body'
    return `clothing (${zone})`
  }
  return targetKind || '—'
}

/** Readable FSM stage progression (`DWELL -> REACH -> AT_BODY -> RETURNED`). */
export const stageChainLabel = (
  timeline: StageEntry[] | null | undefined,
  terminalStage?: string | null,
): string => {
  const stages = (timeline ?? []).map(s => s.stage).filter(Boolean)
  if (terminalStage && stages[stages.length - 1] !== terminalStage) {
    stages.push(terminalStage)
  }
  return stages.length > 0 ? stages.join(' -> ') : '—'
}
