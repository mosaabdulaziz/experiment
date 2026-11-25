import type { Candidate, Severity } from '../api'

export type Channels = {
  toast: boolean
  card: boolean
  badge: boolean
}

/** Resolve UI display severity, treating `high` tier as `critical` even for older persisted rows. */
export function effectiveSeverity(c: Pick<Candidate, 'severity'> & { tier?: string | null }): Severity {
  if (c.severity === 'critical' || c.tier === 'high') return 'critical'
  if (c.severity === 'warning' || c.tier === 'medium') return 'warning'
  return 'info'
}

/**
 * Map a suspected concealment candidate to its live UI channels:
 * - Already-reviewed rows (`status !== 'candidate'`) trigger no live alert surfaces.
 * - Unreviewed `critical` (`high` tier) and `warning` (`medium` tier) candidates
 *   surface in notices, the in-flow verification AlarmCard, and the rail badge.
 * - Unreviewed `info` candidates (`low` tier) surface in notices and the rail
 *   badge without opening an AlarmCard (per HLD §12 T3).
 */
export function channels(c: Pick<Candidate, 'severity' | 'status'>): Channels {
  if (c.status !== 'candidate') {
    return { toast: false, card: false, badge: false }
  }
  if (c.severity === 'critical' || c.severity === 'warning') {
    return { toast: true, card: true, badge: true }
  }
  if (c.severity === 'info') {
    return { toast: true, card: false, badge: true }
  }
  return { toast: false, card: false, badge: false }
}
