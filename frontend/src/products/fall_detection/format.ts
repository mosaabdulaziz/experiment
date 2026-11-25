/** The little formats the house shows everywhere. */

/** The sidecar's frame clock, as a readable UTC stamp. */
export const seconds = (t: number | null): string =>
  t == null ? '—' : new Date(t * 1000).toISOString().replace('T', ' ').slice(0, 19)

/** A score, to two decimals. */
export const pct = (v: number | null | undefined): string => (v == null ? '—' : v.toFixed(2))

/** The house chip for a severity. */
export const severityChip = (severity: string | null): 'chipRed' | 'chipAmber' | 'chipGreen' =>
  severity === 'critical' ? 'chipRed' : severity === 'warning' ? 'chipAmber' : 'chipGreen'

/** The evaluation, as the product names it. */
export const evaluationLabel = (evaluation: string): string => evaluation

/** The house chip for a review verdict: amber while it waits, green once confirmed, neutral otherwise.
 * `dismissed` and `acknowledged` are not confirmations -- drawing them green read as one. */
export const statusChip = (status: string): string =>
  status === 'candidate' ? 'chip chipAmber' : status === 'confirmed' ? 'chip chipGreen' : 'chip'
