import styles from './TriggerTick.module.css'

interface Props {
  /** Where the alert fired inside the clip, 0-100. */
  positionPct: number
  /** Wall-clock time of the trigger, e.g. "14:32:07". */
  label: string
}

/**
 * The span of the clip with the fired moment marked on it.
 *
 * Clips are cut as [T-pad, T+pad], so a healthy one puts the marker dead centre.
 * When the recorder lagged past max_wait_seconds the backend falls back to whatever
 * gap-free run holds the alert, and the marker lands off-centre — which is how an
 * operator sees a truncated clip without playing it.
 */
export function TriggerTick({ positionPct, label }: Props) {
  const clamped = Math.min(Math.max(positionPct, 0), 100)
  return (
    <div className={styles.tick} aria-hidden="true">
      <div className={styles.rail} />
      <div className={styles.marker} style={{ left: `${clamped}%` }} />
      <span className={styles.label} style={{ left: `${clamped}%` }}>
        {label}
      </span>
    </div>
  )
}
