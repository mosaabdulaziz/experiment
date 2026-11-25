import type { AlertKind } from '@/lib/clips'
import { ALERT_KIND_LABEL } from '@/lib/clips'
import styles from './IncidentFilters.module.css'

const KINDS: AlertKind[] = ['intrusion', 'exit', 'overcrowding', 'proximity', 'ai']

interface Props {
  cameras: string[]
  activeCameras: string[]
  activeKinds: AlertKind[]
  footageOnly: boolean
  heldBack: boolean
  onToggleCamera: (camera: string) => void
  onToggleKind: (kind: AlertKind) => void
  onToggleFootageOnly: () => void
  onToggleHeldBack: () => void
}

export function IncidentFilters({
  cameras,
  activeCameras,
  activeKinds,
  footageOnly,
  heldBack,
  onToggleCamera,
  onToggleKind,
  onToggleFootageOnly,
  onToggleHeldBack,
}: Props) {
  return (
    <div className={styles.bar}>
      {cameras.length > 1 && (
        <div className={styles.group}>
          <span className={styles.groupLabel}>Camera</span>
          {cameras.map(c => {
            const on = activeCameras.includes(c)
            return (
              <button
                key={c}
                type="button"
                aria-pressed={on}
                className={`${styles.chip} ${on ? styles.chipOn : ''}`}
                onClick={() => onToggleCamera(c)}
              >
                {c}
              </button>
            )
          })}
        </div>
      )}

      <div className={styles.group}>
        <span className={styles.groupLabel}>Type</span>
        {KINDS.map(k => {
          const on = activeKinds.includes(k)
          return (
            <button
              key={k}
              type="button"
              aria-pressed={on}
              className={`${styles.chip} ${on ? styles.chipOn : ''}`}
              onClick={() => onToggleKind(k)}
            >
              {ALERT_KIND_LABEL[k]}
            </button>
          )
        })}
      </div>

      <button
        type="button"
        aria-pressed={footageOnly}
        className={`${styles.chip} ${styles.footageChip} ${footageOnly ? styles.chipOn : ''}`}
        onClick={onToggleFootageOnly}
        title="Show only incidents whose clip is ready to play"
      >
        With footage only
      </button>

      <button
        type="button"
        aria-pressed={heldBack}
        className={`${styles.chip} ${styles.footageChip} ${heldBack ? styles.chipOn : ''}`}
        onClick={onToggleHeldBack}
        title="Show only the close pairs a zone's depth check held back (not alerts): to check its depth cut-off"
      >
        Held back by depth
      </button>
    </div>
  )
}
