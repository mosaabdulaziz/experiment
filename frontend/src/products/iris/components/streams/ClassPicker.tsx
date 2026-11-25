import styles from './ZoneEditorOverlay.module.css'

export const titleCase = (s: string) => s.replace(/\b\w/g, c => c.toUpperCase())

/** A row of toggle pills, one per object class (or, with `format`, any named choice). */
export function ClassPicker({ label, options, selected, onToggle, disabled, format = titleCase }: {
  label: string
  options: string[]
  selected: string[]
  onToggle: (cls: string) => void
  disabled?: boolean
  format?: (option: string) => string
}) {
  return (
    <div className={styles.classField}>
      <span className={styles.configFieldLabel}>{label}</span>
      <div className={styles.classPills}>
        {options.map(cls => {
          const on = selected.includes(cls)
          return (
            <button
              key={cls}
              type="button"
              className={`${styles.classPill} ${on ? styles.classPillActive : ''}`}
              aria-pressed={on}
              onClick={() => onToggle(cls)}
              disabled={disabled}
            >
              {format(cls)}
            </button>
          )
        })}
      </div>
    </div>
  )
}
