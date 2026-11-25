import { IconPlus, IconTrash } from '@/components/ui/icons'
import { useEditorStrings } from './editorStrings'
import { zoneColor, zoneSummary, type DraftZone } from './zoneDraft'
import styles from './ZoneRail.module.css'

interface Props {
  zones: DraftZone[]
  selectedKey: string | null
  disabled: boolean
  onSelect: (key: string) => void
  onAdd: () => void
  onDelete: (key: string) => void
  onRename: (key: string, name: string) => void
}

/** The layers panel: one row per zone (color swatch, editable name, summary, delete) plus
 *  "Add zone". Selecting a row makes that zone active on the canvas and in the config panel —
 *  the swatch, the polygon, and the config accent all share the zone's purpose color. */
export function ZoneRail({ zones, selectedKey, disabled, onSelect, onAdd, onDelete, onRename }: Props) {
  const s = useEditorStrings()
  return (
    <aside className={styles.rail} aria-label={s.zones}>
      <div className={styles.head}>{s.zones}</div>
      <div className={styles.list}>
        {zones.length === 0 && (
          <p className={styles.empty}>{s.noZones}</p>
        )}
        {zones.map(zone => {
          const selected = zone.key === selectedKey
          return (
            <div
              key={zone.key}
              className={`${styles.row} ${selected ? styles.rowSelected : ''}`}
              onClick={() => onSelect(zone.key)}
              role="button"
              tabIndex={0}
              onKeyDown={e => {
                if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect(zone.key) }
              }}
              aria-current={selected}
            >
              <span className={styles.swatch} style={{ background: zoneColor(zone.intrusion.mode) }} />
              <div className={styles.rowBody}>
                <input
                  className={styles.nameInput}
                  value={zone.name}
                  disabled={disabled}
                  onChange={e => onRename(zone.key, e.target.value)}
                  onClick={e => e.stopPropagation()}
                  aria-label={s.zoneName}
                  maxLength={40}
                />
                <span className={styles.summary}>{zoneSummary(zone, s)}</span>
              </div>
              <button
                type="button"
                className={styles.delete}
                disabled={disabled}
                onClick={e => { e.stopPropagation(); onDelete(zone.key) }}
                aria-label={s.deleteZone(zone.name)}
                title={s.deleteZone(zone.name)}
              >
                <IconTrash />
              </button>
            </div>
          )
        })}
      </div>
      <button
        type="button"
        className={styles.add}
        onClick={onAdd}
        disabled={disabled}
      >
        <IconPlus width={13} height={13} />
        {s.addZone}
      </button>
    </aside>
  )
}
