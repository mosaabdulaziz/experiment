import type { DraftZone } from './zoneDraft'
import { ZONE_FEATURES } from '@/types/zone'
import type { IntrusionMode, ProximityInfo, ZoneFeature, ZoneIntrusion, ZoneOvercrowding } from '@/types/zone'
import styles from './ZoneEditorOverlay.module.css'
import { ClassPicker } from './ClassPicker'
import { useEditorStrings, type EditorStrings } from './editorStrings'
import { ProximitySection } from './ProximitySection'

// Only classes that can meaningfully trigger each feature are offered — the detector also
// surfaces static objects (traffic light, stop sign) that can neither intrude nor crowd.
// Intersected with availableClasses so options the pipeline can't actually detect never show.
const INTRUSION_CLASSES = ['car', 'truck', 'person', 'motorcycle', 'bicycle', 'dog']
const OVERCROWDING_CLASSES = ['person', 'car', 'truck', 'bus', 'motorcycle']

interface Props {
  zone: DraftZone
  availableClasses: string[]
  /** The proximity module's defaults, when the pipeline runs it; null hides the section. */
  proximity: ProximityInfo | null
  disabled: boolean
  onChange: (patch: Partial<DraftZone>) => void
  /** The features this camera runs (the safety product). Given, only their sections show -- intrusion and
   *  overcrowding under `zones` -- plus an on/off section for each of PPE, falls and fire. Absent: iris's sections. */
  features?: string[]
  /** The equipment PPE can check (GET /zones `ppe.items`): the PPE section's choice of what this zone checks. */
  ppeItems?: string[]
}

// Each safety feature's section: its title and what being on means (the editor's strings: editorStrings.ts)
const FEATURE_SECTION: Record<ZoneFeature, (s: EditorStrings) => { label: string; hint: string }> = {
  ppe: s => ({ label: s.ppe, hint: s.ppeHint }),
  fall: s => ({ label: s.falls, hint: s.fallsHint }),
  fire: s => ({ label: s.fire, hint: s.fireHint }),
}

/** Per-zone detection config, bound to the selected zone. Edits flow up via `onChange` and
 *  are persisted together with the points when the operator hits Save Zones. */
export function ZoneConfigPanel({ zone, availableClasses, proximity, disabled, onChange, features, ppeItems }: Props) {
  const s = useEditorStrings()
  const showZones = !features || features.includes('zones')
  const showProximity = Boolean(proximity?.available) && (!features || features.includes('proximity'))
  const featureSections = ZONE_FEATURES.filter(f => features?.includes(f))
  const { intrusion, overcrowding } = zone
  const setIntrusion = (patch: Partial<ZoneIntrusion>) =>
    onChange({ intrusion: { ...intrusion, ...patch } })
  const setOvercrowding = (patch: Partial<ZoneOvercrowding>) =>
    onChange({ overcrowding: { ...overcrowding, ...patch } })
  const toggleIntrusionClass = (cls: string) =>
    setIntrusion({ classes: intrusion.classes.includes(cls)
      ? intrusion.classes.filter(c => c !== cls)
      : [...intrusion.classes, cls] })
  const toggleOvercrowdingClass = (cls: string) =>
    setOvercrowding({ classes: overcrowding.classes.includes(cls)
      ? overcrowding.classes.filter(c => c !== cls)
      : [...overcrowding.classes, cls] })
  const intrusionOptions = INTRUSION_CLASSES.filter(c => availableClasses.includes(c))
  const overcrowdingOptions = OVERCROWDING_CLASSES.filter(c => availableClasses.includes(c))
  // What this zone checks: its own list, else everything PPE checks (a zone saved before the choice existed)
  const ppeChecked = zone.ppe?.items ?? ppeItems ?? []
  const togglePpeItem = (item: string) => onChange({ ppe: { enabled: true,
    items: (ppeItems ?? []).filter(i => (i === item ? !ppeChecked.includes(i) : ppeChecked.includes(i))) } })

  return (
    <div className={styles.configPanel}>
      {showZones && (<>
      <section className={styles.configSection}>
        <div className={styles.configHead}>
          <span className={`${styles.configDot} ${styles.configDotIntrusion}`} />
          <span className={styles.configLabel}>{s.intrusion}</span>
        </div>
        <div className={styles.segment} role="tablist" aria-label={s.intrusionMode}>
          {(['off', 'restricted', 'valid'] as IntrusionMode[]).map(m => {
            const active = intrusion.mode === m
            const activeClass = m === 'restricted' ? styles.segChipAccent
              : m === 'valid' ? styles.segChipGreen : styles.segChipNeutral
            return (
              <button
                key={m}
                type="button"
                role="tab"
                aria-selected={active}
                className={`${styles.segChip} ${active ? activeClass : ''}`}
                onClick={() => setIntrusion({ mode: m })}
                disabled={disabled}
              >
                {m === 'off' ? s.off : m === 'restricted' ? s.restricted : s.valid}
              </button>
            )
          })}
        </div>
        {intrusion.mode !== 'off' && (
          <>
            <p className={styles.configHint}>
              {intrusion.mode === 'restricted' ? s.restrictedHint : s.validHint}
            </p>
            <ClassPicker
              label={s.alertOn}
              options={intrusionOptions}
              selected={intrusion.classes}
              onToggle={toggleIntrusionClass}
              disabled={disabled}
              format={s.className}
            />
          </>
        )}
      </section>

      <section className={styles.configSection}>
        <div className={styles.configHead}>
          <span className={`${styles.configDot} ${styles.configDotOvercrowding}`} />
          <span className={styles.configLabel}>{s.overcrowding}</span>
        </div>
        <div className={styles.segment} role="tablist" aria-label={s.overcrowding}>
          <button
            type="button" role="tab" aria-selected={!overcrowding.enabled}
            className={`${styles.segChip} ${!overcrowding.enabled ? styles.segChipNeutral : ''}`}
            onClick={() => setOvercrowding({ enabled: false })} disabled={disabled}
          >
            {s.off}
          </button>
          <button
            type="button" role="tab" aria-selected={overcrowding.enabled}
            className={`${styles.segChip} ${overcrowding.enabled ? styles.segChipYellow : ''}`}
            onClick={() => setOvercrowding({ enabled: true })} disabled={disabled}
          >
            {s.on}
          </button>
        </div>
        {overcrowding.enabled && (
          <>
            <div className={styles.thresholdField}>
              <span className={styles.configFieldLabel}>{s.threshold}</span>
              <input
                type="number"
                min={1}
                className={styles.numInput}
                value={overcrowding.threshold}
                onChange={e => setOvercrowding({
                  threshold: Math.max(1, Math.floor(Number(e.target.value) || 1)) })}
                disabled={disabled}
              />
              <span className={styles.configHint}>{s.thresholdHint}</span>
            </div>
            <ClassPicker
              label={s.count}
              options={overcrowdingOptions}
              selected={overcrowding.classes}
              onToggle={toggleOvercrowdingClass}
              disabled={disabled}
              format={s.className}
            />
          </>
        )}
      </section>

      </>)}

      {showProximity && proximity && (
        <ProximitySection value={zone.proximity} info={proximity} availableClasses={availableClasses}
          disabled={disabled} onChange={p => onChange({ proximity: p })} />
      )}

      {featureSections.map(f => {
        const on = Boolean(zone[f]?.enabled)
        const section = FEATURE_SECTION[f](s)
        return (
          <section key={f} className={styles.configSection}>
            <div className={styles.configHead}>
              <span className={`${styles.configDot} ${styles.configDotOvercrowding}`} />
              <span className={styles.configLabel}>{section.label}</span>
            </div>
            <div className={styles.segment} role="tablist" aria-label={section.label}>
              <button type="button" role="tab" aria-selected={!on}
                className={`${styles.segChip} ${!on ? styles.segChipNeutral : ''}`}
                onClick={() => onChange({ [f]: { ...zone[f], enabled: false } })} disabled={disabled}>
                {s.off}
              </button>
              <button type="button" role="tab" aria-selected={on}
                className={`${styles.segChip} ${on ? styles.segChipYellow : ''}`}
                onClick={() => onChange({ [f]: { ...zone[f], enabled: true } })} disabled={disabled}>
                {s.on}
              </button>
            </div>
            {on && f === 'ppe' && Boolean(ppeItems?.length) && (
              <ClassPicker label={s.checkFor} options={ppeItems ?? []} selected={ppeChecked}
                onToggle={togglePpeItem} disabled={disabled} format={s.ppeItem} />
            )}
            {on && <p className={styles.configHint}>{f === 'ppe' && !ppeItems?.length
              ? s.ppeHintNoChoice : section.hint}</p>}
          </section>
        )
      })}
    </div>
  )
}
