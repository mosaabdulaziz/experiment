import { useState } from 'react'
import type { HiddenFeet, ProximityInfo, ZoneProximity } from '@/types/zone'
import { ClassPicker } from './ClassPicker'
import { useEditorStrings } from './editorStrings'
import styles from './ZoneEditorOverlay.module.css'

const VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle']
// Only for the "≈ m" hint: distances are measured in person heights, with no metric calibration.
const PERSON_HEIGHT_M = 1.7

interface Props {
  value: ZoneProximity | undefined
  info: ProximityInfo
  availableClasses: string[]
  disabled: boolean
  onChange: (next: ZoneProximity) => void
}

type NumberKey = 'persist_s' | 'grace_s' | 'cooldown_s' | 'stationary_window_s' | 'stationary_motion'

/** The zone's proximity rule: the basics up front, the rest under Advanced. Values start at the running pipeline's
 *  defaults; the backend stores only what the operator changes. */
export function ProximitySection({ value, info, availableClasses, disabled, onChange }: Props) {
  const s = useEditorStrings()
  const [advanced, setAdvanced] = useState(false)
  const current: ZoneProximity = value ?? { ...info.defaults, enabled: false }
  const set = (patch: Partial<ZoneProximity>) => onChange({ ...current, ...patch })
  const options = VEHICLE_CLASSES.filter(c => availableClasses.includes(c))
  // Only the classes this pipeline offers are kept: a default it cannot detect would be refused on save, from a pill
  // the operator cannot see.
  const toggleClass = (cls: string) => {
    const next = current.vehicle_classes.includes(cls)
      ? current.vehicle_classes.filter(c => c !== cls)
      : [...current.vehicle_classes, cls]
    set({ vehicle_classes: next.filter(c => options.includes(c)) })
  }
  const number = (key: NumberKey, label: string, unit: string, min: number, step: number) => (
    <label className={styles.numRow}>
      <span className={styles.configFieldLabel}>{label}</span>
      <input
        type="number" className={styles.numInput} min={min} step={step} value={current[key]} disabled={disabled}
        onChange={e => {
          const v = Number(e.target.value)
          if (Number.isFinite(v) && v >= min) set({ [key]: v } as Partial<ZoneProximity>)
        }}
      />
      <span className={styles.configHint}>{unit}</span>
    </label>
  )

  return (
    <section className={styles.configSection}>
      <div className={styles.configHead}>
        <span className={`${styles.configDot} ${styles.configDotProximity}`} />
        <span className={styles.configLabel}>{s.proximity}</span>
      </div>
      <div className={styles.segment} role="tablist" aria-label={s.proximity}>
        <button type="button" role="tab" aria-selected={!current.enabled} disabled={disabled}
          className={`${styles.segChip} ${!current.enabled ? styles.segChipNeutral : ''}`}
          onClick={() => set({ enabled: false })}>{s.off}</button>
        <button type="button" role="tab" aria-selected={current.enabled} disabled={disabled}
          className={`${styles.segChip} ${current.enabled ? styles.segChipAccent : ''}`}
          onClick={() => set({ enabled: true })}>{s.on}</button>
      </div>
      {current.enabled && (
        <>
          <p className={styles.configHint}>{s.proximityHint}</p>
          <ClassPicker label={s.vehicles} options={options}
            selected={current.vehicle_classes} onToggle={toggleClass} disabled={disabled} format={s.className} />
          <div className={styles.thresholdField}>
            <span className={styles.configFieldLabel}>{s.maxGap}</span>
            <input type="range" className={styles.gapSlider} min={0.25} max={2} step={0.05} value={current.max_gap}
              disabled={disabled} onChange={e => set({ max_gap: Number(e.target.value) })} />
            <span className={styles.configHint}>
              {s.maxGapHint(current.max_gap, current.max_gap * PERSON_HEIGHT_M, current.max_gap / 2)}
            </span>
          </div>
          <label className={styles.checkRow}>
            <input type="checkbox" checked={current.depth_check} disabled={disabled || !info.depth_model}
              onChange={e => set({ depth_check: e.target.checked })} />
            <span>{s.depthCheck}{!info.depth_model && s.noDepthModel}</span>
          </label>
          {current.depth_check && info.depth_model && (
            <div className={styles.thresholdField}>
              <span className={styles.configFieldLabel}>{s.depthCutoff}</span>
              <input type="range" className={styles.gapSlider} min={0.05} max={0.6} step={0.05}
                value={current.depth_max_gap} disabled={disabled}
                onChange={e => set({ depth_max_gap: Number(e.target.value) })} />
              <span className={styles.configHint}>
                {s.depthCutoffHint(current.depth_max_gap)}
              </span>
            </div>
          )}
          <button type="button" className={styles.advancedToggle} aria-expanded={advanced}
            onClick={() => setAdvanced(a => !a)}>{advanced ? '▾' : '▸'} {s.advanced}</button>
          {advanced && (
            <div className={styles.advancedBody}>
              {number('persist_s', s.persistence, s.persistenceUnit, 0.05, 0.1)}
              {number('grace_s', s.grace, s.graceUnit, 0, 0.1)}
              {number('cooldown_s', s.cooldown, s.cooldownUnit, 0, 1)}
              {number('stationary_window_s', s.stationaryWindow, s.seconds, 0.05, 0.5)}
              {number('stationary_motion', s.stationaryMovement, s.personHeights, 0.05, 0.05)}
              <div className={styles.segment} role="tablist" aria-label={s.hiddenFeet}>
                {(['unconfirmed', 'suppress'] as HiddenFeet[]).map(m => (
                  <button key={m} type="button" role="tab" aria-selected={current.hidden_feet === m}
                    disabled={disabled}
                    className={`${styles.segChip} ${current.hidden_feet === m ? styles.segChipNeutral : ''}`}
                    onClick={() => set({ hidden_feet: m })}>
                    {m === 'unconfirmed' ? s.hiddenFeetFlag : s.hiddenFeetIgnore}
                  </button>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </section>
  )
}
