/** A camera's settings: its name and which features run on it (saved at once, running after Apply). Inside the zone
 *  editor (CameraEditor), as IRIS keeps a camera's settings with its zones; where each feature looks, and what cannot
 *  raise anything yet, the editor says below the picture for the zones as drawn. */
import { useEffect, useState } from 'react'
import { useLang, type Key } from '../i18n'
import { api } from '../api'
import { unusedFeatures, type Coverage, type ZoneLike } from '../lib/coverage'
import { CAMERA_FEATURES, type CameraFeature, type SafetyCamera } from '../lib/types'
import { FeatureTag } from './FeatureTag'

const errorText = (e: unknown) => (e instanceof Error ? e.message : String(e))

export function useCoverageText() {
  const { t, list } = useLang()
  return (c: Coverage) => c.kind === 'whole' ? t('coverage.whole')
    : c.kind === 'none' ? t('coverage.none') : t('coverage.zones', { zones: list(c.zones) })
}

/** What a feature switched on that cannot raise anything yet says (lib/coverage.ts unusedFeatures): proximity and zone
 *  rules need a zone. */
export const UNUSED_TEXT: Partial<Record<CameraFeature, Key>> = {
  proximity: 'coverage.warnProximity', zones: 'coverage.warnZones',
}

export function useUnusedWarnings(features: CameraFeature[], zones: ZoneLike[] | null): string[] {
  const { t } = useLang()
  if (!zones) return []
  return unusedFeatures(features, zones).flatMap(f => (UNUSED_TEXT[f] ? [t(UNUSED_TEXT[f])] : []))
}

interface Props {
  camera: SafetyCamera
  onChanged: () => void
}

export function CameraSettings({ camera, onChanged }: Props) {
  const { t } = useLang()
  const [name, setName] = useState(camera.name)
  useEffect(() => setName(camera.name), [camera.name])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const save = (change: { name?: string; features?: string[] }) => {
    setBusy(true)
    setError(null)
    api.saveCamera(camera.id, change)
      .then(onChanged)
      .catch(e => setError(t('common.saveError', { error: errorText(e) })))
      .finally(() => setBusy(false))
  }
  const toggle = (f: CameraFeature, on: boolean) =>
    save({ features: CAMERA_FEATURES.filter(x => (x === f ? on : camera.features.includes(x))) })
  const streamOf = (f: CameraFeature) => camera.streams.find(s => s.feature === f)

  return (
    <div className="sf-camera-settings">
      <div className="sf-camera-head">
        <input className="sf-input" value={name} aria-label={t('cameras.name')} onChange={e => setName(e.target.value)} />
        <button type="button" className="iris-btn" disabled={busy || !name.trim() || name.trim() === camera.name}
          onClick={() => save({ name: name.trim() })}>
          {t('cameras.saveName')}
        </button>
        <span className="sf-muted sf-camera-id">{camera.id}</span>
      </div>
      <div className="sf-camera-features" role="group" aria-label={t('editor.features')}>
        {CAMERA_FEATURES.map(f => {
          const on = camera.features.includes(f)
          const stream = streamOf(f)
          return (
            <label key={f} className="sf-feature-toggle" data-on={on || undefined}>
              <span className="sf-feature-text">
                <FeatureTag feature={f} />
                {on && !stream && <span className="sf-feature-sub sf-warn">{t('cameras.startsOnApply')}</span>}
                {!on && stream && <span className="sf-feature-sub sf-warn">{t('cameras.stopsOnApply')}</span>}
              </span>
              <input type="checkbox" className="sf-switch" checked={on} disabled={busy}
                aria-label={t(`feature.${f}`)} onChange={e => toggle(f, e.target.checked)} />
            </label>
          )
        })}
      </div>
      {error && <p className="sf-error">{error}</p>}
    </div>
  )
}
