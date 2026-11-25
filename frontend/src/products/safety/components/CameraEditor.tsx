/** A camera's settings and zones in one place, IRIS's zone editor: opened from its tile on Live (its Configure
 *  button) or from a setup warning. The editor draws on one of the camera's role streams (zones are the camera's:
 *  every feature on it reads them) -- or, when no feature runs on it, on its still from go2rtc, so zones can be drawn
 *  before a feature starts -- and shows only the sections of the features it has on; above them, the camera's name and
 *  feature switches with Apply; below, where each feature looks and a warning when a change takes PPE / fall / fire
 *  off the whole frame. */
import { useEffect, useRef } from 'react'
import { stillUrl } from '../api'
import { EditorStringsContext } from '@/components/streams/editorStrings'
import { ZoneEditorOverlay } from '@/components/streams/ZoneEditorOverlay'
import type { DraftZone } from '@/components/streams/zoneDraft'
import { useStreams } from '@/hooks/useStreams'
import { useStreamsStore } from '@/stores/streamsStore'
import { useConsole } from '../App'
import { useLang } from '../i18n'
import { editorStrings } from '../i18n/editor'
import { coverage, narrowedFeatures } from '../lib/coverage'
import type { CameraFeature } from '../lib/types'
import { ApplyBar } from './ApplyBar'
import { CameraSettings, useCoverageText, useUnusedWarnings } from './CameraSettings'
import { FeatureTag } from './FeatureTag'

export function CameraEditor({ cameraId, streamId, onClose }: {
  cameraId: string
  /** null: no role stream runs for the camera -- the editor draws on its still and saves under the camera's id. */
  streamId: string | null
  onClose: () => void
}) {
  const { t, lang } = useLang()
  const { cameras, reloadCameras } = useConsole()
  const camera = cameras?.find(c => c.id === cameraId)
  useStreams()                                          // the editor reads the stream and the zones' apply state here
  const zoneEditId = useStreamsStore(s => s.zoneEditId)
  const setZoneEdit = useStreamsStore(s => s.setZoneEdit)
  // The role stream it draws on can stop while it is open (that feature switched off and applied, or its sidecar
  // restarting for another camera's change): it carries on over the camera's still rather than vanish.
  const drawable = useStreamsStore(s => s.streams.some(st => st.id === streamId && st.running))
  const editId = streamId ?? cameraId
  useEffect(() => {
    setZoneEdit(editId)
    return () => setZoneEdit(null)
  }, [editId, setZoneEdit])
  // Closed from inside (its close button, Escape, or after a save went live).
  const opened = useRef(false)
  useEffect(() => {
    if (zoneEditId) opened.current = true
    else if (opened.current) onClose()
  }, [zoneEditId, onClose])
  if (!zoneEditId || !camera) return null
  return (
    // the editor's own words in the console's language (IRIS keeps its English)
    <EditorStringsContext.Provider value={editorStrings(lang)}>
    <ZoneEditorOverlay key={zoneEditId} features={camera.features} title={camera.name}
      source={drawable ? undefined : { src: stillUrl(camera.id), live: false, label: t('live.still') }}
      settings={
        <section className="sf-editor-camera" aria-label={t('editor.camera')}>
          <CameraSettings camera={camera} onChanged={reloadCameras} />
          <ApplyBar inline />
        </section>
      }
      notice={(saved, draft) => <CoverageNotice features={camera.features} saved={saved} draft={draft} />} />
    </EditorStringsContext.Provider>
  )
}

function CoverageNotice({ features, saved, draft }: { features: CameraFeature[]; saved: DraftZone[]; draft: DraftZone[] }) {
  const { t, list } = useLang()
  const coverageText = useCoverageText()
  const drawn = draft.filter(z => z.closed)
  const narrowed = narrowedFeatures(saved, drawn).filter(f => features.includes(f))
  const warnings = useUnusedWarnings(features, drawn)
  if (!features.length) return null
  return (
    <div className="sf-coverage">
      <strong>{t('cameras.coverageTitle')}</strong>
      <ul>
        {features.map(f => (
          <li key={f}><FeatureTag feature={f} /> {coverageText(coverage(f, drawn))}</li>
        ))}
      </ul>
      {narrowed.length > 0 && (
        <p className="sf-warn">{t('zones.narrowed', { features: list(narrowed.map(f => t(`feature.${f}`))) })}</p>
      )}
      {warnings.map(w => <p key={w} className="sf-warn">{w}</p>)}
    </div>
  )
}
