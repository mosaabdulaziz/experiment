/** Setup: the zone editor beside the Areas check, then the settings. The
 * zones are a per-camera draft kept here, so the Areas panel checks unsaved
 * edits too; only the configured cameras are shown or saved (a stale
 * camera still in zones.json is left alone on the server). */
import { useCallback, useEffect, useMemo, useState } from 'react'
import { api } from '../../api/client'
import type { Zone, ZonesByCam } from '../../api/types'
import type { PageProps } from '../../App'
import { useLang } from '../../i18n'
import { AreasPanel } from './AreasPanel'
import { SettingsForm } from './SettingsForm'
import { sameZones, ZoneEditor } from './ZoneEditor'

export function SetupPage({ cameras, camerasState, setDirty }: Pick<PageProps, 'cameras' | 'camerasState' | 'setDirty'>) {
  const { t } = useLang()
  const [saved, setSaved] = useState<ZonesByCam | null>(null)
  const [draft, setDraft] = useState<ZonesByCam>({})
  const [loadFailed, setLoadFailed] = useState(false)
  const camIds = useMemo(() => cameras.map(c => c.id), [cameras])

  useEffect(() => {
    api.zones().then(z => { setSaved(z); setDraft(z) }).catch(() => setLoadFailed(true))
  }, [])

  const dirty = !!saved && camIds.some(c => !sameZones(draft[c], saved[c]))
  useEffect(() => { setDirty(dirty) }, [dirty, setDirty])
  useEffect(() => () => setDirty(false), [setDirty])

  const onSaved = useCallback((cam: string, zones: Zone[]) => {
    setSaved(s => ({ ...s, [cam]: zones }))
    setDraft(d => ({ ...d, [cam]: zones }))
  }, [])

  return (
    <div className="setupPage">
      <header className="pageHead">
        <div className="pageHeadMain"><h1>{t('nav.setup')}</h1></div>
      </header>
      {(loadFailed || camerasState === 'failed') && <p className="errorBar" role="alert">{t('common.loadFailed')}</p>}
      {saved && camerasState !== 'loading'
        ? (
          <div className="setupBody">
            <ZoneEditor cameras={cameras} draft={draft} saved={saved} onDraft={setDraft} onSaved={onSaved} />
            <AreasPanel draft={draft} cameras={camIds} />
          </div>
        )
        : !loadFailed && <p className="card muted">{t('common.loading')}</p>}
      <SettingsForm />
    </div>
  )
}
