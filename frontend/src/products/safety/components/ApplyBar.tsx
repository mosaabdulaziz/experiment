/** What pressing Apply restarts, and the button (decisions D11/D29): a bar along the bottom of the console while
 *  something saved is not running yet (or is being applied), and the same, inline, in the zone editor -- whose overlay
 *  covers the bar. */
import { useEffect, useRef, useState } from 'react'
import { useConsole } from '../App'
import { api } from '../api'
import { usePoll } from '../data'
import { useLang } from '../i18n'
import { pendingChanges } from '../lib/apply'

const errorText = (e: unknown) => (e instanceof Error ? e.message : String(e))

export function ApplyBar({ inline }: { inline?: boolean }) {
  const { t, list } = useLang()
  const { cameras, reloadCameras } = useConsole()
  const [fast, setFast] = useState(false)
  const status = usePoll(api.applyStatus, fast ? 2000 : 10000)
  const applying = Boolean(status.data?.applying)
  const [error, setError] = useState<string | null>(null)
  const wasApplying = useRef(false)
  useEffect(() => {
    setFast(applying)
    if (wasApplying.current && !applying) reloadCameras()     // done: show what runs now
    wasApplying.current = applying
  }, [applying, reloadCameras])
  const pending = pendingChanges(cameras ?? [])
  if (!inline && !applying && !pending.features.length && !error) return null

  const apply = () => {
    setError(null)
    api.apply()
      .then(() => { setFast(true); status.reload() })
      .catch(e => setError(t('common.saveError', { error: errorText(e) })))
  }
  return (
    <div className={inline ? 'sf-apply-inline' : 'sf-applybar'} role="status">
      <span>
        {applying ? t('cameras.applying')
          : pending.features.length
            ? t('cameras.pending', { features: list(pending.features.map(f => t(`feature.${f}`))) })
            : t('cameras.applied')}
        {error && <span className="sf-error"> {error}</span>}
      </span>
      <button type="button" className={`iris-btn iris-btn-accent${inline ? ' iris-btn-sm' : ''}`}
        disabled={applying || !pending.features.length} onClick={apply}>
        {t('cameras.apply')}
      </button>
    </div>
  )
}
