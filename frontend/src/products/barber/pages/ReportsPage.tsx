/** Reports page (2026-10-25; spec 2026-10-25-barber-saved-reports-design.md section 3.2): the reports the service saved
 * when each shop day (or recording lap), week and month ended, newest first. The list reaches anyone; the files open
 * on the shop network only. Building a report for any other period stays on the Insights page's Report panel. */
import { useEffect, useState } from 'react'
import { api, type ReportKind, type SavedReports } from '../api/client'
import type { PageProps } from '../App'
import { useLang, type Key } from '../i18n'
import { periodLabel, reportLines, savedTime, type LinkKind } from './savedReports'

const KINDS: ReportKind[] = ['day', 'week', 'month']
const LINK_LABEL: Record<LinkKind, Key> = { view: 'rep.view', pdf: 'rep.pdf', html: 'rep.htmlFile', docx: 'rep.word' }
const REFRESH_MS = 60_000                       // an end-of-day save shows up without a reload

export function ReportsPage(_props: PageProps) {
  const { t, lang } = useLang()
  const [kind, setKind] = useState<ReportKind>('day')
  const [data, setData] = useState<SavedReports | null>(null)
  const [failed, setFailed] = useState(false)
  useEffect(() => {
    let live = true
    const load = () => api.savedReports(kind)
      .then(d => { if (live) { setData(d); setFailed(false) } })
      .catch(() => { if (live) setFailed(true) })
    setData(null)
    load()
    const timer = window.setInterval(load, REFRESH_MS)
    return () => { live = false; window.clearInterval(timer) }
  }, [kind])

  return (
    <div className="page reportsPage">
      <header className="pageHead">
        <div className="pageHeadMain"><h1>{t('nav.reports')}</h1></div>
        <span className="seg" role="group" aria-label={t('reps.kind')}>
          {KINDS.map(k => (
            <button key={k} type="button" aria-pressed={kind === k} className={kind === k ? 'segBtn segActive' : 'segBtn'}
                    onClick={() => setKind(k)}>{t(`reps.${k}`)}</button>
          ))}
        </span>
      </header>
      {data && !data.allowed && <p className="note">{t('reps.lanOnly')}</p>}
      {failed && !data && <p className="empty">{t('reps.failed')}</p>}
      {data && data.items.length === 0 && <p className="empty">{t(data.enabled ? 'reps.empty' : 'reps.off')}</p>}
      {data && data.items.length > 0 && (
        <ul className="savedList">
          {data.items.map(r => (
            <li key={r.id} className="card savedRow">
              <div className="savedHead">
                <strong>{periodLabel(lang, r)}</strong>
                {r.lap_ended && <span>{t('reps.lapEnded', { time: savedTime(lang, r.lap_ended) })}</span>}
                {r.label && <span className="savedLabel">{r.label}</span>}
                <span className="savedAt">{t('reps.saved', { time: savedTime(lang, r.saved_at) })}</span>
              </div>
              {reportLines(r).map(line => (
                <div key={line.report} className="savedLine">
                  <span className="savedWhat">{t(line.report === 'photos' ? 'rep.photos' : 'reps.analysis')}</span>
                  {line.langs.map(l => (
                    <span key={l.lang} className="savedLang">
                      <span className="savedLangName">{t(`reps.lang.${l.lang}`)}</span>
                      {l.links.map(k => data.allowed
                        ? <a key={k.kind} className="btnGhost" href={k.href}
                             {...(k.kind === 'view' ? { target: '_blank', rel: 'noreferrer' } : { download: true })}>
                            {t(LINK_LABEL[k.kind])}
                          </a>
                        : <button key={k.kind} type="button" className="btnGhost" disabled title={t('reps.lanOnly')}>
                            {t(LINK_LABEL[k.kind])}
                          </button>)}
                    </span>
                  ))}
                </div>
              ))}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
