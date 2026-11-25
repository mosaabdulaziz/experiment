/** Every feature's incidents in one list (filters by feature, camera, severity and state, grouped by day, as IRIS's
 *  incident cards in IRIS's grid), and the falls the fall service was unsure of, to confirm or dismiss (decision D8). */
import { useEffect, useMemo, useState } from 'react'
import ip from '@/pages/IncidentsPage.module.css'
import { useConsole } from '../App'
import { api } from '../api'
import { IncidentCard } from '../components/IncidentCard'
import { Pictogram } from '../components/Pictogram'
import { usePoll } from '../data'
import { useLang, type Key } from '../i18n'
import { en } from '../i18n/en'
import { byDay, filtersFromQuery, matches, type Filters, zonesOf } from '../lib/incidents'
import { FEATURES, type FallCandidate, type Feature, type SafetyEvent, type Severity } from '../lib/types'

type Tab = 'all' | 'review'
const STATUSES: Filters['status'][] = ['all', 'open', 'unacked']
const SEVERITIES: Severity[] = ['critical', 'warning', 'info']

export function IncidentsPage({ query }: { query: Record<string, string> }) {
  const { t, num } = useLang()
  const queryKey = JSON.stringify(query)
  const [tab, setTab] = useState<Tab>(query.tab === 'review' ? 'review' : 'all')
  const [filters, setFilters] = useState<Filters>(() => filtersFromQuery(query))
  // A link into this page while it is open (the banner's Review) resets it to what the link asks for.
  useEffect(() => {
    const q = JSON.parse(queryKey) as Record<string, string>
    setFilters(filtersFromQuery(q))
    setTab(q.tab === 'review' ? 'review' : 'all')
  }, [queryKey])
  const candidates = usePoll(api.fallCandidates, 30000)

  return (
    <div className="sf-page">
      <div className="sf-tabs" role="tablist">
        <button type="button" role="tab" className="sf-tab" aria-selected={tab === 'all'} onClick={() => setTab('all')}>
          {t('incidents.all')}
        </button>
        <button type="button" role="tab" className="sf-tab" aria-selected={tab === 'review'}
          onClick={() => setTab('review')}>
          {t('incidents.review')}{candidates.data?.length ? ` (${num(candidates.data.length)})` : ''}
        </button>
      </div>
      {tab === 'all'
        ? <IncidentList filters={filters} setFilters={setFilters} />
        : <FallReview candidates={candidates.data} error={candidates.error} reload={candidates.reload} />}
    </div>
  )
}

function IncidentList({ filters, setFilters }: { filters: Filters; setFilters: (f: Filters) => void }) {
  const { t, date, num } = useLang()
  const { events, waiting, settings, cameras, cameraName, acknowledge, play } = useConsole()
  // The feed holds the recent incidents; ones waiting for an acknowledgement may be older, so they are added in.
  const shown = useMemo(() => {
    const byId = new Map<number, SafetyEvent>(waiting.map(e => [e.id, e]))
    for (const e of events) byId.set(e.id, e)
    return [...byId.values()].filter(e => matches(e, filters, settings)).sort((a, b) => b.id - a.id)
  }, [events, waiting, filters, settings])
  const groups = byDay(shown, Date.now())
  // the zones the incidents in view fired in (the camera's, when one is picked), and the one picked even if it is gone
  const zones = useMemo(() => {
    const pool = [...events, ...waiting].filter(e => !filters.camera || e.camera_id === filters.camera)
    const found = zonesOf(pool)
    return filters.zone && !found.includes(filters.zone) ? [...found, filters.zone] : found
  }, [events, waiting, filters.camera, filters.zone])

  const toggleFeature = (f: Feature) => setFilters({
    ...filters, features: filters.features.includes(f) ? filters.features.filter(x => x !== f) : [...filters.features, f],
  })
  const dayLabel = (day: string) => {
    if (day === 'today' || day === 'yesterday') return t(`day.${day}`)
    const [y, m, d] = day.split('-').map(Number)
    return date(new Date(y, m - 1, d))
  }

  return (
    <>
      <div className="sf-filters">
        {FEATURES.map(f => (
          <button key={f} type="button" className="sf-chip" aria-pressed={filters.features.includes(f)}
            onClick={() => toggleFeature(f)}>
            <Pictogram feature={f} size={18} />{t(`feature.${f}`)}
          </button>
        ))}
      </div>
      <div className="sf-filters">
        <select className="iris-select" value={filters.camera ?? ''} aria-label={t('filter.cameras')}
          onChange={e => setFilters({ ...filters, camera: e.target.value || null })}>
          <option value="">{t('filter.cameras')}</option>
          {(cameras ?? []).map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
        </select>
        {zones.length > 0 && (
          <select className="iris-select" value={filters.zone ?? ''} aria-label={t('filter.zones')}
            onChange={e => setFilters({ ...filters, zone: e.target.value || null })}>
            <option value="">{t('filter.zones')}</option>
            {zones.map(z => <option key={z} value={z}>{z}</option>)}
          </select>
        )}
        <select className="iris-select" value={filters.severity ?? ''} aria-label={t('filter.severity')}
          onChange={e => setFilters({ ...filters, severity: (e.target.value || null) as Severity | null })}>
          <option value="">{t('filter.severity')}</option>
          {SEVERITIES.map(s => <option key={s} value={s}>{t(`severity.${s}`)}</option>)}
        </select>
        {STATUSES.map(s => (
          <button key={s} type="button" className="sf-chip sf-chip-text" aria-pressed={filters.status === s}
            onClick={() => setFilters({ ...filters, status: s })}>
            {t(`filter.status.${s}`)}
          </button>
        ))}
      </div>
      {!groups.length && <p className="sf-empty">{t(events.length ? 'incidents.empty' : 'incidents.none')}</p>}
      {groups.map(g => (
        <section key={g.day} className="sf-day-group">
          <h2 className="sf-day">{dayLabel(g.day)}<span className="sf-day-count">{num(g.events.length)}</span></h2>
          <div className={ip.grid}>
            {g.events.map(ev => (
              <IncidentCard key={ev.id} event={ev} settings={settings} cameraName={cameraName}
                onAcknowledge={acknowledge} onPlay={play} />
            ))}
          </div>
        </section>
      ))}
    </>
  )
}

function FallReview({ candidates, error, reload }: {
  candidates: FallCandidate[] | null; error: string | null; reload: () => void
}) {
  const { t, dateTime } = useLang()
  // what the fall service judged, in the console's words (its evaluations: products/fall_detection/scoring.py)
  const reviewTitle = (evaluation: string) => {
    const key = `review.${evaluation}`
    return key in en ? t(key as Key) : evaluation
  }
  const { cameraName } = useConsole()
  const [busy, setBusy] = useState<number | null>(null)
  const [saveError, setSaveError] = useState<string | null>(null)
  const review = (id: number, status: 'confirmed' | 'dismissed') => {
    setBusy(id)
    setSaveError(null)
    api.reviewFall(id, status)
      .then(reload)
      .catch(e => setSaveError(e instanceof Error ? e.message : String(e)))
      .finally(() => setBusy(null))
  }

  if (error && !candidates) {
    return (
      <p className="sf-error">
        {t('review.unavailable', { error })}{' '}
        <button type="button" className="iris-btn" onClick={reload}>{t('common.retry')}</button>
      </p>
    )
  }
  if (!candidates) return <p className="sf-empty">{t('common.loading')}</p>
  if (!candidates.length) return <p className="sf-empty">{t('review.empty')}</p>
  return (
    <>
      {saveError && <p className="sf-error">{t('common.saveError', { error: saveError })}</p>}
      {candidates.map(c => (
        <article key={c.id} className="sf-panel sf-incident" data-tone="critical">
          <Pictogram feature="fall" size={30} />
          <div>
            <div className="sf-incident-msg">{reviewTitle(c.evaluation)}</div>
            <div className="sf-incident-meta">
              <span>{cameraName(c.camera.split('@')[0])}</span>
              <span>{dateTime(c.created_at)}</span>
              <span>{t('review.score', { p: Math.round(c.combined * 100) })}</span>
            </div>
          </div>
          <div className="sf-incident-actions">
            <button type="button" className="iris-btn iris-btn-accent" disabled={busy === c.id}
              onClick={() => review(c.id, 'confirmed')}>
              {t('review.confirm')}
            </button>
            <button type="button" className="iris-btn" disabled={busy === c.id} onClick={() => review(c.id, 'dismissed')}>
              {t('review.dismiss')}
            </button>
          </div>
        </article>
      ))}
    </>
  )
}
