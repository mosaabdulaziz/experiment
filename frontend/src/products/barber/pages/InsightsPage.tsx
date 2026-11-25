/** Insights: the owner's history view -- filters, four KPIs and two charts over
 * one date range. The filters live in the hash
 * (#/insights?from=…&to=…&bucket=…&barber=…) so a view can be bookmarked. */
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { api } from '../api/client'
import type { Bucket, Stats } from '../api/types'
import type { PageProps } from '../App'
import { useLang } from '../i18n'
import { BarberBars, CountBars } from '../ui/charts'
import { addDays, fmtTimer, isoDate } from '../ui/format'
import { Num } from '../ui/Num'
import { insightsKpis } from './insightsKpis'
import { ReportPanel } from './ReportPanel'
import { productHref } from '../../../shared/router'

const BUCKETS: Bucket[] = ['hour', 'day', 'week', 'month']
const DAY = /^\d{4}-\d{2}-\d{2}$/

export function InsightsPage({ params, live, staff }: { params: URLSearchParams } & Pick<PageProps, 'live' | 'staff'>) {
  const { t, fmt } = useLang()
  const today = live.snap?.shop_day ?? isoDate(new Date())
  const pFrom = params.get('from')
  const pTo = params.get('to')
  const [from, setFrom] = useState(pFrom && DAY.test(pFrom) ? pFrom : today)
  const [to, setTo] = useState(pTo && DAY.test(pTo) ? pTo : pFrom && DAY.test(pFrom) ? pFrom : today)
  const pBucket = params.get('bucket') as Bucket | null
  const [bucket, setBucket] = useState<Bucket>(pBucket && BUCKETS.includes(pBucket) ? pBucket : from === to ? 'hour' : 'day')
  const [barber, setBarber] = useState(params.get('barber') ?? '')
  // Mounted before the first snapshot, `today` is the browser's date, which
  // after midnight is not the shop day: adopt the server's once it arrives,
  // unless the link gave dates or the user already changed a filter.
  const snapDay = live.snap?.shop_day
  const settled = useRef(!!snapDay || !!pFrom || !!pTo)
  useEffect(() => {
    if (settled.current || !snapDay) return
    settled.current = true
    setFrom(snapDay)
    setTo(snapDay)
  }, [snapDay])
  const [stats, setStats] = useState<Stats | null>(null)
  const [failed, setFailed] = useState(false)

  const names = useMemo(() => Object.fromEntries(staff.map(s => [s.id, s.name])), [staff])
  const rangeEnd = addDays(to, 1) // the picker's `to` is inclusive, the API's exclusive

  // Keep the filters in the hash without re-routing (replaceState fires no hashchange).
  useEffect(() => {
    const q = new URLSearchParams({ from, to, bucket })
    if (barber) q.set('barber', barber)
    history.replaceState(null, '', productHref(`/insights?${q}`))
  }, [from, to, bucket, barber])

  useEffect(() => {
    let off = false
    api.stats({ from, to: rangeEnd, bucket, barber })
      .then(s => { if (!off) { setStats(s); setFailed(false) } })
      .catch(() => { if (!off) { setStats(null); setFailed(true) } })
    return () => { off = true }
  }, [from, rangeEnd, bucket, barber])

  // The average wait and visits are shop-wide; with a barber chosen they come from an unfiltered call.
  const [shop, setShop] = useState<Stats | null>(null)
  useEffect(() => {
    if (!barber) { setShop(null); return }
    let off = false
    api.stats({ from, to: rangeEnd, barber: '' })
      .then(s => { if (!off) setShop(s) })
      .catch(() => { if (!off) setShop(null) })
    return () => { off = true }
  }, [from, rangeEnd, barber])

  const refilter = (fn: () => void) => { settled.current = true; fn() }
  const series = stats?.series ?? []

  return (
    <div className="insightsPage">
      <header className="pageHead">
        {/* a run on recordings reports the footage's day (2026-10-18), not the device's */}
        <div className="pageHeadMain"><h1>{t('nav.insights')}</h1><ReportPanel day={live.snap?.footage?.day ?? from} /></div>
        <div className="filterRow" role="group" aria-label={t('common.filters')}>
          <label className="filterField">
            <span>{t('f.from')}</span>
            <input className="input" type="date" value={from} max={to}
                   onChange={e => e.target.value && refilter(() => setFrom(e.target.value))} />
          </label>
          <label className="filterField">
            <span>{t('f.to')}</span>
            <input className="input" type="date" value={to} min={from}
                   onChange={e => e.target.value && refilter(() => setTo(e.target.value))} />
          </label>
          <span className="filterField">
            <span id="bucketLabel">{t('f.bucket')}</span>
            <span className="seg" role="group" aria-labelledby="bucketLabel">
              {BUCKETS.map(b => (
                <button key={b} type="button" aria-pressed={bucket === b} className={bucket === b ? 'segBtn segActive' : 'segBtn'}
                        onClick={() => refilter(() => setBucket(b))}>{t(`bucket.${b}`)}</button>
              ))}
            </span>
          </span>
          <label className="filterField">
            <span>{t('f.barber')}</span>
            <select className="select" value={barber} onChange={e => refilter(() => setBarber(e.target.value))}>
              <option value="">{t('f.all')}</option>
              {staff.map(b => <option key={b.id} value={b.id}>{b.name}</option>)}
            </select>
          </label>
        </div>
      </header>
      {failed && <p className="errorBar" role="alert">{t('common.loadFailed')}</p>}

      <section className="kpiGrid kpiGrid4" aria-label={t('nav.insights')}>
        {insightsKpis(stats, n => fmt.number(n), fmtTimer, barber ? { shop } : null)
          .map(k => <Kpi key={k.label} label={t(k.label)} value={k.value} sub={k.note && t(k.note)} />)}
      </section>

      <section className="chartRow">
        <div className="card chartCard">
          <h2 className="sectionLabel">{t('ch.haircutsTitle')}</h2>
          {series.length
            ? <CountBars series={series} bucket={bucket} value={p => p.served_customers} label={t('ch.haircutsTitle')} />
            : <div className="empty">{t('ch.noData')}</div>}
        </div>
        <div className="card chartCard">
          <h2 className="sectionLabel">{t('ch.perBarber')}</h2>
          <BarberBars rows={stats?.haircuts_per_barber ?? []} names={names} />
        </div>
      </section>
    </div>
  )
}

function Kpi({ label, value, sub }: { label: string; value: string; sub?: ReactNode }) {
  return (
    <div className="kpi">
      <span className="kpiLabel">{label}</span>
      <Num className="kpiValue">{value}</Num>
      {sub && <span className="kpiSub">{sub}</span>}
    </div>
  )
}
