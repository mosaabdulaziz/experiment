/** Customers: the register of returning customers. They have no names -- only
 * face / body crops (served to shop-LAN clients only, never uploaded), a visit
 * count and dates. The open customer lives in the hash (#/customers?id=…) so
 * an Insights "Customer" link opens its panel; closing puts focus back on the
 * card it came from. */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api/client'
import type { CustomerRow } from '../api/types'
import { customerLink, type PageProps } from '../App'
import { buildAreas } from '../floor/areas'
import { useLang } from '../i18n'
import { Crop } from '../ui/Crop'
import { rangeText } from '../ui/format'
import { NumIn } from '../ui/Num'
import { CustomerPanel } from './CustomerPanel'
import { servedChairs } from './customerEntries'
import { CUSTOMER_PAGE, retentionMonths, visitCount, visitCountKey } from './customers'
import { navigate } from '../../../shared/router'

type Sort = 'last_seen' | 'visit_count'
const SORTS: { id: Sort; key: 'cu.sortLast' | 'cu.sortVisits' }[] = [
  { id: 'last_seen', key: 'cu.sortLast' },
  { id: 'visit_count', key: 'cu.sortVisits' },
]
const REFRESH_MS = 60_000 // a customer is registered when their face is seen at the door; pick it up without a reload
const TOAST_MS = 3000

export function CustomersPage({ params, live, cameras }: { params: URLSearchParams } & Pick<PageProps, 'live' | 'cameras'>) {
  const { t, fmt, lang } = useLang()
  const selected = params.get('id')
  // chair ids by the shop's names (ph5-R1 is "R2"), as the Live page shows them
  const zones = live.snap?.zones
  const camIds = useMemo(() => cameras.map(c => c.id), [cameras])
  const areas = useMemo(() => buildAreas(zones ?? {}, camIds), [zones, camIds])
  const chairName = useCallback((id: string) => areas.byId[id]?.name ?? id, [areas])
  const [sort, setSort] = useState<Sort>('last_seen')
  const [page, setPage] = useState(0)
  const [data, setData] = useState<{ rows: CustomerRow[]; total: number } | null>(null)
  const [error, setError] = useState(false)
  const [months, setMonths] = useState<number | null>(null)
  const [reload, setReload] = useState(0)
  const [toast, setToast] = useState(false)
  const headRef = useRef<HTMLHeadingElement>(null)
  const lastOpen = useRef<string | null>(selected)


  useEffect(() => {
    api.policy().then(p => {
      const days = p.customer_retention_days
      if (typeof days === 'number' && days > 0) setMonths(retentionMonths(days))
    }).catch(() => {})
  }, [])

  useEffect(() => {
    let off = false
    const load = () => api.customers(CUSTOMER_PAGE, page * CUSTOMER_PAGE, sort)
      .then(r => { if (!off) { setData(r); setError(false) } })
      .catch(() => { if (!off) setError(true) })
    load()
    const timer = window.setInterval(load, REFRESH_MS)
    return () => { off = true; window.clearInterval(timer) }
  }, [page, sort, reload])

  // An erase can empty the last page: step back to one that has rows.
  useEffect(() => {
    if (data && data.rows.length === 0 && page > 0) setPage(Math.max(0, Math.ceil(data.total / CUSTOMER_PAGE) - 1))
  }, [data, page])

  // Focus back to the card the panel was opened from (or the title if it is gone).
  useEffect(() => {
    const prev = lastOpen.current
    lastOpen.current = selected
    if (prev && !selected) {
      const card = document.getElementById(`cust-${prev}`)
      ;(card ?? headRef.current)?.focus()
    }
  }, [selected])

  useEffect(() => {
    if (!toast) return
    const timer = window.setTimeout(() => setToast(false), TOAST_MS)
    return () => window.clearTimeout(timer)
  }, [toast])

  const close = useCallback(() => { navigate('/customers') }, [])
  // The erased customer's card is going away: skip the return-to-card focus
  // (it would land on the stale card, then <body>) and focus the title.
  const erased = useCallback(() => {
    lastOpen.current = null
    headRef.current?.focus()
    close()
    setReload(n => n + 1)
    setToast(true)
  }, [close])

  const total = data?.total ?? 0
  const first = total ? page * CUSTOMER_PAGE + 1 : 0
  const last = Math.min(total, (page + 1) * CUSTOMER_PAGE)

  return (
    <div className="customersPage">
      <header className="pageHead">
        <div className="pageHeadMain">
          <h1 ref={headRef} tabIndex={-1}>{t('cu.title')}</h1>
          {months !== null && <span className="muted small">{t('cu.privacy', { m: fmt.number(months) })}</span>}
        </div>
        <div className="pageHeadTools">
          <span className="seg" role="group" aria-label={t('cu.sort')}>
            {SORTS.map(s => (
              <button key={s.id} type="button" aria-pressed={sort === s.id}
                      className={sort === s.id ? 'segBtn segActive' : 'segBtn'}
                      onClick={() => { setSort(s.id); setPage(0) }}>
                {t(s.key)}
              </button>
            ))}
          </span>
        </div>
      </header>

      {error && <p className="errorBar" role="alert">{t('common.loadFailed')}</p>}
      {!data && !error && <p className="muted">{t('common.loading')}</p>}
      {data && data.total === 0 && <p className="empty custEmpty">{t('cu.empty')}</p>}

      {data && data.rows.length > 0 && (
        <section className="card">
          <ul className="custGrid">
            {data.rows.map(c => (
              <li key={c.id}>
                <a id={`cust-${c.id}`} href={customerLink(c.id)}
                   className={c.id === selected ? 'custCard custCardActive' : 'custCard'}
                   aria-label={t('cu.open', { visits: visitCount(lang, c.visit_count), date: fmt.date(c.last_seen) })}>
                  <Crop path={c.thumb} alt="" className={c.thumb_kind === 'body' ? 'custThumb custThumbBody' : 'custThumb'} />
                  <span className="custVisits"><NumIn text={t(visitCountKey(c.visit_count), { n: '{n}' })} n={fmt.number(c.visit_count)} /></span>
                  <span className="custDates">
                    <span><span className="muted">{t('cu.firstSeen')}</span> {fmt.date(c.first_seen)}</span>
                    <span><span className="muted">{t('cu.lastSeen')}</span> {fmt.date(c.last_seen)}</span>
                  </span>
                  {(() => {
                    // 2026-10-18 (user): at which chair he was served, each visit (newest first; the panel has them all)
                    const chairs = servedChairs(c.served, chairName)
                    if (!chairs.length) return null
                    const shown = chairs.length > 3 ? `${chairs.slice(0, 3).join(' · ')} …` : chairs.join(' · ')
                    return <span className="custChairs">{t('cu.servedAt', { chairs: shown })}</span>
                  })()}
                </a>
              </li>
            ))}
          </ul>
          <div className="tableFoot">
            <span>{t('common.range', { range: rangeText(fmt.number(first), fmt.number(last)), n: fmt.number(total) })}</span>
            <span className="pager">
              <button type="button" className="btnPlain btnSmall" disabled={page === 0} onClick={() => setPage(page - 1)}>
                {t('common.prev')}
              </button>
              <button type="button" className="btnPlain btnSmall" disabled={last >= total} onClick={() => setPage(page + 1)}>
                {t('common.next')}
              </button>
            </span>
          </div>
        </section>
      )}

      {selected && (
        <CustomerPanel key={selected} id={selected} onClose={close} onErased={erased} chairName={chairName} />
      )}
      <div className="toast" role="status" aria-live="polite">
        {toast && <span className="toastMsg">{t('cu.erased')}</span>}
      </div>
    </div>
  )
}
