/** The alert log as snapshot cards: every long wait / long haircut in a date range, each with the
 * close-up of its subject (click for the full frame). Open alerts keep counting. */
import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api/client'
import type { Alert, ZonesByCam } from '../api/types'
import type { PageProps } from '../App'
import { alertCard, alertSections, cropView, openAlertsKey, pageSections, type AlertCardModel } from './alertCards'
import { buildAreas } from '../floor/areas'
import { useLang } from '../i18n'
import { ticketLabel } from '../live/selectors'
import { addDays, fmtClock, fmtTimer, isoDate, rangeText } from '../ui/format'
import { Num } from '../ui/Num'

const PAGE = 24

/** The card's picture: the subject close-up drawn from the full frame with uniform scaling. Shows the
 * placeholder until the image has loaded and for good if it fails (retention, off the shop LAN). */
function AlertShot({ card, failed, onFail, onOpen }: {
  card: AlertCardModel; failed: boolean; onFail: (id: string) => void; onOpen: (c: AlertCardModel, btn: HTMLButtonElement) => void
}) {
  const { t } = useLang()
  const [size, setSize] = useState<[number, number] | null>(null)
  const url = failed ? null : card.imageUrl
  const view = size && card.box ? cropView(card.box, size[0], size[1]) : null
  const style = view
    ? { width: `${view.widthPct}%`, left: `${view.leftPct}%`, top: `${view.topPct}%`, height: 'auto' }
    : { width: '100%', left: 0, top: 0 }
  return (
    <button type="button" className="alertShot" disabled={!url} aria-label={t('al.showFrame')}
            onClick={e => onOpen(card, e.currentTarget)}>
      {!(url && size) && <span className="muted small alertShotPh">{url ? '' : t('al.noSnapshot')}</span>}
      {url && (
        <img src={url} alt="" className="alertShotImg" style={{ ...style, opacity: size ? 1 : 0 }}
             onLoad={e => setSize([e.currentTarget.naturalWidth, e.currentTarget.naturalHeight])}
             onError={() => onFail(card.id)} />
      )}
    </button>
  )
}

type KindFilter = '' | 'excessive_wait' | 'excessive_service'
const KINDS: { id: KindFilter; key: 'al.kindAll' | 'alert.excessive_wait' | 'alert.excessive_service' }[] = [
  { id: '', key: 'al.kindAll' }, { id: 'excessive_wait', key: 'alert.excessive_wait' }, { id: 'excessive_service', key: 'alert.excessive_service' },
]

export function AlertsPage({ live, staff, cameras }: Pick<PageProps, 'live' | 'staff' | 'cameras'>) {
  const { t, fmt } = useLang()
  const today = live.snap?.shop_day ?? isoDate(new Date())
  const [from, setFrom] = useState(today)
  const [to, setTo] = useState(today)
  // Mounted before the first snapshot: the browser's date may not be the shop
  // day (after midnight), so adopt the server's once it arrives -- unless the
  // user already picked dates.
  const snapDay = live.snap?.shop_day
  const settled = useRef(!!snapDay)
  useEffect(() => {
    if (settled.current || !snapDay) return
    settled.current = true
    setFrom(snapDay)
    setTo(snapDay)
  }, [snapDay])
  const pick = (set: (v: string) => void) => (v: string) => { settled.current = true; if (v) set(v) }
  const [openOnly, setOpenOnly] = useState(false)
  const [kind, setKind] = useState<KindFilter>('')
  const [full, setFull] = useState<AlertCardModel | null>(null)
  const [failed, setFailed] = useState<Set<string>>(() => new Set())
  const opener = useRef<HTMLButtonElement | null>(null)
  const closeBtn = useRef<HTMLButtonElement | null>(null)
  const close = () => { setFull(null); opener.current?.focus() }
  const [page, setPage] = useState(0)
  const [rows, setRows] = useState<Alert[] | null>(null)
  const [error, setError] = useState(false)
  const [zones, setZones] = useState<ZonesByCam | null>(null)
  const names = useMemo(() => Object.fromEntries(staff.map(s => [s.id, s.name])), [staff])
  const camIds = useMemo(() => cameras.map(c => c.id), [cameras])
  const areas = useMemo(() => buildAreas(zones ?? live.snap?.zones ?? {}, camIds), [zones, live.snap?.zones, camIds])
  // refetch when the set of open alerts changes (one opened or cleared) or an open one's picture is saved
  const openKey = openAlertsKey(live.snap?.open_alerts)

  useEffect(() => { api.zones().then(setZones).catch(() => {}) }, [])
  useEffect(() => { setPage(0) }, [from, to, openOnly, kind])
  useEffect(() => {
    if (!full) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { setFull(null); opener.current?.focus() } }
    closeBtn.current?.focus()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [full])
  useEffect(() => {
    let off = false
    api.alerts({ from, to: addDays(to, 1), open: openOnly || undefined, limit: 500 })
      .then(r => { if (!off) { setRows(r); setError(false) } })
      .catch(() => { if (!off) setError(true) })
    return () => { off = true }
  }, [from, to, openOnly, openKey])

  const cards = useMemo(
    () => (rows ?? []).filter(a => !kind || a.kind === kind)
      .map(a => alertCard(a, live.now, names, id => areas.byId[id]?.name ?? null)),
    [rows, kind, live.now, names, areas])
  // open first, then closed by shop day (2026-10-18, user)
  const sections = useMemo(() => alertSections(cards, live.snap?.day_start_hour ?? 0), [cards, live.snap?.day_start_hour])
  const total = cards.length
  const first = total ? page * PAGE + 1 : 0
  const last = Math.min(total, (page + 1) * PAGE)

  return (
    <div className="alertsPage">
      <header className="pageHead">
        <div className="pageHeadMain"><h1>{t('al.title')}</h1></div>
        <div className="filterRow" role="group" aria-label={t('common.filters')}>
          <label className="filterField">
            <span>{t('f.from')}</span>
            <input className="input" type="date" value={from} max={to} onChange={e => pick(setFrom)(e.target.value)} />
          </label>
          <label className="filterField">
            <span>{t('f.to')}</span>
            <input className="input" type="date" value={to} min={from} onChange={e => pick(setTo)(e.target.value)} />
          </label>
          <span className="seg" role="group" aria-label={t('al.kindFilter')}>
            {KINDS.map(k => (
              <button key={k.id} type="button" aria-pressed={kind === k.id}
                      className={kind === k.id ? 'segBtn segActive' : 'segBtn'} onClick={() => setKind(k.id)}>
                {t(k.key)}
              </button>
            ))}
          </span>
          <button type="button" className="toggleChip" aria-pressed={openOnly} onClick={() => setOpenOnly(o => !o)}>
            {t('al.openOnly')}
          </button>
        </div>
      </header>
      {error && <p className="errorBar" role="alert">{t('common.loadFailed')}</p>}
      <section aria-label={t('al.title')}>
        {rows && total === 0 && <p className="muted tableEmpty">{t('al.none')}</p>}
        {pageSections(sections, page, PAGE).map(sec => (
          <div key={sec.kind === 'open' ? 'open' : sec.day} className="alertSection">
            <h2 className="alertSectionHead">{sec.kind === 'open' ? t('al.sectionOpen') : fmt.date(`${sec.day}T12:00:00`)}</h2>
            <ul className="alertGrid">
              {sec.cards.map(c => (
                <li key={c.id} className={`alertCard ${c.kindKey === 'alert.excessive_service' ? 'alertCut' : 'alertWait'}${c.open ? ' alertOpen' : ''}`}>
                  <AlertShot card={c} failed={failed.has(c.id)}
                             onFail={id => setFailed(f => new Set(f).add(id))}
                             onOpen={(card, btn) => { opener.current = btn; setFull(card) }} />
                  <div className="alertBody">
                    <Num className="alertTicket">{ticketLabel(c.ticket)}</Num>
                    <div className="alertTop">
                      <span className="alertKind">{t(c.kindKey)}</span>
                      <span className={c.open ? 'chip chipRed' : 'chip'}>{t(c.open ? 'al.open' : 'al.closed')}</span>
                    </div>
                    {c.chair && <div><Num>{c.chair}</Num>{c.barber && <span className="muted"> · {c.barber}</span>}</div>}
                    <div className="muted small">{fmt.date(c.startedAt)} <Num>{fmtClock(c.startedAt)}</Num></div>
                    <div><Num className={c.open ? 'timer lvl-over' : 'timer'}>{fmtTimer(c.durationS)}</Num>
                      {c.open && <span className="muted small"> {t('al.stillOpen')}</span>}</div>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        ))}
        <div className="tableFoot">
          <span>{t('common.range', { range: rangeText(fmt.number(first), fmt.number(last)), n: fmt.number(total) })}</span>
          <span className="pager">
            <button type="button" className="btnPlain btnSmall" disabled={page === 0} onClick={() => setPage(p => p - 1)}>{t('common.prev')}</button>
            <button type="button" className="btnPlain btnSmall" disabled={last >= total} onClick={() => setPage(p => p + 1)}>{t('common.next')}</button>
          </span>
        </div>
      </section>
      {full?.imageUrl && !failed.has(full.id) && (
        <div className="alertFull" role="dialog" aria-modal="true" aria-label={t(full.kindKey)} onClick={close}>
          <button ref={closeBtn} type="button" className="alertFullClose" aria-label={t('al.close')} onClick={close}>×</button>
          <div className="alertFullFrame" onClick={e => e.stopPropagation()}>
            <img src={full.imageUrl} alt="" onError={() => { setFailed(f => new Set(f).add(full.id)); close() }} />
            {full.box && <span className="alertFullBox" style={{ left: `${full.box[0] * 100}%`, top: `${full.box[1] * 100}%`,
                                                                 width: `${full.box[2] * 100}%`, height: `${full.box[3] * 100}%` }} />}
          </div>
        </div>
      )}
    </div>
  )
}
