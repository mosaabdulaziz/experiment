/** One customer, sliding in from the inline end: up to five faces, three
 * full-body crops, the visits (date, time and the chair he was served at), the door entries where the shop has a
 * door camera, and Erase.
 * Modal: focus moves in on open and stays inside, Escape closes (or first
 * cancels the erase confirmation). */
import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react'
import { api, ApiError } from '../api/client'
import type { CustomerDetail } from '../api/types'
import { useLang } from '../i18n'
import { Confirm } from '../ui/ui'
import { Crop } from '../ui/Crop'
import { NumIn } from '../ui/Num'
import { entryRows, visitRows } from './customerEntries'
import { MAX_BODIES, MAX_FACES, visitCountKey } from './customers'

const FOCUSABLE = 'a[href], button:not([disabled]), input, select, [tabindex]:not([tabindex="-1"])'

export function CustomerPanel({ id, onClose, onErased, chairName }: {
  id: string
  onClose: () => void
  onErased: () => void
  /** a chair id by the shop's name (ph5-R1 is "R2") */
  chairName: (id: string) => string
}) {
  const { t, fmt, lang } = useLang()
  const [detail, setDetail] = useState<CustomerDetail | null>(null)
  const [error, setError] = useState<'gone' | 'failed' | null>(null)
  const [shown, setShown] = useState(false)
  const [confirm, setConfirm] = useState(false)
  const [busy, setBusy] = useState(false)
  const [eraseFailed, setEraseFailed] = useState(false)
  const panelRef = useRef<HTMLElement>(null)

  useEffect(() => {
    let off = false
    api.customer(id)
      .then(d => { if (!off) setDetail(d) })
      .catch(e => { if (!off) setError(e instanceof ApiError && e.status === 404 ? 'gone' : 'failed') })
    return () => { off = true }
  }, [id])

  // Slide in (the transition runs from the off-screen start state), focus the panel.
  useEffect(() => {
    panelRef.current?.focus()
    const frame = requestAnimationFrame(() => setShown(true))
    return () => cancelAnimationFrame(frame)
  }, [])

  // Escape: cancel the confirmation if it is up, else close the panel.
  const confirmRef = useRef(confirm)
  confirmRef.current = confirm
  useEffect(() => {
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key !== 'Escape') return
      e.preventDefault()
      if (confirmRef.current) setConfirm(false)
      else onClose()
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose])

  // The confirmation takes focus while it is open, and gives it back to Erase.
  const wasConfirm = useRef(false)
  useEffect(() => {
    if (confirm) document.querySelector<HTMLButtonElement>('.confirmBox .btnPlain')?.focus()
    else if (wasConfirm.current) panelRef.current?.querySelector<HTMLButtonElement>('.custErase')?.focus()
    wasConfirm.current = confirm
  }, [confirm])

  const trapTab = (e: KeyboardEvent) => {
    if (e.key !== 'Tab' || !panelRef.current) return
    const items = [...panelRef.current.querySelectorAll<HTMLElement>(FOCUSABLE)]
    if (!items.length) return
    const firstEl = items[0], lastEl = items[items.length - 1]
    if (e.shiftKey && (document.activeElement === firstEl || document.activeElement === panelRef.current)) {
      e.preventDefault(); lastEl.focus()
    } else if (!e.shiftKey && document.activeElement === lastEl) {
      e.preventDefault(); firstEl.focus()
    }
  }

  const erase = () => {
    if (busy) return
    setBusy(true)
    setEraseFailed(false)
    api.eraseCustomer(id)
      .then(() => { setConfirm(false); onErased() })
      .catch(() => { setBusy(false); setConfirm(false); setEraseFailed(true) })
  }

  const entries = useMemo(() => (detail ? entryRows(detail.entries, lang) : []), [detail, lang])
  const visits = useMemo(() => (detail ? visitRows(detail.visits, lang, chairName) : []), [detail, lang, chairName])

  return (
    <>
      <div className={shown ? 'panelScrim panelScrimShown' : 'panelScrim'} onClick={onClose} />
      <aside ref={panelRef} className={shown ? 'custPanel custPanelShown' : 'custPanel'} role="dialog" aria-modal="true"
             aria-labelledby="custPanelTitle" tabIndex={-1} onKeyDown={trapTab}>
        <header className="custPanelHead">
          <div>
            <h2 id="custPanelTitle">{t('cu.customer')}</h2>
            {detail && (
              <p className="custPanelMeta">
                <b><NumIn text={t(visitCountKey(detail.visit_count), { n: '{n}' })} n={fmt.number(detail.visit_count)} /></b>
                <span><span className="muted">{t('cu.firstSeen')}</span> {fmt.date(detail.first_seen)}</span>
                <span><span className="muted">{t('cu.lastSeen')}</span> {fmt.date(detail.last_seen)}</span>
              </p>
            )}
          </div>
          <button type="button" className="panelClose" aria-label={t('cu.close')} title={t('cu.close')} onClick={onClose}>
            <span aria-hidden="true">×</span>
          </button>
        </header>

        <div className="custPanelBody">
          {error && <p className="errorBar">{t(error === 'gone' ? 'cu.gone' : 'common.loadFailed')}</p>}
          {!detail && !error && <p className="muted">{t('common.loading')}</p>}
          {detail && (
            <>
              <section>
                <h3 className="sectionLabel">{t('cu.faces')}</h3>
                {detail.faces.length ? (
                  <div className="cropRow">
                    {detail.faces.slice(0, MAX_FACES).map(f => (
                      <Crop key={f.id} path={f.url} alt={t('cu.photo')} className="cropFace" />
                    ))}
                  </div>
                ) : <p className="muted small">{t('cu.noPhotos')}</p>}
              </section>
              <section>
                <h3 className="sectionLabel">{t('cu.bodies')}</h3>
                {detail.bodies.length ? (
                  <div className="cropRow">
                    {detail.bodies.slice(0, MAX_BODIES).map(b => (
                      <Crop key={b.id} path={b.url} alt={t('cu.photo')} className="cropBody" />
                    ))}
                  </div>
                ) : <p className="muted small">{t('cu.noPhotos')}</p>}
              </section>
              <section>
                <h3 className="sectionLabel">{t('cu.visits')}</h3>
                {visits.length ? (
                  <ul className="entryList">
                    {visits.map(v => (
                      <li key={v.id}>
                        <span>{v.date}</span> <span dir="ltr">{v.time}</span>{' '}
                        <span className={v.chairs.length ? 'visitChair' : 'visitChair muted'}>
                          {v.chairs.length ? t('cu.servedAt', { chairs: v.chairs.join(' + ') }) : t('cu.notServed')}
                        </span>
                      </li>
                    ))}
                  </ul>
                ) : <p className="muted small">{t('cu.noEntries')}</p>}
              </section>
              {/* a shop with a door camera (Bekky) also lists the door crossings; Palm Hills has none */}
              {entries.length > 0 && (
                <section>
                  <h3 className="sectionLabel">{t('cu.entries')}</h3>
                  <ul className="entryList">
                    {entries.map(e => (
                      <li key={e.id}><span>{e.date}</span> <span dir="ltr">{e.time}</span></li>
                    ))}
                  </ul>
                </section>
              )}
            </>
          )}
        </div>

        {detail && (
          <footer className="custPanelFoot">
            {eraseFailed && <p className="errorBar" role="alert">{t('cu.eraseFailed')}</p>}
            <button type="button" className="btnDanger custErase" disabled={busy} onClick={() => setConfirm(true)}>
              {t('cu.erase')}
            </button>
          </footer>
        )}
      </aside>
      {confirm && (
        <Confirm danger title={t('cu.eraseTitle')} confirmLabel={t('cu.erase')}
                 onConfirm={erase} onCancel={() => setConfirm(false)}>
          <p>{t('cu.eraseBody')}</p>
        </Confirm>
      )}
    </>
  )
}
