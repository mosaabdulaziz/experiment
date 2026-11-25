/** Report panel (2026-10-15): a day / 7 days from a day / a month, a language and a format; the server builds the
 * file (GET /api/barber/report). "View" opens the HTML in a new tab; PDF and Word download. With "Daily", the haircut
 * photo report too (2026-10-22, shop network only). */
import { useEffect, useState } from 'react'
import { api, reportStart, reportUrl, type ReportFormat, type ReportKind } from '../api/client'
import { useLang } from '../i18n'
import { photoLinks } from './reportPhotos'

const KINDS: ReportKind[] = ['day', 'week', 'month']

export function ReportPanel({ day }: { day: string }) {
  const { t, lang } = useLang()
  const [open, setOpen] = useState(false)
  const [kind, setKind] = useState<ReportKind>('day')
  // null = follow the page (its day, the dashboard's language) until the user picks; reset on close
  const [picked, setPicked] = useState<string | null>(null)
  const [pickedLang, setPickedLang] = useState<'en' | 'ar' | null>(null)
  const [allowed, setAllowed] = useState<boolean | null>(null)
  useEffect(() => {
    if (!open || allowed !== null) return
    api.photoReportAllowed().then(r => setAllowed(r.allowed)).catch(() => setAllowed(false))
  }, [open, allowed])
  const pick = reportStart(kind, picked, day)
  const repLang = pickedLang ?? lang
  const url = (format: ReportFormat, download = false) => reportUrl({ kind, start: pick, lang: repLang, format, download })
  const photos = photoLinks(kind, pick, repLang, allowed)
  const close = () => { setOpen(false); setPicked(null); setPickedLang(null) }
  if (!open) return <button type="button" className="btnGhost" onClick={() => setOpen(true)}>{t('rep.button')}</button>
  return (
    <div className="card reportPanel" role="group" aria-label={t('rep.title')}>
      <span className="seg" role="group" aria-label={t('rep.kind')}>
        {KINDS.map(k => (
          <button key={k} type="button" aria-pressed={kind === k} className={kind === k ? 'segBtn segActive' : 'segBtn'}
                  onClick={() => setKind(k)}>{t(`rep.kind.${k}`)}</button>
        ))}
      </span>
      <label className="filterField">
        <span>{kind === 'month' ? t('rep.month') : t('rep.firstDay')}</span>
        <input className="input" type={kind === 'month' ? 'month' : 'date'} value={pick}
               onChange={e => e.target.value && setPicked(e.target.value)} />
      </label>
      <label className="filterField">
        <span>{t('rep.lang')}</span>
        <select className="select" value={repLang} onChange={e => setPickedLang(e.target.value as 'en' | 'ar')}>
          <option value="en">English</option>
          <option value="ar">العربية</option>
        </select>
      </label>
      <a className="btnPrimary" href={url('html')} target="_blank" rel="noreferrer">{t('rep.view')}</a>
      <a className="btnGhost" href={url('html', true)} download>{t('rep.htmlFile')}</a>
      <a className="btnGhost" href={url('pdf')} download>{t('rep.pdf')}</a>
      <a className="btnGhost" href={url('docx')} download>{t('rep.word')}</a>
      {photos.show && (
        <span className="filterField" role="group" aria-label={t('rep.photos')}>
          <span>{t('rep.photos')}</span>
          {photos.enabled ? (
            <>
              <a className="btnGhost" href={photos.pdf} download>{t('rep.pdf')}</a>
              <a className="btnGhost" href={photos.html} target="_blank" rel="noreferrer">{t('rep.htmlFile')}</a>
            </>
          ) : (
            <>
              <button type="button" className="btnGhost" disabled title={t('rep.photosLanOnly')}>{t('rep.pdf')}</button>
              <button type="button" className="btnGhost" disabled title={t('rep.photosLanOnly')}>{t('rep.htmlFile')}</button>
              <span className="note">{t('rep.photosLanOnly')}</span>
            </>
          )}
        </span>
      )}
      <button type="button" className="btnPlain" onClick={close}>{t('rep.close')}</button>
    </div>
  )
}
