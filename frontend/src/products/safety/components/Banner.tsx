/** Incidents waiting for an acknowledgement, on every page until someone acknowledges each -- even after the
 *  incident ended (decisions D26/D35). No sound. A strip, side by side: each with its picture (IncidentMedia), its
 *  feature, when, which camera, and Acknowledge; the rest behind "+N more" (the waiting list on Incidents). */
import { navigate } from '../../../shared/router'
import { useLang } from '../i18n'
import { isOpen, toneOf } from '../lib/incidents'
import { messageOf } from '../lib/messages'
import type { SafetyEvent } from '../lib/types'
import { FeatureTag } from './FeatureTag'
import { IncidentMedia } from './IncidentMedia'

const SHOWN = 10

interface Props {
  waiting: SafetyEvent[]
  cameraName: (id: string | null) => string
  onAcknowledge: (ev: SafetyEvent) => void
  onPlay: (ev: SafetyEvent) => void
}

export function Banner({ waiting, cameraName, onAcknowledge, onPlay }: Props) {
  const { t, time, lang } = useLang()
  if (!waiting.length) return null
  const review = () => navigate('/incidents?status=unacked')
  return (
    <section className="sf-banner" aria-label={t('banner.count', { n: waiting.length })}>
      <div className="sf-banner-head">
        <span className="sf-banner-count" role="status" aria-live="polite">{t('banner.count', { n: waiting.length })}</span>
        <button type="button" className="iris-btn iris-btn-sm" onClick={review}>{t('banner.review')}</button>
      </div>
      <ul className="sf-banner-strip">
        {waiting.slice(0, SHOWN).map(ev => (
          <li key={ev.id} className="sf-bcard" data-tone={toneOf(ev)}>
            <IncidentMedia event={ev} onPlay={onPlay} className="sf-bcard-media" />
            <div className="sf-bcard-body">
              <div className="sf-bcard-top">
                {ev.feature && <FeatureTag feature={ev.feature} />}
                <span className="sf-mono">{time(ev.incident_started_at ?? ev.timestamp)}</span>
                {!isOpen(ev) && ev.incident_status && <span>{t('banner.ended')}</span>}
              </div>
              <div className="sf-bcard-msg" title={messageOf(ev, lang, t, cameraName)} dir="auto">
                {messageOf(ev, lang, t, cameraName)}
              </div>
              <div className="sf-bcard-cam">{cameraName(ev.camera_id)}</div>
              <button type="button" className="iris-btn iris-btn-accent iris-btn-sm" onClick={() => onAcknowledge(ev)}>
                {t('incident.ack')}
              </button>
            </div>
          </li>
        ))}
        {waiting.length > SHOWN && (
          <li className="sf-bcard sf-bcard-more">
            <button type="button" className="iris-btn" onClick={review}>{t('banner.more', { n: waiting.length - SHOWN })}</button>
          </li>
        )}
      </ul>
    </section>
  )
}
