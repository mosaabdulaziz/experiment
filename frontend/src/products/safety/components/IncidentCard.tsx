/** One incident. The full card is IRIS's incident card (its stylesheet): the picture on top (IncidentMedia), then the
 *  time, the camera / feature / severity chips, what happened, where and how long, and -- when its feature asks for
 *  it -- the acknowledgement. The compact one (the Live list) is IRIS's alert card from its dashboard (its
 *  stylesheet): the chips and time, what happened, and its Preview incident button (the clip). */
import { useEffect, useState } from 'react'
import ec from '@/components/events/EventCard.module.css'
import ic from '@/components/incidents/IncidentCard.module.css'
import { IconPlay } from '@/components/ui/icons'
import { useLang } from '../i18n'
import { durationSeconds, isOpen, needsAck, toneOf } from '../lib/incidents'
import { mediaOf } from '../lib/media'
import { messageOf } from '../lib/messages'
import type { SafetyEvent, Settings } from '../lib/types'
import { FeatureTag } from './FeatureTag'
import { IncidentMedia } from './IncidentMedia'

interface Props {
  event: SafetyEvent
  settings: Settings | null
  cameraName: (id: string | null) => string
  onAcknowledge: (ev: SafetyEvent) => void
  onPlay: (ev: SafetyEvent) => void
  compact?: boolean
}

/** Re-render every 10 s while something is ongoing, so its running time stays current; once it ended, every minute,
 *  so its clip shows as expired when it is (a wall console stays open for days). */
function useNow(active: boolean): number {
  const [now, setNow] = useState(Date.now())
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), active ? 10000 : 60000)
    return () => clearInterval(t)
  }, [active])
  return now
}

type Severity = 'critical' | 'warning' | 'info'

export function IncidentCard({ event, settings, cameraName, onAcknowledge, onPlay, compact }: Props) {
  const { t, time, duration, lang } = useLang()
  const open = isOpen(event)
  const message = messageOf(event, lang, t, cameraName)
  const now = useNow(open)
  const seconds = durationSeconds(event, now)
  const waiting = needsAck(event, settings)
  const zone = typeof event.details?.zone_name === 'string' ? event.details.zone_name : null
  const severity = (['critical', 'warning', 'info'] as string[]).includes(event.severity ?? '')
    ? (event.severity as Severity) : null
  const facts = (
    <>
      {zone && <span>{t('incident.zone', { zone })}</span>}
      {seconds !== null && (
        <span className={open ? 'sf-status-open' : undefined}>
          {t(open ? 'incident.open' : 'incident.ended', { d: duration(seconds) })}
        </span>
      )}
    </>
  )

  if (compact) {
    const media = mediaOf(event, now)
    const playable = media.kind === 'clip' || (media.kind === 'snapshot' && !!media.clip)
    return (
      <article className={`${ec.card} ${ec.alert} sf-ecard`} data-waiting={waiting || undefined}
        data-tone={toneOf(event)} data-open={open || undefined}>
        <div className={ec.header}>
          {event.feature && <FeatureTag feature={event.feature} />}
          <span className={ec.cameraChip}>{cameraName(event.camera_id)}</span>
          {severity && <span className={`${ec.cameraChip} ${ec[`sev_${severity}`]}`}>{t(`severity.${severity}`)}</span>}
          <span className={ec.meta}><span className={ec.time}>{time(event.incident_started_at ?? event.timestamp)}</span></span>
        </div>
        <div className={ec.message} dir="auto">{message}</div>
        <div className="sf-icard-facts">{facts}</div>
        <div className={ec.clipRow}>
          {playable ? (
            <button type="button" className={ec.previewBtn} onClick={() => onPlay(event)}>
              <IconPlay width={10} height={10} />
              {t('incident.preview')}
            </button>
          ) : media.kind === 'pending' ? (
            <span className={ec.clipPending}><span className={ec.badgeDot} aria-hidden="true" />{t('incident.clipPending')}</span>
          ) : media.kind !== 'none' && (
            <span className={ec.clipNote}>{t(media.kind === 'expired' ? 'media.expired' : 'media.unavailable')}</span>
          )}
          {waiting && (
            <button type="button" className="iris-btn iris-btn-accent iris-btn-sm sf-ecard-ack"
              onClick={() => onAcknowledge(event)}>
              {t('incident.ack')}
            </button>
          )}
        </div>
      </article>
    )
  }

  return (
    <article className={`${ic.card} sf-icard`} data-tone={toneOf(event)} data-waiting={waiting || undefined}>
      {mediaOf(event, now).kind !== 'none' && <IncidentMedia event={event} onPlay={onPlay} />}
      <div className={ic.body}>
        <div className={ic.stamp}>{time(event.incident_started_at ?? event.timestamp)}</div>
        <div className={ic.chips}>
          <span className={ic.chip}>{cameraName(event.camera_id)}</span>
          {event.feature && <FeatureTag feature={event.feature} />}
          {severity && <span className={`${ic.chip} ${ic[`sev_${severity}`]}`}>{t(`severity.${severity}`)}</span>}
        </div>
        <div className={ic.title} dir="auto">{message}</div>
        <div className="sf-icard-facts">{facts}</div>
        {(waiting || event.acknowledged_at) && (
          <div className="sf-icard-ack">
            {waiting && (
              <button type="button" className="iris-btn iris-btn-accent" onClick={() => onAcknowledge(event)}>
                {t('incident.ack')}
              </button>
            )}
            {event.acknowledged_at && <span className="sf-muted">{t('incident.acked')} {time(event.acknowledged_at)}</span>}
          </div>
        )}
      </div>
    </article>
  )
}

export function ClipModal({ event, onClose, cameraName }: {
  event: SafetyEvent
  onClose: () => void
  cameraName: (id: string | null) => string
}) {
  const { t, lang } = useLang()
  const message = messageOf(event, lang, t, cameraName)
  const media = mediaOf(event, Date.now())
  const src = media.kind === 'clip' || media.kind === 'snapshot' ? media.clip : null
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  return (
    <div className="sf-modal" role="dialog" aria-modal="true" aria-label={message} onClick={onClose}>
      <div className="sf-panel sf-modal-box" onClick={e => e.stopPropagation()}>
        <div className="sf-page-head">
          <strong>{message}</strong>
          <button type="button" className="iris-btn" onClick={onClose} autoFocus>{t('common.close')}</button>
        </div>
        {src && <video src={src} controls autoPlay playsInline />}
      </div>
    </div>
  )
}
