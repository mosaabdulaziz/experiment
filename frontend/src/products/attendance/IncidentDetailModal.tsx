import { useEffect, useState } from 'react'
import { alertClipUrl, type CameraInfo, type IrisEvent } from './api'
import { CAM_LABELS } from './components'
import {
  KIND_LABEL, camLabel, effectiveClips, fireKind, fmtDateTime, fmtDuration, onsetOf,
} from './useFireAlerts'

/** Incident detail (3d): player + lifecycle + event log, as a modal over
 * whichever page opened it (live wall or history) so the operator never
 * navigates away from what they were watching.
 *
 * The wireframe's scrub bar marks the exact detection window inside the
 * clip (pre-roll / detection / post-roll) using offsets the backend does not
 * currently expose per clip -- built with the browser's own <video controls>
 * instead of a custom scrub overlay, same as the page this replaces. The
 * "clip strip" becomes a plain numbered filmstrip when an incident has more
 * than one clip (see effectiveClips) -- no thumbnail images exist to show,
 * so numbered buttons are the honest version of that idea. */
export function IncidentDetailModal({ event, cameras, now, onClose, onAcknowledge, onJumpToLive }: {
  event: IrisEvent
  cameras: CameraInfo[]
  now: number
  onClose: () => void
  onAcknowledge: (id: number) => void
  onJumpToLive?: () => void
}) {
  const clips = effectiveClips(event)
  const [clipIndex, setClipIndex] = useState(0)
  useEffect(() => { setClipIndex(0) }, [event.id])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const kind = fireKind(event)
  const cam = cameras.find(c => c.id === event.camera) ?? null
  const streamName = cam ? camLabel(cam, CAM_LABELS) : (event.camera ?? '—')
  const active = event.incident_status === 'ongoing'
  const started = new Date(onsetOf(event)).getTime()
  const lastedSeconds = active
    ? (now - started) / 1000
    : event.incident_duration_seconds

  const clip = clips[clipIndex]

  const log: { time: string; label: string }[] = [
    { time: onsetOf(event), label: `${KIND_LABEL[kind]} — بدء الرصد` },
  ]
  if (event.acknowledged_at) log.push({ time: event.acknowledged_at, label: 'تم الإقرار' })
  if (event.incident_status === 'resolved' && event.incident_duration_seconds != null) {
    log.push({ time: new Date(started + event.incident_duration_seconds * 1000).toISOString(),
              label: 'تم رصد خلو المكان' })
  }
  log.sort((a, b) => new Date(a.time).getTime() - new Date(b.time).getTime())

  return (
    <div className="camModalBackdrop" onClick={onClose} role="presentation">
      <div className="incidentModal" onClick={e => e.stopPropagation()} role="dialog" aria-modal="true">
        <div className="camModalHead">
          <div className="camModalTitle">
            <span className={`chip ${kind === 'smoke' ? 'chip-leave' : 'chip-absent'}`}>
              {KIND_LABEL[kind]} — {active ? 'مستمر' : 'منتهٍ'}
            </span>
            <strong dir="ltr">{streamName}</strong>
            {cam?.zone && <span className="camModalClock">{cam.zone}</span>}
          </div>
          <div className="camModalActions">
            <button className="camModalClose" onClick={onClose} aria-label="إغلاق">✕</button>
          </div>
        </div>

        <div className="incidentModalBody">
          <div className="incidentModalMain">
            {clip ? (
              <video key={clip.clip_id} className="camFrame incidentPlayer"
                    src={alertClipUrl(clip.clip_id, event.camera ?? '')}
                    controls autoPlay muted playsInline />
            ) : (
              <div className="camOffline">لا يوجد مقطع متاح</div>
            )}
            {clips.length > 1 && (
              <div className="clipStrip" role="tablist" aria-label="مقاطع الحادثة">
                {clips.map((c, i) => (
                  <button
                    key={c.clip_id}
                    type="button"
                    role="tab"
                    aria-selected={i === clipIndex}
                    className={i === clipIndex ? 'clipStripBtn clipStripBtnActive' : 'clipStripBtn'}
                    onClick={() => setClipIndex(i)}
                  >
                    مقطع {i + 1}
                  </button>
                ))}
              </div>
            )}
            {event.description && (
              <p className="fireDescription" dir="auto">{event.description}</p>
            )}
          </div>

          <div className="incidentModalRail">
            <div className="incidentLifecycle">
              <span className="incidentLifecycleLabel">النوع</span><span>{KIND_LABEL[kind]}</span>
              <span className="incidentLifecycleLabel">الكاميرا</span><span dir="ltr">{streamName}</span>
              {cam?.zone && <><span className="incidentLifecycleLabel">المنطقة</span><span>{cam.zone}</span></>}
              <span className="incidentLifecycleLabel">بدأ في</span><span>{fmtDateTime(onsetOf(event))}</span>
              <span className="incidentLifecycleLabel">انتهى في</span>
              <span className={active ? 'incidentActiveText' : undefined}>
                {active ? '— لا يزال مستمراً' : (event.incident_duration_seconds != null
                  ? fmtDateTime(new Date(started + event.incident_duration_seconds * 1000).toISOString())
                  : '—')}
              </span>
              <span className="incidentLifecycleLabel">المدة</span>
              <span className={active ? 'incidentActiveText' : undefined}>
                {lastedSeconds != null ? fmtDuration(lastedSeconds) : '—'}
                {active ? ' ومستمرة' : ''}
              </span>
            </div>

            {active && (
              <div className="incidentNote">
                تستمر المدة في العد حتى يبلّغ نظام الكشف عن خلو الكاميرا من الحريق/الدخان،
                عندها تُغلق الحادثة بمدة نهائية.
              </div>
            )}

            <div className="incidentActions">
              {!event.acknowledged_at && (
                <button type="button" className="btnPrimary btnSmall"
                       onClick={() => onAcknowledge(event.id)}>إقرار</button>
              )}
              {active && onJumpToLive && (
                <button type="button" className="btnGhost btnSmall" onClick={onJumpToLive}>
                  الانتقال للبث المباشر
                </button>
              )}
            </div>

            <div className="incidentEventLog">
              <span className="incidentLifecycleLabel">سجل الأحداث</span>
              <ul>
                {log.map((l, i) => (
                  <li key={i} dir="ltr">{fmtDateTime(l.time)} — {l.label}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
