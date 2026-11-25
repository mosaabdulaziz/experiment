/** The verification alarm card: a suspected concealment candidate, in the page's flow.
 *
 * It never covers the camera. The card sits in the side column beside the picture, the
 * camera is ringed in red (high tier) or amber (medium tier), and the operator keeps
 * watching while verifying.
 */
import { useEffect, useRef, useState } from 'react'
import { FrameLayer } from '../live/FrameLayer'
import { durationSeconds, pct, stageChainLabel, targetLabel, timestamp } from '../format'
import { canAcknowledge } from './queue'
import { effectiveSeverity } from './severity'
import type { Candidate } from '../api'

export function AlarmCard({ event, queued, shownAt, onAcknowledge, onShowCamera }: {
  event: Candidate
  queued: number
  shownAt: number
  onAcknowledge: () => void
  onShowCamera: () => void
}) {
  const [now, setNow] = useState(Date.now())
  const armed = canAcknowledge(now, shownAt)
  const card = useRef<HTMLDivElement | null>(null)
  const critical = effectiveSeverity(event) === 'critical'

  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 250)
    return () => clearInterval(id)
  }, [])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Enter') return
      if (document.activeElement !== card.current) return
      if (!armed) return
      onAcknowledge()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [armed, onAcknowledge])

  return (
    <div ref={card} className="alarmCard" role="alert" aria-live="assertive" tabIndex={0}>
      <div className={critical ? 'alarmBar alarmCritical' : 'alarmBar alarmWarning'}>
        <span>VERIFY · {event.tier.toUpperCase()}</span>
        <span>{event.location || event.camera}</span>
        {queued > 0 && <span className="alarmQueued">{queued} waiting</span>}
      </div>
      <div className="alarmBody">
        <div className="alarmFacts">
          <span className="alarmEval">
            {targetLabel(event.target_kind, event.body_zone_name)} · zone {event.zone_id || '—'}
          </span>
          <span className="alarmMeta">
            score {pct(event.score)} · {timestamp(event.display_time, event.pts)} · track #{event.track_id}
          </span>
          <span className="alarmMeta">
            {stageChainLabel(event.stage_timeline, event.terminal_stage)} · dwell {durationSeconds(event.dwell_duration_s)} · hold {durationSeconds(event.hold_duration_s)}
          </span>
        </div>
        <div className="alarmThumb">
          <div className="feed"><FrameLayer camera={event.camera} intervalMs={500} /></div>
        </div>
      </div>
      <div className="alarmActions">
        <button type="button" className="btnPrimary btnSmall" disabled={!armed} onClick={onAcknowledge}>
          acknowledge
        </button>
        <button type="button" className="btnGhost btnSmall" onClick={onShowCamera}>
          show camera
        </button>
      </div>
    </div>
  )
}
