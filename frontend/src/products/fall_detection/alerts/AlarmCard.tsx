/** The alarm card: a fall, in the page's flow.
 *
 * It never covers the camera. The card sits in the side column beside the picture, the
 * camera that fell is ringed, and the operator keeps watching while they decide -- which is
 * the whole point: the fall happened in that frame (phase-5 §3).
 */
import { useEffect, useRef, useState } from 'react'
import { FrameLayer } from '../live/FrameLayer'
import { pct, seconds } from '../format'
import { canAcknowledge } from './queue'
import type { FallEvent } from './severity'

export function AlarmCard({ event, queued, shownAt, onAcknowledge, onShowCamera }: {
  event: FallEvent
  queued: number
  shownAt: number
  onAcknowledge: () => void
  onShowCamera: () => void
}) {
  const [now, setNow] = useState(Date.now())
  const armed = canAcknowledge(now, shownAt)
  const card = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 250)
    return () => clearInterval(id)
  }, [])

  // ENTER acknowledges the card while it holds focus -- and the card does not steal the
  // focus from whatever the operator is doing.
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

  const critical = event.severity === 'critical'
  return (
    <div ref={card} className="alarmCard" role="alert" aria-live="assertive" tabIndex={0}>
      <div className={critical ? 'alarmBar alarmCritical' : 'alarmBar alarmWarning'}>
        <span>{critical ? 'FALL' : 'ALERT'}</span>
        <span>{event.camera}</span>
        {queued > 0 && <span className="alarmQueued">{queued} waiting</span>}
      </div>
      <div className="alarmBody">
        <div className="alarmFacts">
          <span className="alarmEval">{event.evaluation}</span>
          <span className="alarmMeta">score {pct(event.combined)} · {seconds(event.pts)}</span>
          <span className="alarmMeta">track {event.track ?? '—'}</span>
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
