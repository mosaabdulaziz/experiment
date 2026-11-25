/** The Live page: the board. The camera wide, the side column beside it.
 *
 * The camera tile is never covered -- not by the alarm card, not by anything. The alarm
 * annotates the picture (a ring on the tile that fell) rather than sitting on top of it
 * (phase-5 §3).
 */
import { useEffect } from 'react'
import { AlarmCard } from '../alerts/AlarmCard'
import { FrameLayer } from '../live/FrameLayer'
import { pct, severityChip } from '../format'
import type { PageProps } from '../App'

export function LivePage({
  status, cameras, camera, setCamera, error, reviews, events, notices, markSeen,
  alarm, alarmQueued, alarmShownAt, ackAlarm, showCamera,
}: PageProps) {
  const ready = status?.classifier_ready === true && status.weights_verified === true
  const falls = events.filter(e => e.severity === 'critical').length
  const warnings = events.filter(e => e.severity === 'warning').length
  const awaiting = reviews.filter(r => r.status === 'candidate').length
  const last = events[0]

  // The Live page is where the operator looks; showing them is what clears the badge.
  useEffect(() => {
    if (events.length) markSeen(events[0].id)
  }, [events, markSeen])

  return (
    <>
      <div className="pageHead">
        <div className="pageHeadMain">
          <h1>Fall Detection</h1>
          {status && (
            <span className="subLine">
              device {status.device ?? '—'} · weights {status.weights_verified ? 'verified' : 'not verified'} ·
              classifier {status.classifier_ready ? 'ready' : 'NOT ready'} · paths {status.paths_live.join(', ')}
            </span>
          )}
        </div>
        <div className="pageHeadTools">
          {status && (
            <span className={ready ? 'chip chipGreen' : 'chip chipAmber'}>{ready ? 'ready' : 'not ready'}</span>
          )}
          {cameras.length > 1 && (
            <select value={camera} onChange={e => setCamera(e.target.value)} aria-label="camera">
              {cameras.map(id => <option key={id} value={id}>{id}</option>)}
            </select>
          )}
        </div>
      </div>

      {error && <div className="errorBar">{error}</div>}

      <div className="kpiGrid">
        <div className="kpi"><div className="kpiLabel">falls</div><div className="kpiValue num">{falls}</div></div>
        <div className="kpi"><div className="kpiLabel">cameras</div><div className="kpiValue num">{cameras.length}</div></div>
        <div className="kpi"><div className="kpiLabel">warnings</div><div className="kpiValue num">{warnings}</div></div>
        <div className="kpi"><div className="kpiLabel">awaiting review</div><div className="kpiValue num">{awaiting}</div></div>
        <div className="kpi"><div className="kpiLabel">last score</div><div className="kpiValue num">{pct(last?.combined)}</div></div>
      </div>

      <div className="liveGrid">
        <div className="feedTile">
          {camera
            ? <div className={alarm && alarm.camera === camera
                ? (alarm.severity === 'critical' ? 'feed feedRing' : 'feed feedRingWarn')
                : 'feed'}>
                <FrameLayer camera={camera} />
                {alarm && alarm.camera === camera && (
                  <span className={alarm.severity === 'critical' ? 'feedBadge feedLive' : 'feedBadge feedWarn'}>
                    {alarm.severity === 'critical' ? 'FALL' : 'ALERT'}
                  </span>
                )}
              </div>
            : <div className="feed" />}
          <div className="feedCaption">
            <span className="feedName">{camera || 'no camera'}</span>
            <span className="feedClip">the engine's published frame</span>
          </div>
        </div>
        <div className="sideCol">
          {alarm && (
            <AlarmCard event={alarm} queued={alarmQueued} shownAt={alarmShownAt}
                       onAcknowledge={() => void ackAlarm()}
                       onShowCamera={() => showCamera(alarm.camera)} />
          )}
          {notices.map(n => (
            <div key={n.id} className={n.severity === 'critical' ? 'notice noticeCritical' : 'notice noticeWarning'}
                 role="status">
              {n.evaluation} · {pct(n.combined)} · {n.camera}
            </div>
          ))}
          <div className="card eventsCard">
            <div className="sectionLabel">the events</div>
            <ul className="eventList">
              {events.map(e => (
                <li key={e.id} className="eventRow">
                  <span className={e.severity === 'critical' ? 'eventDot eventDotCritical'
                    : e.severity === 'warning' ? 'eventDot eventDotWarning' : 'eventDot'} />
                  <span className="eventEval">{e.evaluation}</span>
                  <span className="eventScore">{e.camera} · {pct(e.combined)}</span>
                  <span className={`chip ${severityChip(e.severity)}`}>{e.severity ?? 'closed'}</span>
                </li>
              ))}
              {events.length === 0 && <li className="eventRow"><span className="eventEval">nothing seen yet</span></li>}
            </ul>
          </div>
        </div>
      </div>
    </>
  )
}
