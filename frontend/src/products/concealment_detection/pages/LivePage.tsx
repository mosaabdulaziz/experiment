/** The Live page: the camera wide, the side column beside it.
 *
 * The camera tile is never covered by the alarm card. The alarm rings the camera in red (high)
 * or amber (medium) while the verification card sits in the right column alongside recent
 * candidates and the insufficient-evidence log (AC-4, AC-5, AC-6).
 */
import { useEffect } from 'react'
import { AlarmCard } from '../alerts/AlarmCard'
import { effectiveSeverity } from '../alerts/severity'
import { FrameLayer } from '../live/FrameLayer'
import { assessmentReasonLabel, pct, severityChip, shortTimestamp, targetLabel } from '../format'
import type { PageProps } from '../App'

export function LivePage({
  status, cameras, camera, setCamera, error, reviews, assessments, events, notices, markSeen,
  dismissNotice, alarm, alarmQueued, alarmShownAt, ackAlarm, showCamera,
}: PageProps) {
  const ready = status?.ready === true
  const awaiting = reviews.filter(r => r.status === 'candidate').length
  const elevated = reviews.filter(r => {
    const level = effectiveSeverity(r)
    return level === 'critical' || level === 'warning'
  }).length
  const concerns = reviews.filter(r => r.status === 'concern_noted').length
  const alarmCritical = alarm != null && effectiveSeverity(alarm) === 'critical'

  const healthEntries = Object.entries(status?.cameras ?? {})
  const okCount = healthEntries.filter(([, h]) => h.camera_ok).length
  const totalCams = Math.max(healthEntries.length, cameras.length)
  const currentCam = cameras.find(c => c.id === camera)

  useEffect(() => {
    if (events.length) markSeen(events[0].id)
  }, [events, markSeen])

  return (
    <>
      <div className="pageHead">
        <div className="pageHeadMain">
          <h1>Concealment Review</h1>
          {status && (
            <span className="subLine">
              {status.running ? 'running' : 'stopped'} · durable clips {status.capabilities?.durable_clips ? 'on' : 'off'} ·
              Suspected activity only · requires human verification
            </span>
          )}
        </div>
        <div className="pageHeadTools">
          {status && (
            <span className={ready ? 'chip chipGreen' : 'chip chipAmber'}>{ready ? 'ready' : 'not ready'}</span>
          )}
          {cameras.length > 0 && (
            <select value={camera} onChange={e => setCamera(e.target.value)} aria-label="camera">
              {cameras.map(c => (
                <option key={c.id} value={c.id}>
                  {c.location && c.location !== c.id ? `${c.location} (${c.id})` : c.id}
                </option>
              ))}
            </select>
          )}
        </div>
      </div>

      {error && <div className="errorBar">{error}</div>}

      <div className="kpiGrid">
        <div className="kpi"><div className="kpiLabel">awaiting review</div><div className="kpiValue num">{awaiting}</div></div>
        <div className="kpi"><div className="kpiLabel">high / warning</div><div className="kpiValue num">{elevated}</div></div>
        <div className="kpi"><div className="kpiLabel">concern noted</div><div className="kpiValue num">{concerns}</div></div>
        <div className="kpi"><div className="kpiLabel">cameras ok</div><div className="kpiValue num">{totalCams > 0 ? `${okCount}/${totalCams}` : '0'}</div></div>
        <div className="kpi"><div className="kpiLabel">insufficient logs</div><div className="kpiValue num">{assessments.length}</div></div>
      </div>

      <div className="liveGrid">
        <div className="feedTile">
          {camera
            ? <div className={alarm && alarm.camera === camera
                ? (alarmCritical ? 'feed feedRing' : 'feed feedRingWarn')
                : 'feed'}>
                <FrameLayer camera={camera} />
                {alarm && alarm.camera === camera && (
                  <span className={alarmCritical ? 'feedBadge feedLive' : 'feedBadge feedWarn'}>
                    {alarmCritical ? 'VERIFY · HIGH' : 'VERIFY'}
                  </span>
                )}
              </div>
            : <div className="feed" />}
          <div className="feedCaption">
            <span className="feedName">{currentCam?.location || camera || 'no camera'}</span>
            <span className="feedClip">the engine's published frame</span>
          </div>
        </div>
        <div className="sideCol">
          {alarm && (
            <AlarmCard
              event={alarm}
              queued={alarmQueued}
              shownAt={alarmShownAt}
              onAcknowledge={() => void ackAlarm()}
              onShowCamera={() => showCamera(alarm.camera)}
            />
          )}
          {notices.map(n => {
            const level = effectiveSeverity(n)
            return (
              <div
                key={n.id}
                className={
                  level === 'critical'
                    ? 'notice noticeCritical'
                    : level === 'warning'
                      ? 'notice noticeWarning'
                      : 'notice'
                }
                role="status"
              >
                <button
                  type="button"
                  className="noticeMain"
                  onClick={() => showCamera(n.camera)}
                  title={`View ${n.location || n.camera}`}
                >
                  {targetLabel(n.target_kind, n.body_zone_name)} · {pct(n.score)} · {n.location || n.camera}
                </button>
                <button
                  type="button"
                  className="noticeClose"
                  aria-label="dismiss notice"
                  onClick={() => dismissNotice(n.id)}
                >
                  ×
                </button>
              </div>
            )
          })}
          <div className="card eventsCard">
            <div className="sectionLabel">the candidates</div>
            <ul className="eventList">
              {reviews.slice(0, 12).map(e => {
                const level = effectiveSeverity(e)
                return (
                  <li key={e.id} className="eventRow">
                    <span
                      className={
                        level === 'critical'
                          ? 'eventDot eventDotCritical'
                          : level === 'warning'
                            ? 'eventDot eventDotWarning'
                            : 'eventDot'
                      }
                    />
                    <span className="eventEval">{targetLabel(e.target_kind, e.body_zone_name)}</span>
                    <span className="eventScore">{e.location || e.camera} · {pct(e.score)}</span>
                    <span className={`chip ${severityChip(level)}`}>{e.tier}</span>
                  </li>
                )
              })}
              {reviews.length === 0 && (
                <li className="eventRow"><span className="eventEval">nothing seen yet</span></li>
              )}
            </ul>
          </div>
          <div className="card assessmentsCard">
            <div className="sectionLabel">insufficient evidence</div>
            <ul className="eventList">
              {assessments.slice(0, 6).map(a => (
                <li key={a.id} className="eventRow">
                  <span className="eventDot" />
                  <span className="eventEval">{assessmentReasonLabel(a.reason)}</span>
                  <span className="eventScore">
                    {a.camera}
                    {a.track_id >= 0 ? ` · #${a.track_id}` : ''} · {shortTimestamp(a.created_at, a.pts)}
                  </span>
                </li>
              ))}
              {assessments.length === 0 && (
                <li className="eventRow"><span className="eventEval">no insufficient-evidence logs</span></li>
              )}
            </ul>
          </div>
        </div>
      </div>
    </>
  )
}
