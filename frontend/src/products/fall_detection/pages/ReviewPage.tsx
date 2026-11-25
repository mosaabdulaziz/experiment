/** The Review page: the queue, in the house's table, with the three verdicts. */
import { AlarmCard } from '../alerts/AlarmCard'
import { pct, seconds, statusChip } from '../format'
import type { PageProps } from '../App'

const VERDICTS = ['acknowledged', 'confirmed', 'dismissed'] as const

export function ReviewPage({ reviews, error, busy, decide, alarm, alarmQueued, alarmShownAt, ackAlarm, showCamera }: PageProps) {
  return (
    <>
      <div className="pageHead">
        <div className="pageHeadMain">
          <h1>Review</h1>
          <span className="subLine">the candidates the product could not confirm</span>
        </div>
      </div>

      {error && <div className="errorBar">{error}</div>}

      {alarm && (
        <AlarmCard event={alarm} queued={alarmQueued} shownAt={alarmShownAt}
                   onAcknowledge={() => void ackAlarm()}
                   onShowCamera={() => showCamera(alarm.camera)} />
      )}

      <div className="card">
        <div className="tableScroll">
          <table className="dataTable">
            <thead>
              <tr>
                <th>time</th><th>camera</th><th>track</th><th>kind</th><th>evaluation</th>
                <th>score</th><th>paths</th><th>status</th><th />
              </tr>
            </thead>
            <tbody>
              {reviews.map(row => (
                <tr key={row.id}>
                  <td>{row.pts == null ? row.created_at : seconds(row.pts)}</td>
                  <td>{row.camera}</td>
                  <td>{row.track ?? '—'}</td>
                  <td>{row.kind ?? '—'}</td>
                  <td>{row.evaluation}</td>
                  <td className="num">{pct(row.combined)}</td>
                  <td>
                    action {pct(row.path_scores.action)} · kinematic {pct(row.path_scores.kinematic)} ·
                    cold {pct(row.path_scores.cold)}
                  </td>
                  <td><span className={statusChip(row.status)}>{row.status}</span></td>
                  <td>
                    {row.status === 'candidate' && VERDICTS.map(v => (
                      <button key={v} type="button" className="btnGhost btnSmall"
                              disabled={busy === row.id} onClick={() => void decide(row.id, v)}>
                        {v}
                      </button>
                    ))}
                  </td>
                </tr>
              ))}
              {reviews.length === 0 && (
                <tr><td className="tableEmpty" colSpan={9}>nothing waiting for review</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </>
  )
}
