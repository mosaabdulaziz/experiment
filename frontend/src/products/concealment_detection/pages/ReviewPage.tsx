/** The Review page: the suspected candidate queue, in the house's table, with the three non-accusatory verdicts. */
import { useState } from 'react'
import { ALLOWED_VERDICTS, type ReviewStatus } from '../api'
import { AlarmCard } from '../alerts/AlarmCard'
import { durationSeconds, pct, stageChainLabel, statusChip, targetLabel, tierChip, timestamp } from '../format'
import type { PageProps } from '../App'

export function ReviewPage({
  reviews, cameras, error, busy, decide, alarm, alarmQueued, alarmShownAt, ackAlarm, showCamera,
}: PageProps) {
  const [cameraFilter, setCameraFilter] = useState('')
  const [statusFilter, setStatusFilter] = useState<ReviewStatus | ''>('')
  const [notes, setNotes] = useState<Record<number, string>>({})

  const filtered = reviews.filter(row => {
    if (cameraFilter && row.camera !== cameraFilter) return false
    if (statusFilter && row.status !== statusFilter) return false
    return true
  })

  return (
    <>
      <div className="pageHead">
        <div className="pageHeadMain">
          <h1>Review</h1>
          <span className="subLine">suspected incidents requiring human verification</span>
        </div>
        <div className="pageHeadTools">
          <select value={cameraFilter} onChange={e => setCameraFilter(e.target.value)} aria-label="filter camera">
            <option value="">all cameras</option>
            {cameras.map(c => (
              <option key={c.id} value={c.id}>
                {c.location && c.location !== c.id ? `${c.location} (${c.id})` : c.id}
              </option>
            ))}
          </select>
          <select
            value={statusFilter}
            onChange={e => setStatusFilter(e.target.value as ReviewStatus | '')}
            aria-label="filter status"
          >
            <option value="">all statuses</option>
            <option value="candidate">candidate</option>
            <option value="acknowledged">acknowledged</option>
            <option value="concern_noted">concern noted</option>
            <option value="dismissed">dismissed</option>
          </select>
        </div>
      </div>

      {error && <div className="errorBar">{error}</div>}

      {alarm && (
        <AlarmCard
          event={alarm}
          queued={alarmQueued}
          shownAt={alarmShownAt}
          onAcknowledge={() => void ackAlarm()}
          onShowCamera={() => showCamera(alarm.camera)}
        />
      )}

      <div className="card">
        <div className="tableScroll">
          <table className="dataTable">
            <thead>
              <tr>
                <th>time</th>
                <th>camera / zone</th>
                <th>track</th>
                <th>target</th>
                <th>stages</th>
                <th>tier</th>
                <th>score</th>
                <th>status</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {filtered.map(row => {
                const stamp = timestamp(row.display_time, row.pts)
                const splitIdx = stamp.length === 19 && stamp[10] === ' ' ? 10 : -1
                return (
                  <tr key={row.id}>
                    <td className="num timeCell">
                      {splitIdx > 0 ? stamp.slice(11) : stamp}
                      {splitIdx > 0 && <span className="metaSub">{stamp.slice(0, 10)}</span>}
                    </td>
                    <td>
                      {row.location || row.camera}
                      <span className="metaSub">{row.camera} · zone {row.zone_id || '—'}</span>
                    </td>
                    <td className="num">#{row.track_id}</td>
                    <td>{targetLabel(row.target_kind, row.body_zone_name)}</td>
                    <td>
                      <span className="stageChain">{stageChainLabel(row.stage_timeline, row.terminal_stage)}</span>
                      <span className="metaSub">
                        dwell {durationSeconds(row.dwell_duration_s)} · hold {durationSeconds(row.hold_duration_s)}
                      </span>
                    </td>
                    <td><span className={tierChip(row.tier)}>{row.tier}</span></td>
                    <td className="num">{pct(row.score)}</td>
                    <td>
                      <span className={statusChip(row.status)}>{row.status.replace(/_/g, ' ')}</span>
                      {row.note && <span className="metaSub">{row.note}</span>}
                    </td>
                    <td>
                      {row.status === 'candidate' && (
                        <div className="actionGroup">
                          <input
                            type="text"
                            className="noteInput"
                            placeholder="optional note"
                            value={notes[row.id] ?? ''}
                            onChange={e => setNotes(prev => ({ ...prev, [row.id]: e.target.value }))}
                          />
                          {ALLOWED_VERDICTS.map(v => (
                            <button
                              key={v}
                              type="button"
                              className="btnGhost btnSmall"
                              disabled={busy === row.id}
                              onClick={() => void decide(row.id, v, notes[row.id])}
                            >
                              {v.replace(/_/g, ' ')}
                            </button>
                          ))}
                        </div>
                      )}
                    </td>
                  </tr>
                )
              })}
              {filtered.length === 0 && (
                <tr><td className="tableEmpty" colSpan={9}>nothing waiting for review</td></tr>
              )}
            </tbody>
          </table>
        </div>
        <div className="tableFoot">
          <span>{filtered.length} {filtered.length === 1 ? 'candidate' : 'candidates'} shown</span>
        </div>
      </div>
    </>
  )
}
