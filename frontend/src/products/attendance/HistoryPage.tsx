import { useEffect, useState } from 'react'
import { historyApi, type DayRow, type EmployeeHistory } from './api'
import { fmtDuration } from './MonitoringPage'
import { productHref } from '../../shared/router'

function fmtClockSec(iso: string): string {
  return new Date(iso).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

// Every visit of the day, one per line (2026-09-14, later the same day: the
// user asked for all intervals again instead of only first/last). A visit is
// a session: sightings closer together than the exit gap count as one.
function arrivalDeparture(row: DayRow) {
  if (row.sessions.length === 0) return '—'
  return (
    <ul className="intervalList">
      {row.sessions.map(s => (
        <li key={s.entry} dir="ltr">{`${fmtClockSec(s.entry)}–${s.exit ? fmtClockSec(s.exit) : 'الآن'}`}</li>
      ))}
    </ul>
  )
}

function todayIso(): string {
  const d = new Date()
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
}

// Notes (2026-09-14): "غائب" only for a day that has started with no sighting
// at all; a day that hasn't happened yet, or a present day with nothing to
// flag, shows "غير مدرج".
function flags(row: DayRow): { label: string; cls: string }[] {
  const notListed = { label: 'غير مدرج', cls: 'chip-neutral' }
  if (row.day > todayIso()) return [notListed]
  if (row.absent) return [{ label: 'غائب', cls: 'chip-absent' }]
  const out: { label: string; cls: string }[] = []
  if (row.early_arrival) out.push({ label: 'حضور مبكر', cls: 'chip-early' })
  if (row.late) out.push({ label: 'تأخير', cls: 'chip-leave' })
  if (row.early_leave) out.push({ label: 'انصراف مبكر', cls: 'chip-leave' })
  if (row.overtime_seconds > 0) out.push({ label: 'عمل إضافي', cls: 'chip-present' })
  return out
}

export function HistoryPage({ employeeId }: { employeeId: string }) {
  const [month, setMonth] = useState(() => new Date().toISOString().slice(0, 7))
  const [data, setData] = useState<EmployeeHistory | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setError(null)
    historyApi.get(employeeId, month).then(setData).catch(e => setError(e instanceof Error ? e.message : 'فشل تحميل السجل'))
  }, [employeeId, month])

  // First load: adopt the month the backend actually has data for -- avoids
  // opening on an all-empty month by default. 2026-09-12 (issue 1, Phase 1):
  // now that the backend's own default is real date.today() too (no more
  // fake simulated day), this just re-confirms the same real current month
  // `month` above already initialized to -- kept anyway since it's still the
  // right behavior for a checkout whose data genuinely lives in a past month.
  useEffect(() => {
    historyApi.get(employeeId).then(d => setMonth(d.month)).catch(() => {})
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [employeeId])

  return (
    <>
      <div className="pageHead">
        <div>
          <h1>{data ? (data.display_name_ar ?? data.name) : '…'}</h1>
          <nav className="crumbs" aria-label="مسار التنقل">
            <span>الرئيسية</span><span className="crumbSep">‹</span>
            <a href={productHref('/admin/employees')}>إدارة الموظفين</a><span className="crumbSep">‹</span>
            <span className="crumbHere">السجل الشهري</span>
          </nav>
        </div>
        <input
          className="dayInput noPrint" type="month" value={month}
          onChange={e => setMonth(e.target.value)}
        />
      </div>

      {error && <div className="errorBar">{error}</div>}

      {data && (
        <>
          <div className="kpiRow">
            <div className="kpiCard"><div className="kpiHead"><span className="kpiLabel">أيام الحضور</span><span className="kpiIcon" /></div><div className="kpiValue">{data.summary.days_present}</div></div>
            <div className="kpiCard kpiLeave"><div className="kpiHead"><span className="kpiLabel">مرات التأخير</span><span className="kpiIcon" /></div><div className="kpiValue">{data.summary.times_late}</div></div>
            <div className="kpiCard kpiLeave"><div className="kpiHead"><span className="kpiLabel">مرات الانصراف المبكر</span><span className="kpiIcon" /></div><div className="kpiValue">{data.summary.times_early_leave}</div></div>
            <div className="kpiCard"><div className="kpiHead"><span className="kpiLabel">إجمالي العمل الإضافي</span><span className="kpiIcon" /></div><div className="kpiValue kpiTime">{fmtDuration(data.summary.total_overtime_seconds)}</div></div>
          </div>

          <section className="tableCard">
            <table className="attTable">
              <thead>
                <tr>
                  <th>اليوم</th>
                  <th>الحضور/الانصراف</th>
                  <th>صافي الساعات</th>
                  <th>ملاحظات</th>
                </tr>
              </thead>
              <tbody>
                {data.days.map(row => (
                  <tr key={row.day} className={row.absent ? 'rowInactive' : undefined}>
                    <td className="cellCode" dir="ltr">{row.day}</td>
                    <td className="cellTime">{arrivalDeparture(row)}</td>
                    <td className="cellTime cellWork">{row.absent ? '—' : fmtDuration(row.span_seconds)}</td>
                    <td>
                      {flags(row).map(f => (
                        <span key={f.label} className={`chip ${f.cls}`} style={{ marginInlineEnd: 4 }}>
                          <span className="chipDot" aria-hidden="true" />{f.label}
                        </span>
                      ))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </>
      )}
    </>
  )
}
