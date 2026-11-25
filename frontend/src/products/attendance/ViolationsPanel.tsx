import { useState } from 'react'
import { monitoringApi, type Violation } from './api'

/** Restricted-zone violations events panel (2026-09-02, /grilling round 2
 * Q6: placed on the attendance page, not monitoring -- "same page as
 * attendance until the UI expert tells me where to put it"). The live red
 * highlight on the video itself lives on the monitoring page, where the
 * camera views are; this is just the list. */

const TYPE_LABEL: Record<Violation['violation_type'], string> = {
  role_not_allowed: 'دخول غير مصرّح',
  unidentified: 'شخص غير معروف',
}

function fmtTime(iso: string): string {
  return new Date(iso).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

export function ViolationsPanel({ violations, day }: { violations: Violation[]; day: string }) {
  const openCount = violations.filter(v => v.open).length
  // "Clear log" (2026-09-13): deletes only the day shown in the date picker,
  // only this log -- attendance is untouched. The list refreshes itself via
  // the websocket snapshot the server pushes after deleting.
  const dayCount = violations.filter(v => v.started_at.slice(0, 10) === day).length
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const clearDay = async () => {
    if (!window.confirm(`سيتم حذف سجل المناطق المقيدة ليوم ${day} فقط، ولن يتأثر الحضور والانصراف. متابعة؟`)) return
    setBusy(true)
    setError(null)
    try {
      await monitoringApi.clearViolations(day)
    } catch {
      setError('تعذر مسح السجل')
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="tableCard violationsCard">
      <div className="violationsHead">
        <h2>الوصول غير المصرح به للمناطق المقيدة</h2>
        <div className="violationsTools">
          {openCount > 0 && <span className="violationsBadge">{openCount} مستمرة الآن</span>}
          {dayCount > 0 && (
            <button type="button" className="btnGhost noPrint" onClick={clearDay} disabled={busy}
                    title={`حذف سجل يوم ${day} فقط`}>
              {busy ? 'جارٍ المسح…' : 'مسح سجل هذا اليوم'}
            </button>
          )}
        </div>
      </div>
      {error && <div className="errorBar">{error}</div>}
      <table className="attTable">
        <thead>
          <tr>
            <th>النوع</th>
            <th>الموظف</th>
            <th>الكاميرا</th>
            <th>المنطقة</th>
            <th>البداية</th>
            <th>الحالة</th>
          </tr>
        </thead>
        <tbody>
          {violations.map(v => (
            <tr key={v.id}>
              <td>
                <span className={`chip ${v.violation_type === 'unidentified' ? 'chip-absent' : 'chip-leave'}`}>
                  <span className="chipDot" aria-hidden="true" />
                  {TYPE_LABEL[v.violation_type]}
                </span>
              </td>
              <td>{v.name ?? '—'}</td>
              <td dir="ltr">{v.camera}</td>
              <td dir="ltr">{v.zone_id}</td>
              <td className="cellTime">{fmtTime(v.started_at)}</td>
              <td>
                <span className={v.open ? 'chip chip-absent' : 'chip chip-present'}>
                  <span className="chipDot" aria-hidden="true" />
                  {v.open ? 'مستمرة' : 'انتهت'}
                </span>
              </td>
            </tr>
          ))}
          {violations.length === 0 && (
            <tr><td colSpan={6} className="tableEmpty">لا توجد مخالفات مسجلة</td></tr>
          )}
        </tbody>
      </table>
    </section>
  )
}
