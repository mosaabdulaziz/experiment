import { useCallback, useEffect, useRef, useState } from 'react'
import { adminApi, type AdminEmployee, type AttendancePolicy, type MonitoringSnapshot, type Violation, type Zone } from './api'
import { ViolationsPanel } from './ViolationsPanel'
import { productHref } from '../../shared/router'

/** Access alerts (2026-11-02): a whitelist of authorised people, and an alert -- a popup at the side of every page and
 * a page of its own -- for anyone else. Nothing here detects anything: it shows the restricted-zone violations the
 * backend already raises (violations.py). A recognised person whose role is not on the zone's allow-list is
 * `role_not_allowed`; a person nobody recognises is `unidentified`.
 *
 * The whitelist IS the zones' `allowed_roles`: an employee is authorised when their role is on one of them, and the
 * switch on the Alerts page just sets the role -- the first allowed role to authorise, DENY_ROLE to remove. */

const DENY_ROLE = 'visitor'
const TOAST_MS = 15_000

const TITLE: Record<Violation['violation_type'], { ar: string; en: string }> = {
  role_not_allowed: { ar: 'دخول غير مصرّح', en: 'Not on the whitelist' },
  unidentified: { ar: 'شخص غير معروف', en: 'Unrecognised person' },
}

export function allowedRolesOf(zones: Record<string, Zone[]> | undefined, policy?: AttendancePolicy | null): string[] {
  const roles = new Set<string>()
  for (const zs of Object.values(zones ?? {})) {
    for (const z of zs) if (z.type === 'restricted') (z.allowed_roles ?? []).forEach(r => roles.add(r))
  }
  for (const cfg of Object.values(policy?.access_control ?? {})) cfg.allowed_roles.forEach(r => roles.add(r))
  return [...roles]
}

/** Where each camera's access check applies: its drawn restricted zones, else the whole stream (backend rule,
 * products/attendance/zones.py effective_restricted_zones). Only cameras with a check or a drawn zone are listed. */
export function checkScopes(zones: Record<string, Zone[]> | undefined, policy?: AttendancePolicy | null) {
  const cams = new Set<string>(Object.keys(policy?.access_control ?? {}))
  for (const [cam, zs] of Object.entries(zones ?? {})) if (zs.some(z => z.type === 'restricted')) cams.add(cam)
  return [...cams].sort().map(cam => {
    const drawn = (zones?.[cam] ?? []).filter(z => z.type === 'restricted')
    return { cam, zones: drawn.map(z => z.name || z.id), wholeStream: drawn.length === 0 }
  })
}

export const openAlertCount = (violations: Violation[]): number => violations.filter(v => v.open).length

function fmtTime(iso: string): string {
  return new Date(iso).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

const photoUrl = (employeeId: string) => `/attendance/api/employees/${employeeId}/photo`

function Who({ v, size = 44 }: { v: Violation; size?: number }) {
  return v.employee_id ? (
    <img className="alertPhoto" src={photoUrl(v.employee_id)} alt="" width={size} height={size} />
  ) : (
    <span className="alertPhoto alertPhotoUnknown" style={{ width: size, height: size }} aria-hidden="true">؟</span>
  )
}

/** The side popup. Every violation that is open and not yet shown pops up once (the ones already open when the page
 * loads too: an alert nobody has seen is exactly what this is for), stays 15 s or until closed. */
export function AccessAlertToasts({ violations }: { violations: Violation[] }) {
  const seen = useRef(new Set<string>())
  const [toasts, setToasts] = useState<Violation[]>([])

  useEffect(() => {
    const fresh = violations.filter(v => v.open && !seen.current.has(v.id))
    if (fresh.length === 0) return
    fresh.forEach(v => seen.current.add(v.id))
    setToasts(t => [...fresh, ...t].slice(0, 4))
    for (const v of fresh) {
      setTimeout(() => setToasts(t => t.filter(x => x.id !== v.id)), TOAST_MS)
    }
  }, [violations])

  const close = useCallback((id: string) => setToasts(t => t.filter(x => x.id !== id)), [])
  if (toasts.length === 0) return null
  return (
    <div className="alertToasts noPrint" role="region" aria-label="تنبيهات الدخول" aria-live="assertive">
      {toasts.map(v => {
        const live = violations.find(x => x.id === v.id) ?? v
        return (
          <div key={v.id} className={`alertToast alertToast-${v.violation_type}`} role="alert">
            <Who v={v} />
            <div className="alertToastBody">
              <strong>{TITLE[v.violation_type].ar}</strong>
              <span className="alertToastEn">{TITLE[v.violation_type].en}</span>
              <span className="alertToastMeta">
                {v.name ?? 'شخص غير معروف'} · <bdi dir="ltr">{v.camera}</bdi> · {fmtTime(v.started_at)}
                {!live.open && ' · انتهت'}
              </span>
              <a className="alertToastLink" href={productHref('/alerts')}>عرض التنبيهات</a>
            </div>
            <button type="button" className="alertToastClose" onClick={() => close(v.id)} aria-label="إغلاق">×</button>
          </div>
        )
      })}
    </div>
  )
}

/** The whitelist: who is authorised. Switching writes the employee's role (PATCH), the engine reads it next cycle. */
function Whitelist({ allowedRoles }: { allowedRoles: string[] }) {
  const [employees, setEmployees] = useState<AdminEmployee[]>([])
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(() => {
    adminApi.list().then(setEmployees).catch(() => setError('تعذر تحميل الموظفين'))
  }, [])
  useEffect(load, [load])

  const toggle = async (e: AdminEmployee, authorise: boolean) => {
    setBusy(e.id)
    setError(null)
    try {
      await adminApi.update(e.id, { role: authorise ? allowedRoles[0] : DENY_ROLE })
      load()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'تعذر الحفظ')
    } finally {
      setBusy(null)
    }
  }

  return (
    <section className="tableCard">
      <div className="violationsHead">
        <h2>قائمة المصرّح لهم <small className="whitelistEn">Whitelist</small></h2>
      </div>
      {allowedRoles.length === 0 ? (
        <div className="errorBar">لا توجد منطقة مقيّدة: لا أحد مقيَّد بعد (restricted zone with allowed roles needed)</div>
      ) : (
        <p className="whitelistHint">
          المصرّح لهم: من دورهم <bdi dir="ltr">{allowedRoles.join(' / ')}</bdi>. أي شخص آخر، أو شخص غير معروف، يُطلق تنبيهًا.
        </p>
      )}
      {error && <div className="errorBar">{error}</div>}
      <table className="attTable">
        <thead>
          <tr><th>الموظف</th><th>الدور</th><th>الصلاحية</th><th /></tr>
        </thead>
        <tbody>
          {employees.filter(e => e.active).map(e => {
            const ok = !!e.role && allowedRoles.includes(e.role)
            return (
              <tr key={e.id}>
                <td><span className="whitelistName"><img className="alertPhoto" src={photoUrl(e.id)} alt="" width={32} height={32} />{e.name}</span></td>
                <td dir="ltr">{e.role ?? '—'}</td>
                <td>
                  <span className={ok ? 'chip chip-present' : 'chip chip-absent'}>
                    <span className="chipDot" aria-hidden="true" />{ok ? 'مصرّح' : 'غير مصرّح'}
                  </span>
                </td>
                <td>
                  <button type="button" className="btnGhost" disabled={busy === e.id || allowedRoles.length === 0}
                          onClick={() => toggle(e, !ok)}>
                    {ok ? 'منع' : 'السماح'}
                  </button>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </section>
  )
}

/** #/alerts: what is wrong right now, who is authorised, and the log. */
export function AlertsPage({ snap, day, policy }: { snap: MonitoringSnapshot | null; day: string; policy: AttendancePolicy | null }) {
  const violations = snap?.violations ?? []
  const open = violations.filter(v => v.open)
  const scopes = checkScopes(snap?.zones, policy)
  return (
    <>
      <div className="pageHead">
        <div>
          <h1>تنبيهات الدخول</h1>
          <nav className="crumbs" aria-label="مسار التنقل">
            <span>الرئيسية</span><span className="crumbSep">‹</span>
            <span className="crumbHere">التنبيهات</span>
          </nav>
        </div>
      </div>

      <section className={open.length ? 'alertsNow alertsNowOn' : 'alertsNow'}>
        <h2>{open.length ? `${open.length} تنبيه مستمر الآن` : 'لا توجد تنبيهات مستمرة'}</h2>
        {open.length === 0 && <p>كل من في الكاميرا مصرّح له.</p>}
        <div className="alertCards">
          {open.map(v => (
            <article key={v.id} className={`alertCard alertToast-${v.violation_type}`}>
              <Who v={v} size={56} />
              <div>
                <strong>{v.name ?? 'شخص غير معروف'}</strong>
                <span>{TITLE[v.violation_type].ar}</span>
                <span className="alertToastEn">{TITLE[v.violation_type].en}</span>
                <span className="alertToastMeta"><bdi dir="ltr">{v.camera}</bdi> · منذ {fmtTime(v.started_at)}</span>
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="tableCard scopeCard">
        <h2>أين يُطبَّق الفحص <small className="whitelistEn">Where the check applies</small></h2>
        {scopes.length === 0 ? (
          <p className="whitelistHint">لا فحص على أي كاميرا. ارسم منطقة مقيّدة في صفحة الكاميرات، أو فعّل الفحص على كاميرا.</p>
        ) : (
          <ul className="scopeList">
            {scopes.map(sc => (
              <li key={sc.cam}>
                <bdi dir="ltr" className="scopeCam">{sc.cam}</bdi>
                <span className={sc.wholeStream ? 'chip chip-leave' : 'chip chip-present'}>
                  {sc.wholeStream ? 'البث كاملاً (لا توجد منطقة)' : `داخل: ${sc.zones.join('، ')}`}
                </span>
              </li>
            ))}
          </ul>
        )}
        <p className="whitelistHint">إن لم تُرسم أي منطقة مقيّدة على كاميرا مفعّل عليها الفحص، يُطبَّق على كامل البث. وعند رسم منطقة يقتصر الفحص عليها.</p>
      </section>

      <div className="dashStack">
        <Whitelist allowedRoles={allowedRolesOf(snap?.zones, policy)} />
      </div>
      <div className="dashStack">
        <ViolationsPanel violations={violations} day={day} />
      </div>
    </>
  )
}
