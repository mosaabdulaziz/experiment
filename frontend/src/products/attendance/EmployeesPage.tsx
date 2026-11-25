import { useEffect, useState } from 'react'
import { adminApi, api, attendancePolicyApi, type AdminEmployee, type AttendancePolicy, type PhotoUploadResult } from './api'
import { PhotoPicker } from './PhotoPicker'
import { navigate, productHref } from '../../shared/router'

const REASON_LABEL: Record<string, string> = {
  no_face: 'لم يتم اكتشاف وجه في الصورة',
  multiple_faces: 'الصورة تحتوي أكثر من وجه — يجب أن تحتوي وجهًا واحدًا فقط',
  not_distinct: 'الوجه يتطابق مع موظف آخر مسجل مسبقًا',
  unreadable_photo: 'تعذر قراءة الصورة',
  gallery_write_failed: 'فشل حفظ الصورة في قاعدة بيانات التعرف',
  timeout: 'انتهت مهلة المعالجة',
  subprocess_failed: 'فشلت معالجة الصورة',
}

function resultLine(r: PhotoUploadResult): { text: string; ok: boolean } {
  if (r.ok) {
    return { text: `${r.filename}: تم التسجيل (${r.mode === 'primary' ? 'صورة أساسية' : 'زاوية إضافية'})`, ok: true }
  }
  const base = REASON_LABEL[r.reason ?? ''] ?? 'رُفضت الصورة'
  const extra = r.reason === 'not_distinct' && r.conflicts_with_name
    ? ` — يتشابه مع "${r.conflicts_with_name}" (تطابق ${Math.round((r.similarity ?? 0) * 100)}%)`
    : ''
  return { text: `${r.filename}: ${base}${extra}`, ok: false }
}

/* ── add-employee form ────────────────────────────────────────────────── */

/** prompt.md point 3 -- redone (2026-09-04) to actually match image-4.png:
 * photo circle top-right of the card, name/email/code/role in a two-column
 * grid, work hours full-width, buttons bottom-left. Work hours are NOT a
 * per-employee column -- the /grilling design decided one global schedule
 * for v1 (see sessions.py), and the user confirmed that's staying: these
 * two fields edit that same global AttendancePolicy, pre-filled with its
 * current value.
 *
 * The photo picker is staged locally and only actually uploaded (via the
 * existing two-step API -- create row, then register photos against its
 * new id) inside this same submit -- looks like one form because the admin
 * only presses one button, even though it's still two requests under the
 * hood; the backend genuinely can't register a photo before the employee
 * row (and its id) exists. */
function AddEmployeeForm({ attendancePolicy, onCreated, onPolicySaved, onCancel }: {
  attendancePolicy: AttendancePolicy | null
  onCreated: (e: AdminEmployee) => void
  onPolicySaved: (p: AttendancePolicy) => void
  onCancel: () => void
}) {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [code, setCode] = useState('')
  const [role, setRole] = useState('')
  const [department, setDepartment] = useState('')
  const [workStart, setWorkStart] = useState(attendancePolicy?.work_start ?? '09:00')
  const [workEnd, setWorkEnd] = useState(attendancePolicy?.work_end ?? '17:00')
  const [photos, setPhotos] = useState<File[]>([])
  const [pickerKey, setPickerKey] = useState(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [photoResults, setPhotoResults] = useState<PhotoUploadResult[] | null>(null)
  const [savedName, setSavedName] = useState<string | null>(null)

  useEffect(() => {
    if (attendancePolicy) {
      setWorkStart(attendancePolicy.work_start)
      setWorkEnd(attendancePolicy.work_end)
    }
  }, [attendancePolicy])

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!name.trim()) return
    setBusy(true)
    setError(null)
    setPhotoResults(null)
    setSavedName(null)
    try {
      const created = await adminApi.create({
        name: name.trim(),
        role: role.trim() || undefined,
        department: department.trim() || undefined,
        email: email.trim() || undefined,
        code: code.trim() || undefined,
      })
      // onCreated is deferred until AFTER the photo upload attempt (success
      // or failure) instead of firing right after create(): it's what adds
      // the row -- and its avatar <img> -- to the table, and an <img> that
      // starts loading before the primary photo file exists races a 404 it
      // never recovers from (browsers don't retry a failed <img> load on
      // their own, even once the photo shows up moments later -- only a full
      // page reload does). Wrapped in its own try/catch so a photo-upload
      // failure still surfaces the created employee instead of being
      // swallowed by the outer catch as a misleading "creation failed".
      let results: PhotoUploadResult[] | null = null
      if (photos.length > 0) {
        try {
          const res = await adminApi.uploadPhotos(created.id, photos)
          results = res.results
        } catch (photoErr) {
          results = [{ filename: '—', ok: false, reason: 'subprocess_failed' }]
          void photoErr
        }
      }
      onCreated(created)
      setPhotoResults(results)
      if (attendancePolicy && (workStart !== attendancePolicy.work_start || workEnd !== attendancePolicy.work_end)) {
        onPolicySaved(await attendancePolicyApi.save({ work_start: workStart, work_end: workEnd }))
      }
      setSavedName(created.name)
      setName(''); setEmail(''); setCode(''); setRole(''); setDepartment('')
      setPhotos([]); setPickerKey(k => k + 1)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'فشل إنشاء الموظف')
    } finally {
      setBusy(false)
    }
  }

  const field = (label: string, input: React.ReactNode, wide = false) => (
    <label className={wide ? 'fieldBox fieldBoxWide' : 'fieldBox'}>
      <span>{label}</span>
      {input}
    </label>
  )

  return (
    <form className="addEmployeeCard" onSubmit={submit}>
      <h3 className="cardSubhead">برجاء تسجيل بيانات الموظف</h3>
      <div className="addEmployeeTop">
        <div className="addEmployeePhoto">
          <strong>اضافة صورة الموظف</strong>
          <small>اضغط أي صورة لجعلها الصورة الأساسية</small>
          <PhotoPicker key={pickerKey} onChange={setPhotos} disabled={busy} />
        </div>
        <div className="addEmployeeGrid">
          {field('الاسم *', <input value={name} onChange={e => setName(e.target.value)} required />)}
          {field('البريد الالكتروني', <input type="email" value={email} onChange={e => setEmail(e.target.value)} />)}
          {field('المسمى الوظيفي', <input value={role} onChange={e => setRole(e.target.value)} />)}
          {field('الرقم الوظيفي', <input value={code} onChange={e => setCode(e.target.value)} />)}
          {field('القسم (اختياري)', <input value={department} onChange={e => setDepartment(e.target.value)} />, true)}
          {field('ساعات العمل', (
            <span className="workHoursRow">
              <input type="time" value={workStart} onChange={e => setWorkStart(e.target.value)} />
              <span>إلى</span>
              <input type="time" value={workEnd} onChange={e => setWorkEnd(e.target.value)} />
            </span>
          ), true)}
        </div>
      </div>

      {photoResults && (
        <ul className="photoUploadResults">
          {photoResults.map((r, i) => {
            const { text, ok } = resultLine(r)
            return <li key={i} className={ok ? 'uploadOk' : 'uploadFail'}>{text}</li>
          })}
        </ul>
      )}
      {savedName && !photoResults && <p className="uploadOk">تمت إضافة {savedName} بنجاح</p>}
      {error && <p className="errorBar">{error}</p>}

      <div className="addEmployeeActions">
        <button className="btnPrimary" disabled={busy || !name.trim()}>{busy ? '…' : 'اضافة موظف جديد'}</button>
        <button type="button" className="btnGhost" onClick={onCancel}>الغاء</button>
      </div>
    </form>
  )
}

/* ── one employee's photo-upload widget ───────────────────────────────── */

function PhotoUpload({ employee, onUploaded }: { employee: AdminEmployee; onUploaded: () => void }) {
  const [staged, setStaged] = useState<File[]>([])
  const [pickerKey, setPickerKey] = useState(0)
  const [busy, setBusy] = useState(false)
  const [results, setResults] = useState<PhotoUploadResult[] | null>(null)

  const submit = async () => {
    if (staged.length === 0) return
    setBusy(true)
    setResults(null)
    try {
      const res = await adminApi.uploadPhotos(employee.id, staged)
      setResults(res.results)
      setStaged([]); setPickerKey(k => k + 1)
      // This row's avatar <img> mounted (and likely already failed to load)
      // before any of these photos existed -- a browser never retries a
      // failed <img> on its own, so tell EmployeeRow to cache-bust it now
      // that at least one file in this batch may have registered.
      if (res.results.some(r => r.ok)) onUploaded()
    } catch (e) {
      setResults([{ filename: '—', ok: false, reason: 'subprocess_failed' }])
      void e
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="photoUpload">
      <PhotoPicker key={pickerKey} onChange={setStaged} disabled={busy} />
      <button
        type="button" className="btnGhost btnSmall photoUploadBtn"
        disabled={busy || staged.length === 0} onClick={submit}
      >
        {busy ? '… جارٍ المعالجة (قد يستغرق ثوانٍ)' : staged.length > 0 ? `📷 رفع (${staged.length})` : '📷 رفع صور'}
      </button>
      {results && (
        <ul className="photoUploadResults">
          {results.map((r, i) => {
            const { text, ok } = resultLine(r)
            return <li key={i} className={ok ? 'uploadOk' : 'uploadFail'}>{text}</li>
          })}
        </ul>
      )}
    </div>
  )
}

/* ── row (view or edit mode) ──────────────────────────────────────────── */

function EmployeeRow({ employee, onUpdated, onDeleted }: {
  employee: AdminEmployee
  onUpdated: (e: AdminEmployee) => void
  onDeleted: (id: string) => void
}) {
  const [editing, setEditing] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [form, setForm] = useState({
    name: employee.name, role: employee.role ?? '', department: employee.department ?? '',
    email: employee.email ?? '', code: employee.code ?? '',
  })
  const [busy, setBusy] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  // Cache-busts the avatar <img> after a photo upload succeeds (see
  // PhotoUpload's onUploaded) -- the row, and its avatar, are already
  // mounted by the time an existing employee gets a photo, so nothing else
  // would make the browser re-request a URL whose earlier load it already
  // marked failed.
  const [photoVersion, setPhotoVersion] = useState(0)

  const save = async () => {
    setBusy(true)
    setError(null)
    try {
      const updated = await adminApi.update(employee.id, {
        name: form.name.trim(), role: form.role.trim() || undefined, department: form.department.trim() || undefined,
        email: form.email.trim() || undefined, code: form.code.trim() || undefined,
      })
      onUpdated(updated)
      setEditing(false)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'فشل الحفظ')
    } finally {
      setBusy(false)
    }
  }

  // Leave toggle (moved here from the dashboard table, prompt.md point 4,
  // 2026-09-14). The dashboard reflects it through its websocket snapshot.
  const toggleLeave = async () => {
    setBusy(true)
    setError(null)
    try {
      await api.setLeave(employee.id, !employee.on_leave)
      onUpdated({ ...employee, on_leave: !employee.on_leave })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'فشل تحديث الإجازة')
    } finally { setBusy(false) }
  }

  const toggleActive = async () => {
    setBusy(true)
    try {
      onUpdated(await adminApi.update(employee.id, { active: !employee.active }))
    } catch { /* row keeps its old state; retry available */ } finally { setBusy(false) }
  }

  const remove = async () => {
    if (!window.confirm(`سيتم حذف "${employee.name}" نهائيًا مع كل بياناته (الحضور، الصور، بيانات التعرف). هل أنت متأكد؟`)) {
      return
    }
    setDeleting(true)
    setError(null)
    try {
      await adminApi.remove(employee.id)
      onDeleted(employee.id)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'فشل حذف الموظف')
      setDeleting(false)
    }
  }

  return (
    <>
      <tr className={employee.active ? undefined : 'rowInactive'}>
        <td className="cellName">
          <img
            className="avatar"
            src={photoVersion > 0 ? `${api.employeePhotoUrl(employee.id)}?v=${photoVersion}` : api.employeePhotoUrl(employee.id)}
            alt=""
          />
          {editing ? (
            <input className="inlineEdit" value={form.name} onChange={e => setForm(f => ({ ...f, name: e.target.value }))} />
          ) : (
            <span>{employee.name}</span>
          )}
        </td>
        <td>{editing
          ? <input className="inlineEdit" value={form.code} onChange={e => setForm(f => ({ ...f, code: e.target.value }))} placeholder="—" />
          : (employee.code ?? '—')}
        </td>
        <td>{editing
          ? <input className="inlineEdit" type="email" value={form.email} onChange={e => setForm(f => ({ ...f, email: e.target.value }))} placeholder="—" />
          : (employee.email ?? '—')}
        </td>
        <td>{editing
          ? <input className="inlineEdit" value={form.role} onChange={e => setForm(f => ({ ...f, role: e.target.value }))} placeholder="—" />
          : (employee.role ?? '—')}
        </td>
        <td>{editing
          ? <input className="inlineEdit" value={form.department} onChange={e => setForm(f => ({ ...f, department: e.target.value }))} placeholder="—" />
          : (employee.department ?? '—')}
        </td>
        <td className="cellCode" dir="ltr">{employee.gallery_rec_id}</td>
        <td>
          <span className={employee.active ? 'chip chip-present' : 'chip chip-absent'}>
            <span className="chipDot" aria-hidden="true" />
            {employee.active ? 'نشط' : 'معطّل'}
          </span>
          {employee.on_leave && (
            <span className="chip chip-leave" style={{ marginInlineStart: 4 }}>
              <span className="chipDot" aria-hidden="true" />في إجازة
            </span>
          )}
        </td>
        <td className="employeeActions">
          {editing ? (
            <>
              <button className="btnPrimary btnSmall" onClick={save} disabled={busy}>{busy ? '…' : 'حفظ'}</button>
              <button className="btnGhost btnSmall" onClick={() => setEditing(false)} disabled={busy}>إلغاء</button>
            </>
          ) : (
            <>
              <button className="btnGhost btnSmall" onClick={() => setEditing(true)} disabled={deleting}>تعديل</button>
              <button className="btnGhost btnSmall" onClick={() => setUploading(u => !u)} disabled={deleting}>{uploading ? 'إغلاق الصور' : 'إضافة صور'}</button>
              <button className="btnGhost btnSmall" onClick={toggleLeave} disabled={busy || deleting}>
                {employee.on_leave ? 'إنهاء الإجازة' : 'تسجيل إجازة'}
              </button>
              <button className="btnGhost btnSmall" onClick={toggleActive} disabled={busy || deleting}>{employee.active ? 'تعطيل' : 'تنشيط'}</button>
              <a className="btnGhost btnSmall" href={productHref(`/employee/${employee.id}`)}>السجل الشهري</a>
              <button className="btnDanger btnSmall" onClick={remove} disabled={deleting}>{deleting ? '…' : 'حذف'}</button>
            </>
          )}
        </td>
      </tr>
      {error && (
        <tr><td colSpan={8}><p className="errorBar">{error}</p></td></tr>
      )}
      {uploading && !editing && (
        <tr><td colSpan={8}><PhotoUpload employee={employee} onUploaded={() => setPhotoVersion(v => v + 1)} /></td></tr>
      )}
    </>
  )
}

/* ── add-employee page (#/admin/employees/new) ────────────────────────── */

export function AddEmployeePage({ attendancePolicy, setAttendancePolicy }: {
  attendancePolicy: AttendancePolicy | null
  setAttendancePolicy: (p: AttendancePolicy) => void
}) {
  return (
    <>
      <div className="pageHead">
        <div>
          <h1>إضافة موظف جديد</h1>
          <nav className="crumbs" aria-label="مسار التنقل">
            <span>الرئيسية</span><span className="crumbSep">‹</span>
            <a href={productHref('/admin/employees')}>إدارة الموظفين</a><span className="crumbSep">‹</span>
            <span className="crumbHere">إضافة موظف جديد</span>
          </nav>
        </div>
        <a className="btnGhost noPrint" href={productHref('/admin/employees')}>← العودة إلى القائمة</a>
      </div>
      {/* The form stays on this page after saving so its photo results are
          readable; the list reloads when you go back. */}
      <AddEmployeeForm
        attendancePolicy={attendancePolicy}
        onPolicySaved={setAttendancePolicy}
        onCreated={() => {}}
        onCancel={() => { navigate('/admin/employees') }}
      />
    </>
  )
}

/* ── the page ──────────────────────────────────────────────────────────── */

export function EmployeesPage({ attendancePolicy: _attendancePolicy, setAttendancePolicy: _setAttendancePolicy }: {
  attendancePolicy: AttendancePolicy | null
  setAttendancePolicy: (p: AttendancePolicy) => void
}) {
  const [employees, setEmployees] = useState<AdminEmployee[]>([])
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  const load = () => {
    adminApi.list()
      .then(list => { setEmployees(list); setError(null) })
      .catch(e => setError(e instanceof Error ? e.message : 'فشل تحميل قائمة الموظفين'))
      .finally(() => setLoading(false))
  }
  useEffect(load, [])

  return (
    <>
      <div className="pageHead">
        <div>
          <h1>إدارة الموظفين</h1>
          <nav className="crumbs" aria-label="مسار التنقل">
            <span>الرئيسية</span><span className="crumbSep">‹</span><span className="crumbHere">إدارة الموظفين</span>
          </nav>
        </div>
        {/* Own page now (prompt.md point 8, 2026-09-14), not a form that
            opens above the list. */}
        <a className="btnPrimary noPrint" href={productHref('/admin/employees/new')}>+ إضافة موظف جديد</a>
      </div>

      {error && <div className="errorBar">{error}</div>}

      <section className="tableCard" style={{ marginTop: 16 }}>
        <table className="attTable">
          <thead>
            <tr>
              <th>الاسم</th>
              <th>الرقم الوظيفي</th>
              <th>البريد الالكتروني</th>
              <th>المسمى الوظيفي</th>
              <th>القسم</th>
              <th>معرّف التعرف</th>
              <th>الحالة</th>
              <th>إجراءات</th>
            </tr>
          </thead>
          <tbody>
            {employees.map(e => (
              <EmployeeRow
                key={e.id}
                employee={e}
                onUpdated={updated => setEmployees(es => es.map(x => (x.id === updated.id ? { ...x, ...updated } : x)))}
                onDeleted={id => setEmployees(es => es.filter(x => x.id !== id))}
              />
            ))}
            {!loading && employees.length === 0 && (
              <tr><td colSpan={8} className="tableEmpty">لا يوجد موظفون بعد — أضف أول موظف أعلاه</td></tr>
            )}
          </tbody>
        </table>
      </section>
    </>
  )
}
