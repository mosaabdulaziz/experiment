import { useCallback, useEffect, useRef, useState } from 'react'
import {
  api, attendancePolicyApi, monitoringApi,
  type AttendancePolicy, type CameraInfo, type CameraType, type DeskState,
  type MonitoringRecord, type MonitoringSnapshot,
  type Policy, type RecordingInfo, type SceneEntry, type Zone, type ZoneType,
} from './api'
import { useAttendanceWebSocket } from './useAttendanceWebSocket'
import { ATTENDANCE_CAM_IDS, CAM_LABELS, LiveFrame, useWallClock } from './components'
import { productHref } from '../../shared/router'

/* ── helpers ───────────────────────────────────────────────────────── */

export function fmtDuration(seconds: number): string {
  const s = Math.max(0, Math.round(seconds))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const r = s % 60
  return `${h}:${String(m).padStart(2, '0')}:${String(r).padStart(2, '0')}`
}

const STATE_LABEL: Record<DeskState, string> = { at_desk: 'على المكتب', away: 'بعيد', absent: 'غائب' }
const ZONE_FILL = 'rgba(20, 160, 94, 0.18)'
const ZONE_STROKE = 'rgba(20, 160, 94, 0.95)'
// Restricted zones (2026-09-02): red, not green -- a zone type absent
// entirely means "seat" (zones saved before this field existed, see
// zones.py's seat_zones()/restricted_zones()).
const RESTRICTED_FILL = 'rgba(214, 69, 80, 0.2)'
const RESTRICTED_STROKE = 'rgba(214, 69, 80, 0.95)'

function fmtClock(d: Date): string {
  return d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

/* ── overlay drawing (shared by the live view and the editor) ───────── */

function drawZones(ctx: CanvasRenderingContext2D, zones: Zone[], w: number, h: number,
                   selectedId: string | null, showVertices: boolean) {
  for (const z of zones) {
    if (z.points.length === 0) continue
    const selected = z.id === selectedId
    const restricted = z.type === 'restricted'
    const stroke = restricted ? RESTRICTED_STROKE : ZONE_STROKE
    ctx.beginPath()
    z.points.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x * w, y * h) : ctx.lineTo(x * w, y * h)))
    if (z.points.length >= 3) {
      ctx.closePath()
      ctx.fillStyle = selected
        ? (restricted ? 'rgba(214, 69, 80, 0.34)' : 'rgba(20, 160, 94, 0.32)')
        : (restricted ? RESTRICTED_FILL : ZONE_FILL)
      ctx.fill()
    }
    ctx.lineWidth = selected ? 3 : 2
    ctx.strokeStyle = selected ? (restricted ? '#ffb3b8' : '#7cf2b4') : stroke
    ctx.setLineDash(z.points.length < 3 ? [6, 4] : [])
    ctx.stroke()
    ctx.setLineDash([])
    // label at the polygon's top-most vertex
    const top = z.points.reduce((a, b) => (b[1] < a[1] ? b : a))
    ctx.font = '700 12px Cairo, sans-serif'
    ctx.textAlign = 'center'
    const label = restricted ? `⛔ ${z.name}` : z.name
    const tw = ctx.measureText(label).width + 12
    ctx.fillStyle = restricted ? 'rgba(69, 15, 18, 0.85)' : 'rgba(12, 53, 39, 0.85)'
    ctx.fillRect(top[0] * w - tw / 2, top[1] * h - 18, tw, 16)
    ctx.fillStyle = restricted ? '#ffd9db' : '#d7f3e3'
    ctx.fillText(label, top[0] * w, top[1] * h - 6)
    if (showVertices) {
      for (const [x, y] of z.points) {
        ctx.beginPath()
        ctx.arc(x * w, y * h, selected ? 6 : 4, 0, Math.PI * 2)
        ctx.fillStyle = selected ? '#fff' : (restricted ? '#f6c6c9' : '#bfe9d2')
        ctx.fill()
        ctx.strokeStyle = stroke
        ctx.lineWidth = 1.5
        ctx.stroke()
      }
    }
  }
}

function drawScene(ctx: CanvasRenderingContext2D, scene: SceneEntry[], w: number, h: number,
                   restrictedIds: Set<string>) {
  ctx.font = '700 12px Cairo, sans-serif'
  ctx.textAlign = 'right'
  for (const e of scene) {
    if (!e.box) continue
    const [l, t, bw, bh] = e.box
    const inRestricted = e.zone_id !== null && restrictedIds.has(e.zone_id)
    const inZone = e.zone_id !== null && !inRestricted
    // A seated body is labelled by the seat's occupant (survives occlusion,
    // back-turned sitting and track-id churn); otherwise by its own binding.
    const seatedAs = e.credited_name
    const who = seatedAs ?? e.name ?? `#${e.track_id}`
    const known = !!(seatedAs || e.employee_id)
    const status = inRestricted ? (known ? 'دخول غير مصرّح' : 'شخص غير معروف — منطقة مقيدة')
      : seatedAs ? 'على المكتب' : !known ? 'غير معروف'
      : inZone && !e.moving && !e.seated ? 'واقف عند المكتب'
      : inZone && !e.moving ? 'بجوار مكتب' : inZone ? 'يمر' : 'بعيد'
    const label = `${who} · ${status}`
    const tw = ctx.measureText(label).width + 14
    const x = Math.min(w - tw, Math.max(0, (l + bw) * w - tw))
    const y = Math.max(0, t * h - 22)
    if (inRestricted) {
      ctx.strokeStyle = 'rgba(214, 69, 80, 0.95)'
      ctx.lineWidth = 3
      ctx.strokeRect(l * w, t * h, bw * w, bh * h)
    }
    ctx.fillStyle = inRestricted ? 'rgba(153, 27, 36, 0.94)'
      : !known ? 'rgba(60, 60, 60, 0.85)' : seatedAs ? 'rgba(20, 122, 67, 0.92)' : 'rgba(185, 124, 16, 0.92)'
    ctx.fillRect(x, y, tw, 18)
    ctx.fillStyle = '#fff'
    ctx.fillText(label, x + tw - 7, y + 13)
  }
}

/** Canvas kept the same pixel size as the displayed frame (the frame keeps
 * its native 3:2 aspect, so normalized coords map 1:1). */
function useOverlayCanvas(frameRef: React.RefObject<HTMLElement | null>) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [size, setSize] = useState({ w: 0, h: 0 })
  useEffect(() => {
    const el = frameRef.current
    if (!el) return
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }))
    ro.observe(el)
    setSize({ w: el.clientWidth, h: el.clientHeight })
    return () => ro.disconnect()
  }, [frameRef])
  return { canvasRef, size }
}

/* ── live camera view with overlays ─────────────────────────────────── */

export function CameraView({ camId, zones, scene, recording, large, onClick }: {
  camId: string
  zones: Zone[]
  scene: SceneEntry[]
  recording: RecordingInfo | undefined
  large?: boolean
  onClick?: () => void
}) {
  const wrapRef = useRef<HTMLDivElement | null>(null)
  const { canvasRef, size } = useOverlayCanvas(wrapRef)
  const wallClock = useWallClock(recording)
  useEffect(() => {
    const c = canvasRef.current
    if (!c || size.w === 0) return
    c.width = size.w
    c.height = size.h
    const ctx = c.getContext('2d')!
    ctx.clearRect(0, 0, size.w, size.h)
    drawZones(ctx, zones, size.w, size.h, null, false)
    const restrictedIds = new Set(zones.filter(z => z.type === 'restricted').map(z => z.id))
    drawScene(ctx, scene, size.w, size.h, restrictedIds)
  }, [zones, scene, size, canvasRef])
  return (
    <div
      ref={wrapRef}
      className={large ? 'monView monViewLarge' : 'monView'}
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
    >
      <LiveFrame camId={camId} className="monFrame" />
      <canvas ref={canvasRef} className="monCanvas" />
      <span className="camLive">بث مباشر</span>
      <span className="camName">{CAM_LABELS[camId] ?? camId}</span>
      {wallClock && <span className="camClock" title={recording?.clock_source === 'camera' ? 'ساعة الكاميرا (RTCP)' : 'ساعة الخادم'}>{fmtClock(wallClock)}</span>}
    </div>
  )
}

/* ── zone editor (mirrors the surveillance app's editor interaction) ── */

const VERTEX_GRAB_PX = 12

export function ZoneEditor({ camId, initial, onClose, onSaved, inline, cameraType, onCameraTypeSaved }: {
  camId: string
  initial: Zone[]
  onClose?: () => void
  onSaved: (zones: Zone[]) => void
  /** Cameras page (2026-09-04): embedded directly in the page flow instead of
   * a modal overlay -- no backdrop, no close button, Esc does nothing. */
  inline?: boolean
  /** When given, also renders an office/break-room selector for this camera
   * and folds saving it into the same "حفظ" click -- one place to configure
   * a camera (zones + restriction + place type), per prompt.md point 4,
   * instead of a duplicate list on the Settings page. */
  cameraType?: CameraType
  onCameraTypeSaved?: (type: CameraType) => Promise<void>
}) {
  const [zones, setZones] = useState<Zone[]>(() => initial.map(z => ({ ...z, points: z.points.map(p => [...p]) })))
  const [selectedId, setSelectedId] = useState<string | null>(initial[0]?.id ?? null)
  const [campType, setCampType] = useState<CameraType>(cameraType ?? 'office')
  const [dragging, setDragging] = useState<number | null>(null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const wrapRef = useRef<HTMLDivElement | null>(null)
  const { canvasRef, size } = useOverlayCanvas(wrapRef)
  const selected = zones.find(z => z.id === selectedId) ?? null

  useEffect(() => {
    if (!onClose) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  useEffect(() => {
    const c = canvasRef.current
    if (!c || size.w === 0) return
    c.width = size.w
    c.height = size.h
    const ctx = c.getContext('2d')!
    ctx.clearRect(0, 0, size.w, size.h)
    drawZones(ctx, zones, size.w, size.h, selectedId, true)
  }, [zones, selectedId, size, canvasRef])

  const updateSelected = (patch: Partial<Zone>) =>
    setZones(zs => zs.map(z => (z.id === selectedId ? { ...z, ...patch } : z)))

  const toCanvas = (e: React.PointerEvent | React.MouseEvent) => {
    const rect = canvasRef.current!.getBoundingClientRect()
    return { px: e.clientX - rect.left, py: e.clientY - rect.top }
  }

  const vertexAt = (px: number, py: number): number => {
    if (!selected) return -1
    for (let i = 0; i < selected.points.length; i++) {
      const [x, y] = selected.points[i]
      if (Math.hypot(x * size.w - px, y * size.h - py) <= VERTEX_GRAB_PX) return i
    }
    return -1
  }

  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const { px, py } = toCanvas(e)
    const idx = vertexAt(px, py)
    if (idx >= 0) {
      setDragging(idx)
      canvasRef.current!.setPointerCapture(e.pointerId)
    }
  }
  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (dragging === null || !selected) return
    const { px, py } = toCanvas(e)
    const pts = selected.points.map((p, i) =>
      i === dragging ? [Math.min(1, Math.max(0, px / size.w)), Math.min(1, Math.max(0, py / size.h))] : p)
    updateSelected({ points: pts })
  }
  const handlePointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (dragging !== null) {
      canvasRef.current!.releasePointerCapture(e.pointerId)
      setDragging(null)
    }
  }
  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!selected || dragging !== null) return
    const { px, py } = toCanvas(e)
    if (vertexAt(px, py) >= 0) return // clicking a vertex (incl. the first) never adds a point
    updateSelected({ points: [...selected.points, [px / size.w, py / size.h]] })
  }

  const addZone = () => {
    const id = `zone-${Date.now().toString(36)}`
    setZones(zs => [...zs, { id, name: `مكتب ${zs.length + 1}`, points: [], type: 'seat' as ZoneType }])
    setSelectedId(id)
  }
  const deleteZone = (id: string) => {
    setZones(zs => zs.filter(z => z.id !== id))
    if (selectedId === id) setSelectedId(null)
  }
  const save = async () => {
    setSaving(true)
    setError(null)
    try {
      const saved = await monitoringApi.saveZones(camId, zones.filter(z => z.points.length >= 3))
      onSaved(saved)
      if (onCameraTypeSaved && campType !== cameraType) {
        await onCameraTypeSaved(campType)
      }
      onClose?.()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'فشل الحفظ')
    } finally {
      setSaving(false)
    }
  }

  const body = (
    <div className={inline ? 'camModal zoneEditor zoneEditorInline' : 'camModal zoneEditor'} onClick={e => e.stopPropagation()}>
        <div className="camModalHead">
          <div className="camModalTitle">
            <strong>تعديل مناطق المكاتب — {CAM_LABELS[camId] ?? camId}</strong>
            <span className="zoneHint">اضغط على الصورة لإضافة نقطة · اسحب النقاط لتعديلها{onClose ? ' · Esc للإلغاء' : ''}</span>
          </div>
          {onClose && <button className="camModalClose" onClick={onClose} aria-label="إغلاق">✕</button>}
        </div>
        <div className="zoneEditorBody">
          <div ref={wrapRef} className="monView monViewLarge zoneCanvasWrap">
            <LiveFrame camId={camId} className="monFrame" />
            <canvas
              ref={canvasRef}
              className="monCanvas zoneCanvas"
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onClick={handleClick}
            />
          </div>
          <aside className="zoneRail">
            {cameraType !== undefined && (
              <label className="zoneTypeToggle placeTypeToggle">
                <span>نوع المكان</span>
                <select value={campType} onChange={e => setCampType(e.target.value as CameraType)}>
                  <option value="office">مكتب</option>
                  <option value="break_room">غرفة استراحة</option>
                </select>
                <small>التواجد في غرفة استراحة لا يُنهي جلسة الحضور بغض النظر عن مدته</small>
              </label>
            )}
            <div className="zoneRailHead">
              <strong>المناطق</strong>
              <button className="btnGhost btnSmall" onClick={addZone}>+ منطقة</button>
            </div>
            {zones.length === 0 && <p className="camEmpty">لا توجد مناطق بعد — أضف واحدة ثم ارسمها</p>}
            {zones.map(z => (
              <div
                key={z.id}
                className={z.id === selectedId ? 'zoneItem zoneItemActive' : 'zoneItem'}
                onClick={() => setSelectedId(z.id)}
              >
                <input
                  value={z.name}
                  onChange={e => setZones(zs => zs.map(o => (o.id === z.id ? { ...o, name: e.target.value } : o)))}
                  onClick={e => e.stopPropagation()}
                />
                <span className="zonePts">{z.points.length} نقاط</span>
                <button className="zoneDel" title="حذف" onClick={e => { e.stopPropagation(); deleteZone(z.id) }}>🗑</button>
              </div>
            ))}

            {selected && (
              <div className="zoneTypeEditor">
                <label className="zoneTypeToggle">
                  <span>نوع المنطقة</span>
                  <select
                    value={selected.type ?? 'seat'}
                    onChange={e => updateSelected({ type: e.target.value as ZoneType })}
                  >
                    <option value="seat">مكتب (لمراقبة الإنتاجية)</option>
                    <option value="restricted">منطقة مقيدة (وصول مصرّح)</option>
                  </select>
                </label>
                {selected.type === 'restricted' && (
                  <label className="zoneTypeToggle">
                    <span>الأدوار المسموح لها بالدخول</span>
                    <input
                      placeholder="مثال: it, admin (مفرّقة بفواصل)"
                      defaultValue={(selected.allowed_roles ?? []).join(', ')}
                      onBlur={e => updateSelected({
                        allowed_roles: e.target.value.split(',').map(s => s.trim()).filter(Boolean),
                      })}
                    />
                    <small>قائمة فارغة = لا يُسمح لأي دور بالدخول (الوضع الافتراضي الأكثر أمانًا)</small>
                  </label>
                )}
              </div>
            )}

            {error && <p className="errorBar">{error}</p>}
            <div className="zoneActions">
              <button className="btnPrimary" onClick={save} disabled={saving}>{saving ? '…' : 'حفظ'}</button>
              {onClose && <button className="btnGhost" onClick={onClose}>إلغاء</button>}
            </div>
          </aside>
        </div>
    </div>
  )

  return inline ? body : (
    <div className="camModalBackdrop" onClick={onClose}>{body}</div>
  )
}

/* ── expanded live view with the edit entry point ───────────────────── */

function ExpandedView({ camId, zones, scene, recording, onClose }: {
  camId: string
  zones: Zone[]
  scene: SceneEntry[]
  recording: RecordingInfo | undefined
  onClose: () => void
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  return (
    <div className="camModalBackdrop" onClick={onClose}>
      <div className="camModal" onClick={e => e.stopPropagation()}>
        <div className="camModalHead">
          <div className="camModalTitle">
            <span className="camLiveInline">بث مباشر</span>
            <strong>{CAM_LABELS[camId] ?? camId}</strong>
            <span className="zoneHint">{zones.length} مناطق</span>
          </div>
          <button className="camModalClose" onClick={onClose} aria-label="إغلاق">✕</button>
        </div>
        <CameraView camId={camId} zones={zones} scene={scene} recording={recording} large />
      </div>
    </div>
  )
}

/* ── settings panel (policy knobs) ──────────────────────────────────── */

export function SettingsPanel({ policy, onSaved }: { policy: Policy; onSaved: (p: Policy) => void }) {
  const [form, setForm] = useState({
    debounce_seconds: policy.debounce_seconds,
    away_grace_seconds: policy.away_grace_seconds,
    phone_sustained_seconds: policy.phone_sustained_seconds,
    seated_max_aspect: policy.seated_max_aspect,
  })
  const [dirty, setDirty] = useState(false)
  const [saving, setSaving] = useState(false)
  useEffect(() => {
    if (!dirty) {
      setForm({
        debounce_seconds: policy.debounce_seconds,
        away_grace_seconds: policy.away_grace_seconds,
        phone_sustained_seconds: policy.phone_sustained_seconds,
        seated_max_aspect: policy.seated_max_aspect,
      })
    }
  }, [policy, dirty])
  const field = (key: keyof typeof form, label: string) => (
    <label>
      <span>{label}</span>
      <input
        type="number" min={0} step={key === 'seated_max_aspect' ? 0.1 : 1}
        value={form[key]}
        onChange={e => { setForm(f => ({ ...f, [key]: Number(e.target.value) })); setDirty(true) }}
      />
    </label>
  )
  const save = async () => {
    setSaving(true)
    try {
      onSaved(await monitoringApi.savePolicy(form))
      setDirty(false)
    } catch { /* next snapshot corrects the form */ } finally { setSaving(false) }
  }
  return (
    <section className="settingsRowCard">
      <div className="settingsRowInfo">
        <h3>إعدادات مراقبة الإنتاجية</h3>
        <p>
          استخدام الهاتف، تثبيت الحالة، سماحية الابتعاد، وحد شكل الجلوس. وقت العمل = على المكتب −
          استخدام الهاتف المستمر · خارج العمل = فترة الحضور − وقت العمل.
        </p>
      </div>
      <div className="settingsRowControl">
        <div className="settingsRowFields">
          {field('phone_sustained_seconds', 'الحد الأدنى لاستخدام الهاتف (ثانية)')}
          {field('debounce_seconds', 'تثبيت الحالة (ثانية)')}
          {field('away_grace_seconds', 'سماحية الابتعاد (ثانية)')}
          {field('seated_max_aspect', 'حد شكل الجلوس (طول ÷ عرض)')}
        </div>
        <button className="btnPrimary" onClick={save} disabled={!dirty || saving}>{saving ? '…' : 'حفظ'}</button>
      </div>
    </section>
  )
}

/* ── settings panel (attendance v2: schedule/gaps/camera types) --
 * a SEPARATE backend store from the one above (see sessions.py's module
 * docstring for why); "one settings place" was a UI decision, not a
 * mandate to merge the two ─────────────────────────────────────────── */

/** Split into row-cards matching prompt.md image-6's "اعدادات مواعيد العمل"
 * and "اعدادات التنبيهات" -- both those cards, plus a third for the fields
 * the mockup doesn't show (break/exit-gap/violation-debounce, still real
 * settings this app needs), share one form/dirty/save: all three map onto
 * this same AttendancePolicy store, so -- unlike the theme toggle or the
 * productivity panel above, which are genuinely separate concerns -- one
 * shared "حفظ" is the honest one, not three that'd silently race each
 * other. */
export function AttendanceSettingsPanel({ policy, onSaved }: { policy: AttendancePolicy; onSaved: (p: AttendancePolicy) => void }) {
  const [form, setForm] = useState(policy)
  const [dirty, setDirty] = useState(false)
  const [saving, setSaving] = useState(false)
  useEffect(() => { if (!dirty) setForm(policy) }, [policy, dirty])

  const numberField = (key: keyof AttendancePolicy, label: string, step = 1) => (
    <label>
      <span>{label}</span>
      <input
        type="number" min={0} step={step}
        value={form[key] as number}
        onChange={e => { setForm(f => ({ ...f, [key]: Number(e.target.value) })); setDirty(true) }}
      />
    </label>
  )
  const timeField = (key: 'work_start' | 'work_end' | 'departure_reveal_time', label: string) => (
    <label>
      <span>{label}</span>
      <input
        type="time" value={form[key] ?? '19:00'}
        onChange={e => { setForm(f => ({ ...f, [key]: e.target.value })); setDirty(true) }}
      />
    </label>
  )

  const save = async () => {
    setSaving(true)
    try {
      onSaved(await attendancePolicyApi.save(form))
      setDirty(false)
    } catch { /* next load corrects the form */ } finally { setSaving(false) }
  }

  return (
    <div className="settingsGroup">
      <section className="settingsRowCard">
        <div className="settingsRowInfo">
          <h3>اعدادات مواعيد العمل</h3>
          <p>تحديث مواعيد العمل. سيؤثر هذا الخيار على الحضور والتأخير لكل الموظفين.</p>
        </div>
        <div className="settingsRowControl">
          <div className="settingsRowFields">
            {timeField('work_end', 'الى')}
            {timeField('work_start', 'من')}
          </div>
        </div>
      </section>

      <section className="settingsRowCard">
        <div className="settingsRowInfo">
          <h3>وقت إظهار الانصراف</h3>
          <p>حتى هذا الوقت يظهر آخر ظهور للموظف فقط، وبعده يُعتمد آخر ظهور كوقت الانصراف في جدول اليوم.</p>
        </div>
        <div className="settingsRowControl">
          <div className="settingsRowFields">
            {timeField('departure_reveal_time', 'الوقت')}
          </div>
        </div>
      </section>

      <section className="settingsRowCard">
        <div className="settingsRowInfo">
          <h3>اعدادات التنبيهات</h3>
          <p>حدود التأخير والمغادرة المبكرة التي يُبنى عليها تصنيف كل يوم حضور.</p>
        </div>
        <div className="settingsRowControl">
          <div className="settingsRowFields">
            {numberField('early_leave_grace_minutes', 'حد المغادرة المبكرة ( بالدقيقة ) *')}
            {numberField('late_grace_minutes', 'حد التنبيه للتأخير ( بالدقيقة ) *')}
          </div>
        </div>
      </section>

      <section className="settingsRowCard">
        <div className="settingsRowInfo">
          <h3>اعدادات اضافية للحضور</h3>
          <p>ضبط دقيق لخوارزمية الجلسات: الحد الأدنى للاستراحة، فجوة الانصراف، وتثبيت مخالفات المناطق المقيدة.
             صافي الساعات = الحضور − الاستراحات.</p>
        </div>
        <div className="settingsRowControl">
          <div className="settingsRowFields">
            {numberField('break_min_seconds', 'الحد الأدنى للاستراحة (ثانية)')}
            {numberField('exit_gap_seconds', 'فجوة الانصراف (ثانية)')}
            {numberField('violation_debounce_seconds', 'تثبيت مخالفة المناطق (ثانية)')}
          </div>
        </div>
      </section>

      <p className="settingsNote">
        نوع كل كاميرا (مكتب / غرفة استراحة) يُضبط الآن من <a href={productHref('/cameras')}>صفحة الكاميرات</a> —
        اختر الكاميرا لتعديله مع مناطقها معًا في مكان واحد.
      </p>

      <div className="settingsSaveBar">
        <button className="btnPrimary" onClick={save} disabled={!dirty || saving}>{saving ? '…' : 'حفظ'}</button>
      </div>
    </div>
  )
}

/* ── the page ───────────────────────────────────────────────────────── */

export function MonitoringPage({ live }: { live: boolean }) {
  const [snap, setSnap] = useState<MonitoringSnapshot | null>(null)
  const [cameras, setCameras] = useState<CameraInfo[]>([])
  const [recordings, setRecordings] = useState<RecordingInfo[]>([])
  const [expanded, setExpanded] = useState<string | null>(null)

  useEffect(() => {
    monitoringApi.snapshot().then(setSnap).catch(() => {})
    let cancelled = false
    const tick = () => {
      api.listCameras().then(c => { if (!cancelled) setCameras(c) }).catch(() => {})
      api.listRecordings().then(r => { if (!cancelled) setRecordings(r) }).catch(() => {})
    }
    tick()
    const id = setInterval(tick, 2000)
    return () => { cancelled = true; clearInterval(id) }
  }, [])

  const onMonitoring = useCallback((msg: MonitoringSnapshot) => setSnap(msg), [])
  useAttendanceWebSocket(() => {}, onMonitoring)

  const recordingByCam = Object.fromEntries(recordings.map(r => [r.cam_id, r]))
  const liveCams = cameras.filter(c => ATTENDANCE_CAM_IDS.includes(c.id) && recordingByCam[c.id]?.stream_started_at)
  const zonesFor = (cam: string) => snap?.zones[cam] ?? []
  const sceneFor = (cam: string) => snap?.scene[cam] ?? []
  const records: MonitoringRecord[] = snap?.records ?? []
  const atDeskNow = records.filter(r => r.state === 'at_desk').length
  const awayNow = records.filter(r => r.state === 'away').length
  const totalWorking = records.reduce((a, r) => a + r.working_seconds, 0)
  const totalNotWorking = records.reduce((a, r) => a + r.not_working_seconds, 0)

  return (
    <>
      <div className="pageHead">
        <div>
          <h1>مراقبة الإنتاجية</h1>
          <nav className="crumbs" aria-label="مسار التنقل">
            <span>الرئيسية</span><span className="crumbSep">‹</span><span className="crumbHere">مراقبة الإنتاجية</span>
          </nav>
        </div>
        <span className={live ? 'liveBadge liveOn' : 'liveBadge'}>
          <span className="liveDot" aria-hidden="true" />
          {live ? 'مباشر' : 'جارٍ الاتصال'}
        </span>
      </div>

      <div className="kpiRow">
        <div className="kpiCard"><div className="kpiHead"><span className="kpiLabel">على المكتب الآن</span><span className="kpiIcon" /></div><div className="kpiValue">{atDeskNow}</div><span className="kpiTag">داخل مناطق العمل</span></div>
        <div className="kpiCard kpiLeave"><div className="kpiHead"><span className="kpiLabel">بعيد الآن</span><span className="kpiIcon" /></div><div className="kpiValue">{awayNow}</div><span className="kpiTag">حاضر خارج المكتب</span></div>
        <div className="kpiCard"><div className="kpiHead"><span className="kpiLabel">إجمالي وقت العمل</span><span className="kpiIcon" /></div><div className="kpiValue kpiTime">{fmtDuration(totalWorking)}</div><span className="kpiTag">كل الموظفين</span></div>
        <div className="kpiCard kpiAbsent"><div className="kpiHead"><span className="kpiLabel">إجمالي خارج العمل</span><span className="kpiIcon" /></div><div className="kpiValue kpiTime">{fmtDuration(totalNotWorking)}</div><span className="kpiTag">كل الموظفين</span></div>
      </div>

      <section className="monCameras">
        {liveCams.length === 0 ? (
          <p className="camEmpty">لا توجد كاميرات تبث حاليًا…</p>
        ) : (
          <div className="monGrid">
            {liveCams.map(cam => (
              <CameraView
                key={cam.id}
                camId={cam.id}
                zones={zonesFor(cam.id)}
                scene={sceneFor(cam.id)}
                recording={recordingByCam[cam.id]}
                onClick={() => setExpanded(cam.id)}
              />
            ))}
          </div>
        )}
      </section>

      <section className="tableCard">
        <table className="attTable">
          <thead>
            <tr>
              <th>الموظف</th>
              <th>الحالة</th>
              <th>على المكتب</th>
              <th>الهاتف</th>
              <th>بعيد</th>
              <th>وقت العمل</th>
              <th>خارج العمل</th>
            </tr>
          </thead>
          <tbody>
            {records.map(r => (
              <tr key={r.employee_id}>
                <td className="cellName">
                  <img className="avatar" src={api.employeePhotoUrl(r.employee_id)} alt="" />
                  <span>{r.display_name_ar ?? r.name}</span>
                </td>
                <td><span className={`chip chip-desk-${r.state}`}><span className="chipDot" />{STATE_LABEL[r.state]}</span></td>
                <td className="cellTime">{fmtDuration(r.at_desk_seconds)}</td>
                <td className="cellTime">{fmtDuration(r.phone_seconds)}</td>
                <td className="cellTime">{fmtDuration(r.away_seconds)}</td>
                <td className="cellTime cellWork">{fmtDuration(r.working_seconds)}</td>
                <td className="cellTime cellNotWork">{fmtDuration(r.not_working_seconds)}</td>
              </tr>
            ))}
            {records.length === 0 && <tr><td colSpan={7} className="tableEmpty">لا توجد بيانات بعد</td></tr>}
          </tbody>
        </table>
      </section>

      {/* Settings + the zone editor now live on the attendance page (App.tsx's
         DashboardPage) -- moved 2026-09-03 at the user's request. This page
         stays a read-only live view: video + overlays + the desk-state table. */}
      {expanded && (
        <ExpandedView
          camId={expanded}
          zones={zonesFor(expanded)}
          scene={sceneFor(expanded)}
          recording={recordingByCam[expanded]}
          onClose={() => setExpanded(null)}
        />
      )}
    </>
  )
}
