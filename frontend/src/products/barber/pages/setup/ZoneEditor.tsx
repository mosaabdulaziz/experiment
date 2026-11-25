/** Setup's zone editor: one camera at a time, drawn over its still. Edits go
 * into the page's per-camera draft (held by SetupPage, so the Areas panel
 * sees them); Save sends one camera's zones. Keys, anywhere in the
 * editor outside a form field: Enter finishes the zone being drawn, Escape cancels it (or
 * clears the selection), Delete / Backspace removes the selected zone. */
import { useRef, useState, type KeyboardEvent } from 'react'
import { api, ApiError, stillUrl } from '../../api/client'
import type { CameraConf, Zone, ZonesByCam, ZoneType } from '../../api/types'
import { useLang } from '../../i18n'
import { cameraLabel } from '../../ui/cameraLabel'
import { centroid } from '../../ui/geometry'
import { Confirm } from '../../ui/ui'
import { ZoneForm } from './ZoneForm'
import { inPolygon, ZoneSurface, type Handle } from './ZoneSurface'
import { linkedScope, revertLinked } from './revertScope'
import { addPoint, deleteZone, isClosable, movePoint, newZone } from './zoneModel'

const TOOLS: ZoneType[] = ['entrance_line', 'waiting', 'chair', 'seat', 'washing', 'row', 'door_area', 'face_area']

export const sameZones = (a: Zone[] | undefined, b: Zone[] | undefined) => JSON.stringify(a ?? []) === JSON.stringify(b ?? [])

export function ZoneEditor({ cameras, draft, saved, onDraft, onSaved }: {
  cameras: CameraConf[]
  draft: ZonesByCam
  saved: ZonesByCam
  onDraft: (next: ZonesByCam) => void
  onSaved: (cam: string, zones: Zone[]) => void
}) {
  const { t, lang } = useLang()
  const [camId, setCamId] = useState(() => cameras[0]?.id ?? '')
  const cam = cameras.find(c => c.id === camId) ?? cameras[0]
  const [drawing, setDrawing] = useState<Zone | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [bust, setBust] = useState(() => Date.now())
  // a pending discard: `to` is the camera to switch to afterwards (null: stay)
  const [confirm, setConfirm] = useState<{ to: string | null } | null>(null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [justSaved, setJustSaved] = useState(false)
  const surfaceRef = useRef<HTMLDivElement | null>(null)

  if (!cam) return <section className="card"><p className="empty">{t('cam.noVideo')}</p></section>
  const zones = draft[cam.id] ?? []
  const dirty = (id: string) => !sameZones(draft[id], saved[id])
  const selected = zones.find(z => z.id === selectedId) ?? null

  const edit = (next: ZonesByCam) => { onDraft(next); setJustSaved(false); setError(null) }
  const putZones = (list: Zone[]) => edit({ ...draft, [cam.id]: list })

  const goTo = (id: string) => {
    setCamId(id)
    setDrawing(null)
    setSelectedId(null)
    setError(null)
    setJustSaved(false)
  }
  const camIds = cameras.map(c => c.id)
  // an owner move edits the pair's other cameras too: those areas are discarded with this camera
  const scope = linkedScope(draft, saved, cam.id, camIds)
  const linked = scope.cameras
  const pickCamera = (id: string) => {
    if (id === cam.id || saving) return
    if (dirty(cam.id)) setConfirm({ to: id })
    else goTo(id)
  }
  const discard = () => {
    onDraft(revertLinked(draft, saved, cam.id, camIds))
    setDrawing(null)
    setSelectedId(null)
    setError(null)
  }
  const sep = lang === 'ar' ? '، ' : ', '
  const askDiscard = () => (linked.length ? setConfirm({ to: null }) : discard())

  const startTool = (type: ZoneType) => {
    setSelectedId(null)
    setDrawing(d => (d?.type === type ? null : newZone(type, cam.id, zones)))
    surfaceRef.current?.focus()
  }
  const onAddPoint = (p: [number, number]) => {
    if (!drawing) return
    const next = addPoint(drawing, p)
    // a door line is done at its second point
    if (next.type === 'entrance_line' && isClosable(next)) finish(next)
    else setDrawing(next)
  }
  const finish = (z: Zone | null = drawing) => {
    if (!z || !isClosable(z)) return
    let done: Zone = { ...z, name: z.name || z.id }
    if (done.type === 'seat') {
      const c = centroid(done.points)
      const chair = zones.find(q => q.type === 'chair' && inPolygon(c[0], c[1], q.points))
      if (chair) done = { ...done, seat_of: chair.id }
    }
    putZones([...zones, done])
    setDrawing(null)
    setSelectedId(done.id)
  }
  const onMove = (id: string, handle: Handle, p: [number, number]) => {
    putZones(zones.map(z => (z.id !== id ? z : handle === 'inside' ? { ...z, inside: p } : movePoint(z, handle, p))))
  }
  const remove = () => {
    if (!selected) return
    putZones(deleteZone(zones, selected.id))
    setSelectedId(null)
  }

  const onKeyDown = (e: KeyboardEvent<HTMLElement>) => {
    // keys typed into the form stay there; Enter on a button presses it
    if (confirm) { if (e.key === 'Escape') setConfirm(null); return }
    if (saving) return
    const tag = (e.target as HTMLElement).tagName
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || e.defaultPrevented) return
    if (e.key === 'Enter' && drawing && tag !== 'BUTTON') { e.preventDefault(); finish() }
    else if (e.key === 'Escape') { e.preventDefault(); if (drawing) setDrawing(null); else setSelectedId(null) }
    else if ((e.key === 'Delete' || e.key === 'Backspace') && selected && !drawing) { e.preventDefault(); remove() }
  }

  async function save() {
    setSaving(true)
    setError(null)
    try {
      const result = await api.saveZones(cam.id, zones)
      onSaved(cam.id, result)
      setJustSaved(true)
    } catch (e) {
      setError(e instanceof ApiError ? e.detail : String(e))
    } finally {
      setSaving(false)
    }
  }

  const label = cameraLabel(cam.id, t)
  return (
    <section className={saving ? 'card zoneEditor zoneSaving' : 'card zoneEditor'} onKeyDown={onKeyDown} aria-busy={saving}>
      <div className="feedHead">
        <div className="toolStrip" role="group" aria-label={t('ze.cameras')}>
          {cameras.map(c => (
            <button key={c.id} type="button" className="toggleChip" aria-pressed={c.id === cam.id} disabled={saving} onClick={() => pickCamera(c.id)}>
              {cameraLabel(c.id, t)}{dirty(c.id) && <span className="dirtyDot" title={t('ze.unsaved')} aria-label={t('ze.unsaved')} />}
            </button>
          ))}
        </div>
        <div className="toolStrip" role="group" aria-label={t('ze.tools')}>
          {TOOLS.map(type => (
            <button key={type} type="button" className={`toolBtn tool-${type}`} aria-pressed={drawing?.type === type}
                    disabled={saving} onClick={() => startTool(type)}>
              <span className="toolSwatch" aria-hidden="true" />{t(`zt.${type}`)}
            </button>
          ))}
        </div>
      </div>

      <div ref={surfaceRef} className="zoneSurface" tabIndex={saving ? -1 : 0} aria-label={t('ze.hint')}>
        <ZoneSurface src={stillUrl(cam.stream, bust)} zones={zones} drawing={drawing} selectedId={selectedId}
                     onAddPoint={onAddPoint} onFinish={() => finish()} onSelect={setSelectedId} onMove={onMove} />
      </div>
      <div className="editorBar">
        <p className="editorHint">{t('ze.hint')}{drawing ? ` ${t('ze.drawingHint')}` : ''}</p>
        <button type="button" className="btnPlain btnSmall" disabled={saving} onClick={() => setBust(Date.now())}>{t('ze.refresh')}</button>
      </div>

      <div className="zoneEditorBottom">
        <div className="zoneEditorList">
          {zones.length === 0
            ? <p className="muted small">{t('ze.noZones')}</p>
            : (
              <ul className="zoneChips">
                {zones.map(z => (
                  <li key={z.id}>
                    <button type="button" className={`zoneChip tool-${z.type}`} aria-pressed={z.id === selectedId} disabled={saving}
                            onClick={() => { setDrawing(null); setSelectedId(z.id) }}>
                      <span className="toolSwatch" aria-hidden="true" /><bdi dir="auto">{z.name || z.id}</bdi>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          {selected && (
            <fieldset className="zoneFormSet" disabled={saving}>
              <ZoneForm cam={cam.id} draft={draft} zone={selected} onChange={edit} onDelete={remove} />
            </fieldset>
          )}
        </div>
        <div className="zoneSave">
          <button type="button" className="btnPrimary" disabled={!dirty(cam.id) || saving || !!drawing} onClick={save}>
            {t('ze.save', { cam: label })}
          </button>
          <button type="button" className="btnPlain" disabled={!dirty(cam.id) || saving} onClick={askDiscard}>{t('ze.discard')}</button>
          {error && <p className="fieldError" role="alert" dir="ltr">{error}</p>}
          <p className="saveNote" role="status">{justSaved && !dirty(cam.id) ? t('ze.saved') : ''}</p>
        </div>
      </div>

      {confirm && (
        <Confirm title={t('ze.unsaved')} confirmLabel={t('ze.discard')} danger
                 onCancel={() => setConfirm(null)}
                 onConfirm={() => { discard(); if (confirm.to) goTo(confirm.to); setConfirm(null) }}>
          {confirm.to && <p>{t('ze.switchUnsaved')}</p>}
          {linked.length > 0 && (
            <p>{t('ze.alsoReverts', { areas: scope.pairs.join(sep), cams: linked.map(c => cameraLabel(c, t)).join(sep) })}</p>
          )}
        </Confirm>
      )}
    </section>
  )
}
