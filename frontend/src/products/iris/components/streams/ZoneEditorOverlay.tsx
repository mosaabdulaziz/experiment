import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { useStreamsStore } from '@/stores/streamsStore'
import { useElementSize } from '@/hooks/useElementSize'
import { zonesApi, type ZoneInput } from '@/api/zones'
import { IconClose } from '@/components/ui/icons'
import type { ProximityInfo, ZonePoint } from '@/types/zone'
import { ZoneRail } from './ZoneRail'
import { ZoneConfigPanel } from './ZoneConfigPanel'
import { useEditorStrings } from './editorStrings'
import {
  makeDraftZone, draftFromZone, zoneColor, zoneInvalidReason, type DraftZone,
} from './zoneDraft'
import styles from './ZoneEditorOverlay.module.css'

// Hit radius (canvas pixels) for both closing the ring (click on/near the first point) and
// grabbing a vertex to drag — generous enough that neither requires a pixel-perfect click.
const VERTEX_HIT_RADIUS = 10

// How long to wait for the sidecar to ack a zone reload before showing the non-blocking
// warning (sidecar polls at 1s, frontend polls /api/cameras at 2s, so ~3-4s worst case).
const APPLY_TIMEOUT_MS = 12000

// Post-save lifecycle. 'applying' waits for the sidecar to ack the new zones version;
// 'apply_timeout' is a non-blocking warning (the zones are saved and will apply when the
// sidecar comes back — its startup ack catches up automatically).
type SavePhase = 'idle' | 'saving' | 'saved' | 'applying' | 'active' | 'apply_timeout'

interface OverlayProps {
  /** The features this camera runs (the safety product): only their sections show (see ZoneConfigPanel). */
  features?: string[]
  /** Shown in the header instead of the stream id (the safety product shows the camera's name). */
  title?: string
  /** Rendered under the toolbar with the zones as last saved and as drafted (the safety product's coverage summary
   *  and warnings). */
  notice?: (saved: DraftZone[], draft: DraftZone[]) => ReactNode
  /** Rendered under the picture, above the selected zone's settings (the safety product's camera settings: its
   *  name and features, kept with its zones). */
  settings?: ReactNode
  /** The picture to draw on when `zoneEditId` is no running stream (the safety product's camera with no feature on:
   *  a still from go2rtc), with the status shown beside the title. Saves still go to `zoneEditId`. */
  source?: { src: string; live: boolean; label: string }
}

export function ZoneEditorOverlay({ features, title, notice, settings, source }: OverlayProps = {}) {
  const s = useEditorStrings()
  const zoneEditId = useStreamsStore(s => s.zoneEditId)
  const streams = useStreamsStore(s => s.streams)
  const setZoneEdit = useStreamsStore(s => s.setZoneEdit)
  // Zones are saved per camera (a role stream `<camera>@<role>` saves its camera's): wait on that camera's own ack,
  // so another zones process answering first cannot announce this one as active. An older sidecar acks only globally.
  const appliedVersion = useStreamsStore(s => {
    const camera = s.zoneEditId?.split('@')[0]
    return (camera !== undefined ? s.zonesStatus?.appliedByCamera?.[camera] : undefined)
      ?? s.zonesStatus?.appliedVersion ?? 0
  })
  const analyticsOff = useStreamsStore(s => s.zonesStatus?.enabled === false)
  const proximityApplied = useStreamsStore(s => s.zonesStatus?.proximity?.appliedVersion ?? 0)

  const [zones, setZones] = useState<DraftZone[]>([])
  const [selectedKey, setSelectedKey] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [phase, setPhase] = useState<SavePhase>('idle')
  const [error, setError] = useState<string | null>(null)
  const [streamKey, setStreamKey] = useState(0)
  const [naturalSize, setNaturalSize] = useState({ width: 0, height: 0 })
  const [draggingIndex, setDraggingIndex] = useState<number | null>(null)
  const [availableClasses, setAvailableClasses] = useState<string[]>(['person'])
  const [proximityInfo, setProximityInfo] = useState<ProximityInfo | null>(null)
  const [ppeItems, setPpeItems] = useState<string[] | undefined>(undefined)
  const [saved, setSaved] = useState<DraftZone[]>([])

  // The versions the save announced, per part of the pipeline that reads zones (nvdsanalytics, proximity).
  const targetsRef = useRef<{ zones?: number; proximity?: number } | null>(null)
  const applyTimerRef = useRef<number | null>(null)
  const closeTimerRef = useRef<number | null>(null)

  const clearPhaseTimers = () => {
    if (applyTimerRef.current !== null) window.clearTimeout(applyTimerRef.current)
    if (closeTimerRef.current !== null) window.clearTimeout(closeTimerRef.current)
    applyTimerRef.current = null
    closeTimerRef.current = null
  }

  // The component unmounts entirely on close (rendered conditionally on zoneEditId), so
  // pending timers must be cleared here.
  useEffect(() => clearPhaseTimers, [])

  // Sidecar ack: the poll loop saw applied_version catch up to the version our save
  // announced — the running pipeline is now enforcing the new zones.
  useEffect(() => {
    if (phase !== 'applying') return
    const t = targetsRef.current
    if (t && (t.zones === undefined || appliedVersion >= t.zones)
        && (t.proximity === undefined || proximityApplied >= t.proximity)) {
      clearPhaseTimers()
      setPhase('active')
      closeTimerRef.current = window.setTimeout(() => setZoneEdit(null), 1500)
    }
  }, [appliedVersion, proximityApplied, phase, setZoneEdit])

  const { ref: boxRef, width, height } = useElementSize<HTMLDivElement>()
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // The classes the pipeline can detect — the choices offered in the config panel.
  useEffect(() => {
    let cancelled = false
    zonesApi
      .detectionClasses()
      .then(res => { if (!cancelled && res.classes.length) setAvailableClasses(res.classes) })
      .catch(() => {/* fall back to the default ['person'] */})
    return () => { cancelled = true }
  }, [])

  // Load previously saved zones once per mount (remounted via key={zoneEditId} on switch).
  useEffect(() => {
    if (!zoneEditId) return
    let cancelled = false
    zonesApi
      .list(zoneEditId)
      .then(res => {
        if (cancelled) return
        const drafts = (res.zones ?? []).map(draftFromZone)
        setProximityInfo(res.proximity ?? null)
        setPpeItems(res.ppe?.items)
        setZones(drafts)
        setSaved(drafts)
        setSelectedKey(drafts[0]?.key ?? null)
      })
      .catch(err => {
        if (!cancelled) setError(err instanceof Error ? err.message : s.loadFailed)
      })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [zoneEditId])

  // Close on Escape while the overlay is open.
  useEffect(() => {
    if (!zoneEditId) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setZoneEdit(null) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [zoneEditId, setZoneEdit])

  // object-fit: contain math — the true displayed sub-rect of the video within its box,
  // since the camera's native aspect ratio may not match the panel's rendered box.
  const displayRect = useMemo(() => {
    if (!width || !height || !naturalSize.width || !naturalSize.height) return null
    const boxAspect = width / height
    const imgAspect = naturalSize.width / naturalSize.height
    if (imgAspect > boxAspect) {
      const w = width
      const h = width / imgAspect
      return { x: 0, y: (height - h) / 2, w, h }
    }
    const h = height
    const w = height * imgAspect
    return { x: (width - w) / 2, y: 0, w, h }
  }, [width, height, naturalSize])

  const selected = zones.find(z => z.key === selectedKey) ?? null

  // Redraw all zones whenever any zone, the selection, or the display box changes. The
  // selected zone is drawn solid with draggable vertex handles; the rest are dimmed.
  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx) return
    canvas.width = width
    canvas.height = height
    ctx.clearRect(0, 0, width, height)
    if (!displayRect) return

    const toPx = (p: ZonePoint): [number, number] => [
      p.x * displayRect.w + displayRect.x,
      p.y * displayRect.h + displayRect.y,
    ]

    for (const zone of zones) {
      if (zone.points.length === 0) continue
      const isSelected = zone.key === selectedKey
      const color = zoneColor(zone.intrusion.mode)
      ctx.globalAlpha = isSelected ? 1 : 0.4
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.beginPath()
      zone.points.forEach((p, i) => {
        const [x, y] = toPx(p)
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      if (zone.ringClosed || zone.closed) {
        ctx.closePath()
        const lineAlpha = ctx.globalAlpha
        ctx.globalAlpha = isSelected ? 0.2 : 0.1
        ctx.fillStyle = color
        ctx.fill()
        ctx.globalAlpha = lineAlpha
      }
      ctx.stroke()
      if (isSelected) {
        zone.points.forEach((p, i) => {
          const [x, y] = toPx(p)
          ctx.beginPath()
          ctx.arc(x, y, i === 0 ? 6 : 4, 0, Math.PI * 2)
          ctx.fillStyle = color
          ctx.fill()
        })
      }
    }
    ctx.globalAlpha = 1
  }, [zones, selectedKey, width, height, displayRect])

  const stream = streams.find(s => s.id === zoneEditId)
  if (!zoneEditId || (!stream && !source)) return null

  const live = source ? source.live : Boolean(stream?.running)
  const saving = phase === 'saving'
  const editable = phase === 'idle'

  const updateZone = (key: string, patch: Partial<DraftZone>) =>
    setZones(zs => zs.map(z => (z.key === key ? { ...z, ...patch } : z)))
  const updateSelected = (patch: Partial<DraftZone>) => {
    if (selectedKey) updateZone(selectedKey, patch)
  }

  const addZone = () => {
    const draft = makeDraftZone(s.newZone(zones.length + 1))
    setZones(zs => [...zs, draft])
    setSelectedKey(draft.key)
    setError(null)
  }
  const deleteZone = (key: string) => {
    setZones(zs => {
      const next = zs.filter(z => z.key !== key)
      if (key === selectedKey) setSelectedKey(next[0]?.key ?? null)
      return next
    })
    setError(null)
  }

  const handleImgLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget
    setNaturalSize({ width: img.naturalWidth, height: img.naturalHeight })
  }
  const handleImgError = () => setTimeout(() => setStreamKey(k => k + 1), 2000)

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (loading || !selected || selected.ringClosed || selected.closed || !displayRect) return
    const rect = e.currentTarget.getBoundingClientRect()
    const px = e.clientX - rect.left - displayRect.x
    const py = e.clientY - rect.top - displayRect.y
    if (px < 0 || py < 0 || px > displayRect.w || py > displayRect.h) return // in the letterbox

    if (selected.points.length >= 3) {
      const firstPx = selected.points[0].x * displayRect.w
      const firstPy = selected.points[0].y * displayRect.h
      if (Math.hypot(px - firstPx, py - firstPy) <= VERTEX_HIT_RADIUS) {
        updateSelected({ ringClosed: true })
        return
      }
    }
    updateSelected({ points: [...selected.points, { x: px / displayRect.w, y: py / displayRect.h }] })
  }

  const hitTestVertex = (px: number, py: number): number | null => {
    if (!displayRect || !selected) return null
    for (let i = 0; i < selected.points.length; i++) {
      const vx = selected.points[i].x * displayRect.w
      const vy = selected.points[i].y * displayRect.h
      if (Math.hypot(px - vx, py - vy) <= VERTEX_HIT_RADIUS) return i
    }
    return null
  }

  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!selected || !(selected.ringClosed || selected.closed) || !displayRect) return
    if (e.button !== 0) return
    const rect = e.currentTarget.getBoundingClientRect()
    const px = e.clientX - rect.left - displayRect.x
    const py = e.clientY - rect.top - displayRect.y
    const hit = hitTestVertex(px, py)
    if (hit === null) return
    setDraggingIndex(hit)
    e.currentTarget.setPointerCapture(e.pointerId)
  }

  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (draggingIndex === null || !selected || !displayRect) return
    const rect = e.currentTarget.getBoundingClientRect()
    const px = Math.min(Math.max(e.clientX - rect.left - displayRect.x, 0), displayRect.w)
    const py = Math.min(Math.max(e.clientY - rect.top - displayRect.y, 0), displayRect.h)
    const idx = draggingIndex
    updateSelected({
      points: selected.points.map((pt, i) =>
        i === idx ? { x: px / displayRect.w, y: py / displayRect.h } : pt),
    })
  }

  const handlePointerUp = () => setDraggingIndex(null)

  const handleClear = () =>
    updateSelected({ points: [], ringClosed: false, closed: false })

  const handleFinish = () => updateSelected({ closed: true })

  const blockingReason = zones.map(z => zoneInvalidReason(z, s)).find(r => r !== null) ?? null

  const handleSave = async () => {
    if (blockingReason) { setError(blockingReason); return }
    clearPhaseTimers()
    setPhase('saving')
    setError(null)
    try {
      const inputs: ZoneInput[] = zones.map(z => ({
        name: z.name.trim() || s.zoneFallback,
        points: z.points,
        intrusion: z.intrusion,
        overcrowding: z.overcrowding,
        ...(z.proximity ? { proximity: z.proximity } : {}),
        ...(z.ppe ? { ppe: z.ppe } : {}),
        ...(z.fall ? { fall: z.fall } : {}),
        ...(z.fire ? { fire: z.fire } : {}),
      }))
      const res = await zonesApi.saveAll(zoneEditId, inputs)
      setSaved(zones)
      const status = useStreamsStore.getState().zonesStatus
      const targets: { zones?: number; proximity?: number } = {}
      if (res.zones_version !== undefined && status?.enabled !== false) targets.zones = res.zones_version
      if (res.proximity_version !== undefined && status?.proximity) targets.proximity = res.proximity_version
      if (targets.zones !== undefined || targets.proximity !== undefined) {
        // Wait for the pipeline to acknowledge every part it runs before declaring the zones live.
        targetsRef.current = targets
        setPhase('applying')
        applyTimerRef.current = window.setTimeout(() => setPhase('apply_timeout'), APPLY_TIMEOUT_MS)
      } else {
        // Legacy backend, bump failure, or nothing in the pipeline reads zones — no ack will come.
        setPhase('saved')
        closeTimerRef.current = window.setTimeout(() => setZoneEdit(null), 2000)
      }
    } catch (err) {
      setPhase('idle')
      setError(err instanceof Error ? err.message : s.saveFailed)
    }
  }

  const saveDisabled = saving || phase === 'applying' || blockingReason !== null

  const info = loading
    ? s.loading
    : !selected
      ? s.addFirst
      : selected.ringClosed && !selected.closed
        ? s.ringClosed
        : selected.closed
          ? s.editing(selected.name)
          : `${s.points(selected.points.length)}${
              selected.points.length >= 3 ? s.closeHint : ''}`

  return (
    <div className={styles.backdrop} onClick={() => setZoneEdit(null)}>
      <div
        className={styles.panel}
        role="dialog"
        aria-modal="true"
        aria-label={s.editorLabel(zoneEditId)}
        onClick={e => e.stopPropagation()}
      >
        <header className={styles.header}>
          <div className={styles.headLeft}>
            <span className={styles.id}>{title ?? zoneEditId}</span>
            <span className={`${styles.status} ${live ? styles.statusLive : styles.statusOff}`}>
              <span className={styles.dot} />
              {source?.label ?? (live ? s.live : s.off)}
            </span>
          </div>
          <button
            type="button"
            className={styles.close}
            onClick={() => setZoneEdit(null)}
            aria-label={s.closeEditor}
          >
            <IconClose /> {s.close}
          </button>
        </header>

        <div className={styles.body}>
          <div className={styles.stage}>
            <div className={styles.video} ref={boxRef}>
              <img
                key={streamKey}
                className={styles.stream}
                src={source ? `${source.src}?t=${streamKey}` : `/stream/${zoneEditId}?t=${streamKey}`}
                alt={s.feedAlt(zoneEditId)}
                onLoad={handleImgLoad}
                onError={handleImgError}
              />
              <canvas
                ref={canvasRef}
                className={styles.canvas}
                style={{ cursor: selected && (selected.ringClosed || selected.closed) ? 'grab' : 'crosshair' }}
                onClick={handleCanvasClick}
                onPointerDown={handlePointerDown}
                onPointerMove={handlePointerMove}
                onPointerUp={handlePointerUp}
                onPointerCancel={handlePointerUp}
              />
            </div>
            <ZoneRail
              zones={zones}
              selectedKey={selectedKey}
              disabled={!editable}
              onSelect={setSelectedKey}
              onAdd={addZone}
              onDelete={deleteZone}
              onRename={(key, name) => updateZone(key, { name })}
            />
          </div>

          {settings}

          {selected && (
            <ZoneConfigPanel
              zone={selected}
              availableClasses={availableClasses}
              proximity={proximityInfo}
              ppeItems={ppeItems}
              disabled={!editable}
              onChange={updateSelected}
              features={features}
            />
          )}

          <div className={styles.toolbar}>
            <div className={styles.toolbarInfo}>{info}</div>
            <div className={styles.toolbarActions}>
              <button
                type="button"
                className="iris-btn iris-btn-ghost"
                onClick={handleClear}
                disabled={!selected || selected.points.length === 0 || !editable}
              >
                {s.clear}
              </button>
              <button
                type="button"
                className="iris-btn iris-btn-ghost"
                onClick={handleFinish}
                disabled={!selected || selected.closed || !selected.ringClosed || !editable}
              >
                {s.finish}
              </button>
              <button
                type="button"
                className="iris-btn iris-btn-accent"
                onClick={handleSave}
                disabled={saveDisabled}
              >
                {saving ? s.saving : s.save}
              </button>
            </div>
            {blockingReason && phase === 'idle' && (
              <span className={styles.warnText}>{blockingReason}</span>
            )}
            {phase === 'saved' && (
              <span className={analyticsOff && !proximityInfo?.available ? styles.mutedBadge : styles.savedBadge}>
                {analyticsOff && !proximityInfo?.available ? s.savedNoAnalytics : s.saved}
              </span>
            )}
            {phase === 'applying' && (
              <span className={styles.applyingBadge}>
                <span className={styles.badgeDot} />
                {s.applying}
              </span>
            )}
            {phase === 'active' && <span className={styles.savedBadge}>{s.active}</span>}
            {phase === 'apply_timeout' && (
              <span className={styles.warnText}>
                {s.applyTimeout}
              </span>
            )}
            {error && <span className={styles.errorText}>{error}</span>}
          </div>
          {notice?.(saved, zones)}
        </div>
      </div>
    </div>
  )
}
