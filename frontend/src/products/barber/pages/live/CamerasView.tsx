import { useCallback, useEffect, useRef, useState, type MouseEvent } from 'react'
import { api } from '../../api/client'
import type { CameraConf, KidsCarState } from '../../api/types'
import type { AreaTable } from '../../floor/areas'
import { useLang } from '../../i18n'
import type { Live } from '../../live/store'
import { cameraLabel } from '../../ui/cameraLabel'
import { Feed, type FeedBadge, type Rect } from '../../ui/Feed'
import { fmtVideoTime } from '../../ui/format'
import { CameraWindow } from './CameraWindow'
import { pointIn } from './kidsCarPick'
import { shownZones } from './zoneDisplay'
import { ChairsOverlay, OccupancyOverlay, PeopleOverlay, TracksOverlay, ZoneOverlay, ZoneOverlayWithChairs } from './Overlay'

type Layers = { zones: boolean; chairs: boolean; people: boolean; occupancy: boolean; tracks: boolean }
const LAYERS_KEY = 'barber.camLayers'
const LAYER_LABEL = { zones: 'cam.zones', chairs: 'cam.layer.chairs', people: 'cam.people', occupancy: 'cam.layer.occupancy', tracks: 'cam.layer.tracks' } as const

function readLayers(): Layers {
  try {
    const v = JSON.parse(localStorage.getItem(LAYERS_KEY) ?? 'null')
    if (v && typeof v === 'object') {
      return { zones: v.zones !== false, chairs: v.chairs !== false, people: v.people !== false, occupancy: v.occupancy === true, tracks: v.tracks === true }
    }
  } catch { /* not remembered */ }
  return { zones: true, chairs: true, people: true, occupancy: false, tracks: false }
}

/** The three cameras with what the engine sees drawn on them: the centre
 * camera big, the side cameras stacked beside it. Same physical order in
 * Arabic (a floor view never mirrors). A tile opens its camera enlarged. */
export function CamerasView({ live, cameras, areas }: { live: Live; cameras: CameraConf[]; areas: AreaTable }) {
  const { t } = useLang()
  const [layers, setLayersState] = useState<Layers>(readLayers)
  const toggle = (k: keyof Layers) => {
    const next = { ...layers, [k]: !layers[k] }
    setLayersState(next)
    try { localStorage.setItem(LAYERS_KEY, JSON.stringify(next)) } catch { /* not remembered */ }
  }
  const [open, setOpen] = useState<string | null>(null)
  const tileBtns = useRef<Record<string, HTMLButtonElement | null>>({})
  const openRef = useRef(open)
  openRef.current = open
  // Closing gives focus back to the tile that opened the window.
  const close = useCallback(() => {
    const cam = openRef.current
    setOpen(null)
    if (cam) requestAnimationFrame(() => tileBtns.current[cam]?.focus())
  }, [])
  // The kids car override (2026-10-15): from an enlarged camera, "the car is here" then a click on the picture.
  const [car, setCar] = useState<KidsCarState | null>(null)
  const [placing, setPlacing] = useState(false)
  const [carFailed, setCarFailed] = useState(false)
  useEffect(() => {
    setPlacing(false)
    setCarFailed(false)
    if (!open) return
    let off = false
    api.kidsCar().then(c => { if (!off) setCar(c) }).catch(() => { if (!off) setCar(null) })
    return () => { off = true }
  }, [open])
  const pinCar = (cam: string, rect: Rect) => (e: MouseEvent<SVGRectElement>) => {
    const svg = e.currentTarget.ownerSVGElement
    const pt = svg && pointIn(rect, e.clientX, e.clientY, svg.getBoundingClientRect())
    if (!pt) return
    setPlacing(false)
    api.pinKidsCar(cam, pt[0], pt[1]).then(c => { setCar(c); setCarFailed(false) }).catch(() => setCarFailed(true))
  }
  const clearCar = () => { api.clearKidsCar().then(c => { setCar(c); setCarFailed(false) }).catch(() => setCarFailed(true)) }
  const carControls = (cam: string) => car?.enabled ? (
    <span className="feedKidsCar">
      {car.place?.camera === cam && <span className="feedClip">{t(car.place.source === 'manual' ? 'kc.manual' : 'kc.auto')}</span>}
      <button type="button" className="toggleChip" aria-pressed={placing} onClick={() => setPlacing(v => !v)}>
        {placing ? t('kc.placing') : t('kc.place')}
      </button>
      {car.place?.source === 'manual' && <button type="button" className="toggleChip" onClick={clearCar}>{t('kc.clear')}</button>}
      {carFailed && <span className="feedClip" role="alert">{t('kc.failed')}</span>}
    </span>
  ) : null
  const { snap, now } = live
  const main = cameras.find(c => c.id === areas.centre) ?? cameras[0]
  const even = !areas.centre && cameras.length > 3
  const ordered = main ? [main, ...cameras.filter(c => c !== main)] : []

  if (!ordered.length) return <p className="card muted">{t('cam.noVideo')}</p>

  const caption = (cam: string) => <span className="feedName">{cameraLabel(cam, t)}</span>

  // 2026-10-18 (user): on the picture, not above it -- "Live", or a recording's position (it was the caption's clip text)
  const badge = (cam: string): FeedBadge => {
    const clip = snap?.clips?.[cam]
    if (!clip || clip.duration_s <= 0) return { kind: 'live', text: t('cam.live') }
    const at = fmtVideoTime(((now - clip.started_ms) / 1000) % clip.duration_s)
    return { kind: 'clip', text: `${at} / ${fmtVideoTime(clip.duration_s)}`, label: t('cam.replayAt', { t: at }) }
  }

  const overlays = (cam: string) => (rect: Rect) => (
    <>
      {layers.zones && (layers.chairs
        ? <ZoneOverlayWithChairs zones={shownZones(snap?.zones?.[cam] ?? [], snap?.zone_display)} camZones={snap?.zones?.[cam] ?? []}
                                 rect={rect} chairs={live.chairs} cam={cam} now={now} />
        : <ZoneOverlay zones={shownZones(snap?.zones?.[cam] ?? [], snap?.zone_display)} rect={rect} />)}
      {layers.chairs && <ChairsOverlay chairs={live.chairs} zones={snap?.zones?.[cam] ?? []} cam={cam} rect={rect} now={now} />}
      {layers.people && <PeopleOverlay store={live.scene} cam={cam} rect={rect} now={now} customers={snap?.customers} occupancyOn={layers.occupancy} />}
      {layers.occupancy && <OccupancyOverlay store={live.scene} cam={cam} rect={rect} />}
      {layers.tracks && <TracksOverlay store={live.scene} cam={cam} rect={rect} />}
    </>
  )
  const openCam = ordered.find(c => c.id === open)

  return (
    <div className="camerasView">
      <div className={even ? 'camGrid camGridEven' : ordered.length > 1 ? 'camGrid camGridMulti' : 'camGrid'}>
        {ordered.map((c, i) => (
          <Feed key={c.id} live={c.stream} caption={caption(c.id)} badge={badge(c.id)}
                className={`camTile ${!even && i === 0 ? 'camMain' : 'camSide'}`}
                action={
                  <button ref={el => { tileBtns.current[c.id] = el }} type="button" className="camEnlarge"
                          aria-label={t('cam.enlarge', { cam: cameraLabel(c.id, t) })}
                          title={t('cam.enlarge', { cam: cameraLabel(c.id, t) })}
                          aria-haspopup="dialog" onClick={() => setOpen(c.id)} />
                }>
            {overlays(c.id)}
          </Feed>
        ))}
      </div>
      {openCam && (
        <CameraWindow live={openCam.stream} title={cameraLabel(openCam.id, t)} onClose={close} badge={badge(openCam.id)}
                      caption={<>{caption(openCam.id)}{carControls(openCam.id)}</>}>
          {(rect: Rect) => (
            <>
              {overlays(openCam.id)(rect)}
              {placing && <rect x={rect.x} y={rect.y} width={rect.w} height={rect.h} fill="transparent"
                                style={{ pointerEvents: 'all', cursor: 'crosshair' }} onClick={pinCar(openCam.id, rect)} />}
            </>
          )}
        </CameraWindow>
      )}
      <div className="camToggles" role="group" aria-label={t('cam.layers')}>
        {(['zones', 'chairs', 'people', 'occupancy', 'tracks'] as const).map(k => (
          <button key={k} type="button" className="toggleChip" aria-pressed={layers[k]} onClick={() => toggle(k)}>
            {t(LAYER_LABEL[k])}
          </button>
        ))}
      </div>
    </div>
  )
}
