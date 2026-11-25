/** The camera wall and, beside it, what still needs setting up and the latest incidents. The tiles fill the wall,
 *  however many there are (IRIS's layoutRows), and are IRIS's tiles: the camera's name and state on top, its Configure
 *  button (the zone editor with the camera's name and features, CameraEditor), and along the bottom its features as tags -- each role sidecar draws
 *  its own overlay, the tags pick whose the tile shows. A camera with an incident going on is outlined, and its tile
 *  flashes IRIS's alert ring when a new one starts. A camera no
 *  feature runs on stays on the wall with its still from go2rtc, refreshed every few seconds. */
import { useEffect, useRef, useState, type CSSProperties } from 'react'
import { useFramePoll } from '@/components/streams/framePoll'
import { layoutRows } from '@/components/streams/layout'
import ts from '@/components/streams/StreamTile.module.css'
import { IconAlert, IconGear, IconZone } from '@/components/ui/icons'
import { ResizeHandle } from '@/components/ui/ResizeHandle'
import { useElementSize } from '@/hooks/useElementSize'
import { usePanelWidth } from '@/hooks/usePanelWidth'
import { navigate } from '../../../shared/router'
import { stillUrl } from '../api'
import { useConsole } from '../App'
import { UNUSED_TEXT } from '../components/CameraSettings'
import { FeatureTag } from '../components/FeatureTag'
import { IncidentCard } from '../components/IncidentCard'
import { useStill } from '../data'
import { useLang } from '../i18n'
import { isOpen, newIncident, sortIncidents } from '../lib/incidents'
import type { SafetyCamera } from '../lib/types'

const RAIL = 12
const SETUP_SHOWN = 4
// The incidents beside the wall: as wide as the operator drags them (IRIS's alerts panel does the same).
const RAIL_DEFAULT = 360
const RAIL_LIMITS = { min: 280, max: 820 }

export function LivePage() {
  const { t } = useLang()
  const { events, cameras, settings, cameraName, acknowledge, play } = useConsole()
  const shown = cameras ?? []
  const openBy = new Map<string, number>()
  const newestOpen = new Map<string, number>()
  for (const e of events) {
    if (!isOpen(e) || !e.camera_id) continue
    openBy.set(e.camera_id, (openBy.get(e.camera_id) ?? 0) + 1)
    newestOpen.set(e.camera_id, Math.max(newestOpen.get(e.camera_id) ?? 0, e.id))
  }
  const latest = sortIncidents(events).slice(0, RAIL)
  const [railWidth, setRailWidth] = usePanelWidth('safety.railWidth', RAIL_DEFAULT, RAIL_LIMITS)
  const { ref: wallRef, width, height } = useElementSize<HTMLDivElement>()
  const rows = layoutRows(shown, width, height)

  return (
    <div className="sf-page sf-live" style={{ '--rail-w': `${railWidth}px` } as CSSProperties}>
      <div className="sf-wall" ref={wallRef}>
        {cameras && !shown.length && <p className="sf-empty">{t('live.none')}</p>}
        {rows.map((row, i) => (
          <div key={i} className="sf-wall-row">
            {row.map(c => c.streams.length
              ? <Tile key={c.id} camera={c} open={openBy.get(c.id) ?? 0} newest={newestOpen.get(c.id) ?? null} />
              : <StillTile key={c.id} camera={c} />)}
          </div>
        ))}
      </div>
      {/* the handle sits on the rail's edge, outside it: beside the scrolling rail, not in it (it would be clipped) */}
      <div className="sf-rail-wrap">
        <ResizeHandle width={railWidth} onChange={setRailWidth} limits={RAIL_LIMITS} defaultWidth={RAIL_DEFAULT}
          label={t('live.resize')} />
        <aside className="sf-rail" aria-label={t('live.latest')}>
          <SetupList />
          <div className="sf-page-head">
            <h2>{t('live.latest')}</h2>
            <button type="button" className="iris-btn" onClick={() => navigate('/incidents')}>{t('live.allIncidents')}</button>
          </div>
          {!latest.length && <p className="sf-muted">{t('live.quiet')}</p>}
          <div className="sf-rail-list">
            {latest.map(ev => (
              <IncidentCard key={ev.id} event={ev} settings={settings} cameraName={cameraName} onAcknowledge={acknowledge}
                onPlay={play} compact />
            ))}
          </div>
        </aside>
      </div>
    </div>
  )
}

/** True for the length of IRIS's alert ring (two 1.3 s rings) each time a newer incident than the tile has seen
 *  starts on its camera (lib/incidents.ts newIncident). */
function useFlash(newest: number | null, ms = 2700): boolean {
  const [on, setOn] = useState(false)
  const seen = useRef<number | null>(null)
  useEffect(() => {
    const next = newIncident(seen.current, newest)
    seen.current = next.seen
    if (!next.flash) return
    setOn(true)
    const timer = setTimeout(() => setOn(false), ms)
    return () => clearTimeout(timer)
  }, [newest, ms])
  return on
}

function Tile({ camera, open, newest }: { camera: SafetyCamera; open: number; newest: number | null }) {
  const { t } = useLang()
  const { editCamera } = useConsole()
  const flash = useFlash(newest)
  const streams = [...camera.streams].sort((a, b) => Number(b.running) - Number(a.running))
  const [picked, setPicked] = useState<string | null>(null)
  const stream = streams.find(s => s.id === picked) ?? streams[0]
  const frame = useFramePoll(stream.id, stream.running, 400)
  return (
    <article className={`${ts.tile} sf-tile ${flash ? ts.flash : ''}`} data-alert={open ? true : undefined}
      aria-label={camera.name}>
      <div className={ts.video}>
        {stream.running
          ? <img key={stream.id} className={ts.stream} {...frame}
              alt={`${camera.name}: ${stream.feature ? t(`feature.${stream.feature}`) : ''}`} />
          : <span className="sf-tile-off">{t('live.noPicture')}</span>}
        {!stream.running && <div className={ts.offScrim} />}
      </div>
      <div className={ts.topBar}>
        <span className={ts.id}>{camera.name}</span>
        {open > 0 && <span className="sf-tile-alert">{t('live.open', { n: open })}</span>}
        <span className={`${ts.status} ${stream.running ? ts.statusLive : ts.statusOff}`}>
          <span className={ts.dot} />
          {t(stream.running ? 'live.on' : 'live.off')}
        </span>
      </div>
      <div className={`${ts.bottomBar} sf-tile-tags`}>
        {streams.map(s => s.feature && (
          <FeatureTag key={s.id} feature={s.feature} pressed={s.id === stream.id} stopped={!s.running}
            onClick={() => setPicked(s.id)} />
        ))}
      </div>
      <button type="button" className={`${ts.zoneBtn} ${ts.zoneBtnCorner}`}
        aria-label={t('live.editCamera', { camera: camera.name })}
        onClick={() => editCamera(camera.id, stream.running ? stream.id : null)}>
        <IconGear width={14} height={14} />
        {t('live.configure')}
      </button>
    </article>
  )
}

/** A camera no feature runs on: its still from go2rtc (no sidecar decodes it), what it is waiting for, and its
 *  Configure button -- features switched on and zones drawn before anything starts. */
function StillTile({ camera }: { camera: SafetyCamera }) {
  const { t } = useLang()
  const { editCamera } = useConsole()
  const still = useStill(stillUrl(camera.id))
  const [failed, setFailed] = useState(false)
  return (
    <article className={`${ts.tile} sf-tile sf-tile-still`} aria-label={camera.name}>
      <div className={ts.video}>
        <img className={ts.stream} src={still.src} alt={camera.name}
          onLoad={() => { setFailed(false); still.onLoad() }} onError={() => { setFailed(true); still.onError() }} />
        {failed && <span className="sf-tile-off">{t('live.noPicture')}</span>}
      </div>
      <div className={ts.topBar}>
        <span className={ts.id}>{camera.name}</span>
        <span className={`${ts.status} ${ts.statusOff}`}>
          <span className={ts.dot} />
          {t('live.still')}
        </span>
      </div>
      <div className={`${ts.bottomBar} sf-tile-tags`}>
        <span className="sf-tile-note">{t(camera.features.length ? 'live.startsOnApply' : 'live.noFeatures')}</span>
      </div>
      <button type="button" className={`${ts.zoneBtn} ${ts.zoneBtnCorner}`}
        aria-label={t('live.editCamera', { camera: camera.name })} onClick={() => editCamera(camera.id, null)}>
        <IconGear width={14} height={14} />
        {t('live.configure')}
      </button>
    </article>
  )
}

/** Features switched on that cannot raise anything yet (a camera's proximity or zone rules with no zone for them),
 *  above the incidents: each opens that camera's zone editor. */
function SetupList() {
  const { t } = useLang()
  const { setup, editCamera } = useConsole()
  const [all, setAll] = useState(false)
  if (!setup.length) return null
  return (
    <section className="sf-setup" aria-label={t('setup.title', { n: setup.length })}>
      <h2 className="sf-setup-head">
        <IconAlert width={14} height={14} aria-hidden="true" />
        {t('setup.title', { n: setup.length })}
      </h2>
      <ul>
        {(all ? setup : setup.slice(0, SETUP_SHOWN)).map(({ camera, feature }) => {
          const text = UNUSED_TEXT[feature]
          const running = camera.streams.find(s => s.running)
          return (
            <li key={`${camera.id}:${feature}`} className="sf-setup-item">
              <div className="sf-setup-top">
                <FeatureTag feature={feature} />
                <span className="sf-setup-cam">{camera.name}</span>
              </div>
              {text && <p>{t(text)}</p>}
              <button type="button" className="iris-btn iris-btn-sm"
                onClick={() => editCamera(camera.id, running?.id ?? null)}>
                <IconZone width={12} height={12} />
                {t('setup.draw')}
              </button>
            </li>
          )
        })}
      </ul>
      {setup.length > SETUP_SHOWN && (
        <button type="button" className="iris-btn iris-btn-sm" aria-expanded={all} onClick={() => setAll(a => !a)}>
          {all ? t('setup.fewer') : t('setup.more', { n: setup.length - SETUP_SHOWN })}
        </button>
      )}
    </section>
  )
}

