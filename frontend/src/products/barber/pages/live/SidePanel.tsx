import { snapshotUrl } from '../../api/client'
import { Fragment, useCallback, useRef, useState, type ReactNode } from 'react'
import type { CameraConf, LiveCustomer, Scene, Staff, ZonesByCam } from '../../api/types'
import type { AreaTable } from '../../floor/areas'
import { useLang } from '../../i18n'
import { canPreview, previewTarget, shadow, type Seen } from '../../live/alertPreview'
import { holdFor, sidePanelModel, type Held, type LooseSnapshot, type SideAlert } from '../../live/sidePanel'
import { useStore, type Store } from '../../live/store'
import { Num } from '../../ui/Num'
import { Timer } from '../../ui/Timer'
import { AlertPreview, type Shot } from './AlertPreview'
import { WaitingList } from './WaitingList'
import { productHref } from '../../../../shared/router'

/** The average wait refreshes at most this often on screen. */
const AVG_WAIT_HOLD_MS = 60_000
/** Shadow tracking: an alert's customer counts as seen this long after he
 * drops out of the scene (97% of dropouts are shorter; see shadow()). */
const SHADOW_MS = 10_000

/** A translated text with its `{name}` placeholders replaced by elements. */
function fill(text: string, parts: Record<string, ReactNode>): ReactNode {
  return text.split(/(\{\w+\})/).map((s, i) => <Fragment key={i}>{parts[s.slice(1, -1)] ?? s}</Fragment>)
}

/** "A customer has been waiting for 18:40", the time running. */
function AlertLine({ a, now }: { a: SideAlert; now: number }) {
  const { t } = useLang()
  const chair = a.chair && <><Num>{a.chair}</Num>{a.barber && <span className="muted"> ({a.barber})</span>}</>
  const key = a.kind === 'excessive_wait' ? 'alert.waitingFor' : chair ? 'alert.cutAtFor' : 'alert.cutFor'
  return <>{fill(t(key), { chair, timer: <Timer since={a.since} now={now} target={a.target} className="sideAlertTimer" /> })}</>
}

const targetOf = (a: SideAlert, customers: LiveCustomer[], scene: Scene, areas: AreaTable, zones: ZonesByCam) =>
  previewTarget({ kind: a.kind, visit_id: a.visitId, chair_id: a.chairId }, customers, scene, areas, zones)

type Sight = (a: SideAlert, scene: Scene) => Seen | null

/** Preview, enabled while the customer is in the live scene (or was, within
 * SHADOW_MS). Subscribes to the scene itself, so only the buttons re-render
 * at the scene rate. */
function PreviewButton({ a, sight, scene, onOpen, btnRef }: {
  a: SideAlert; sight: Sight; scene: Store<Scene>
  onOpen: (a: SideAlert) => void; btnRef: (el: HTMLButtonElement | null) => void
}) {
  const { t } = useLang()
  // the live still while the scene has a target, else the picture saved when the alert opened (2026-10-18)
  const ok = canPreview(sight(a, useStore(scene)), a.stored)
  return (
    <button ref={btnRef} type="button" className="btnGhost btnSmall sideAlertPreview" disabled={!ok}
            title={ok ? undefined : t('alert.notTracked')} aria-haspopup="dialog" onClick={() => onOpen(a)}>
      {t('alert.preview')}
    </button>
  )
}

/** The open preview; Refresh is enabled while the customer is still seen and his alert still listed (a long
 * haircut's customer away: the chair would be empty). */
function OpenPreview({ a, shot, sight, scene, cameras, listed, retake, onClose }: {
  a: SideAlert; shot: Shot; sight: Sight; scene: Store<Scene>; cameras: CameraConf[]; listed: boolean
  retake: (a: SideAlert) => void; onClose: () => void
}) {
  const found = !!sight(a, useStore(scene))
  return <AlertPreview shot={shot} cameras={cameras} canRefresh={found && listed} onRefresh={() => retake(a)} onClose={onClose} />
}

export function SidePanel({ snap, now, staff, areas, scene, cameras }: {
  snap: LooseSnapshot; now: number; staff: Staff[]; areas: AreaTable; scene: Store<Scene>; cameras: CameraConf[]
}) {
  const { t, fmt } = useLang()
  // holdFor is pure in (previous, value, now): repeating a render is harmless
  const avgWait = useRef<Held | null>(null)
  avgWait.current = holdFor(avgWait.current, snap.today?.average_wait_s ?? null, now, AVG_WAIT_HOLD_MS)
  const today = snap.today && { ...snap.today, average_wait_s: avgWait.current.value }
  const customers = snap.customers ?? []
  const m = sidePanelModel({ ...snap, today }, areas, staff, n => fmt.number(n))
  // Preview: a still taken at the click, of the customer as the scene sees him then
  const [open, setOpen] = useState<{ a: SideAlert; shot: Shot } | null>(null)
  const btns = useRef<Record<string, HTMLButtonElement | null>>({})
  // Each alert's last sighting (by visit), shared by its button, the click and Refresh.
  // shadow() is pure in (previous, current, time): repeating a render is harmless.
  const seen = useRef(new Map<string, Seen>())
  const sight: Sight = (a, sc) => {
    const s = shadow(seen.current.get(a.visitId), targetOf(a, customers, sc, areas, snap.zones ?? {}), Date.now(), SHADOW_MS)
    if (s) seen.current.set(a.visitId, s)
    else seen.current.delete(a.visitId)
    return s
  }
  const take = (a: SideAlert) => {
    const s = sight(a, scene.get())
    if (!s) {
      if (a.stored) setOpen({ a, shot: { cam: a.stored.cam, box: a.stored.box, kind: a.kind, at: Date.parse(a.stored.openedAt),
                                         bust: 0, ago: 0, src: snapshotUrl(a.id) } })
      return
    }
    const ago = Math.floor((Date.now() - s.at) / 1000)
    setOpen({ a, shot: { ...s.target, kind: a.kind, at: now, bust: Date.now(), ago } })
  }
  const openId = open?.a.id
  // Closing gives focus back to the alert's Preview button (if the alert is still open).
  const close = useCallback(() => {
    setOpen(null)
    if (openId) requestAnimationFrame(() => btns.current[openId]?.focus())
  }, [openId])
  return (
    <aside className="sidePanel">
      <section className="card sideBlock">
        <h2 className="sectionLabel">{t('side.today')}</h2>
        <dl className="sideFigures">
          {m.figures.map(f => (
            <div key={f.label}>
              <dt>{t(f.label)}</dt>
              <dd><Num>{f.value}</Num></dd>
            </div>
          ))}
        </dl>
      </section>

      <WaitingList snap={snap} now={now} areas={areas} staff={staff} />

      <section className="card sideBlock sideScroll">
        <h2 className="sectionLabel">{t('side.openAlerts')}</h2>
        {m.alerts.length === 0
          ? <p className="muted small sideNone">{t('side.noAlerts')}</p>
          : (
            <ul className="sideAlerts">
              {m.alerts.map(a => (
                <li key={a.id} className="sideAlert">
                  <span><AlertLine a={a} now={now} /></span>
                  <PreviewButton a={a} sight={sight} scene={scene} onOpen={take}
                                 btnRef={el => { btns.current[a.id] = el }} />
                </li>
              ))}
            </ul>
          )}
      </section>

      {m.rosterTodo && (
        <section className="card sideBlock sideNudge">
          <p>{t('side.rosterTodo')}</p>
          <a className="btnGhost btnSmall" href={productHref('/staff')}>{t('side.rosterGo')}</a>
        </section>
      )}
      {m.zoneProblems > 0 && (
        <section className="card sideBlock sideNudge">
          <p>{t('side.zoneProblems', { n: fmt.number(m.zoneProblems) })}</p>
          <a className="btnGhost btnSmall" href={productHref('/setup')}>{t('side.zoneGo')}</a>
        </section>
      )}
      {open && (
        <OpenPreview a={open.a} shot={open.shot} sight={sight} scene={scene} cameras={cameras}
                     listed={m.alerts.some(a => a.id === open.a.id)} retake={take} onClose={close} />
      )}
    </aside>
  )
}
