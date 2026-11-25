/** The safety console: the top bar, the acknowledgement banner on every page, and the pages (decision D28).
 *  The top bar is IRIS's header (its stylesheet and logo, reused as they are): the brand, the pages in the middle,
 *  the connection and the language at the far end. */
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ComponentType, type SVGProps } from 'react'
import hs from '@/components/layout/Header.module.css'
import { BrandLogo } from '@/components/ui/BrandLogo'
import { IconAlert, IconDashboard, IconGear, IconIncidents } from '@/components/ui/icons'
import { onRouteChange, productHref, productPath } from '../../shared/router'
import { api } from './api'
import { ApplyBar } from './components/ApplyBar'
import { Banner } from './components/Banner'
import { CameraEditor } from './components/CameraEditor'
import { ClipModal } from './components/IncidentCard'
import { useFeed, usePoll, useSiteZones } from './data'
import { LangProvider, useLang, type Key } from './i18n'
import { setupWarnings, type SetupWarning } from './lib/coverage'
import { waitingList } from './lib/incidents'
import { PAGES, parseRoute, type Page } from './lib/route'
import type { SafetyCamera, SafetyEvent, Settings } from './lib/types'
import { HealthPage } from './pages/HealthPage'
import { IncidentsPage } from './pages/IncidentsPage'
import { LivePage } from './pages/LivePage'
import { SettingsPage } from './pages/SettingsPage'

interface Console {
  events: SafetyEvent[]
  /** Incidents waiting for an acknowledgement (the banner's list), newest first. */
  waiting: SafetyEvent[]
  connected: boolean
  settings: Settings | null
  reloadSettings: () => void
  cameras: SafetyCamera[] | null
  camerasError: string | null
  reloadCameras: () => void
  cameraName: (id: string | null) => string
  acknowledge: (ev: SafetyEvent) => void
  play: (ev: SafetyEvent) => void
  /** Open a camera's settings and zones (the zone editor), drawing on `streamId`, one of its role streams -- or, with
   *  none (no feature runs on it), on its still from go2rtc. */
  editCamera: (cameraId: string, streamId: string | null) => void
  /** Bumped when the zone editor closes: what shows a camera's zones loads them again. */
  zonesVersion: number
  /** Features switched on that cannot raise anything yet (no zone for them), across the site. */
  setup: SetupWarning[]
}
const Ctx = createContext<Console | null>(null)

export function useConsole(): Console {
  const v = useContext(Ctx)
  if (!v) throw new Error('useConsole outside the safety console')
  return v
}

const PAGE_LABEL: Record<Page, Key> = {
  live: 'nav.live', incidents: 'nav.incidents', health: 'nav.health', settings: 'nav.settings',
}
const PAGE_ICON: Record<Page, ComponentType<SVGProps<SVGSVGElement>>> = {
  live: IconDashboard, incidents: IconIncidents, health: IconAlert, settings: IconGear,
}

function ConsoleRoot() {
  const { t, lang, setLang, setDefault } = useLang()
  const [route, setRoute] = useState(() => parseRoute(productPath()))
  useEffect(() => onRouteChange(() => setRoute(parseRoute(productPath()))), [])

  const feed = useFeed()
  const settings = usePoll(api.settings, 60000)
  const cameras = usePoll(api.cameras, 15000)
  const unacked = usePoll(api.unacknowledged, 30000)
  const [justAcked, setJustAcked] = useState<Set<number>>(new Set())
  const [playing, setPlaying] = useState<SafetyEvent | null>(null)
  const [editing, setEditing] = useState<{ cameraId: string; streamId: string | null } | null>(null)
  const [zonesVersion, setZonesVersion] = useState(0)
  const editCamera = useCallback((cameraId: string, streamId: string | null) => setEditing({ cameraId, streamId }), [])
  const siteZones = useSiteZones(cameras.data, zonesVersion)
  const setup = useMemo(() => setupWarnings(cameras.data ?? [], siteZones), [cameras.data, siteZones])
  const closeEditor = useCallback(() => { setEditing(null); setZonesVersion(v => v + 1) }, [])

  useEffect(() => { if (settings.data) setDefault(settings.data.language) }, [settings.data, setDefault])

  const names = useMemo(() => new Map((cameras.data ?? []).map(c => [c.id, c.name])), [cameras.data])
  const cameraName = useCallback((id: string | null) => (id && names.get(id)) || id || '—', [names])

  const { patch } = feed
  const reloadUnacked = unacked.reload
  const acknowledge = useCallback((ev: SafetyEvent) => {
    setJustAcked(s => new Set(s).add(ev.id))
    api.acknowledge(ev.id)
      .then(r => { patch(ev.id, { acknowledged_at: r.acknowledged_at }); reloadUnacked() })
      .catch(() => setJustAcked(s => { const n = new Set(s); n.delete(ev.id); return n }))
  }, [patch, reloadUnacked])

  const waiting = useMemo(() => waitingList(unacked.data ?? [], feed.events, settings.data, justAcked),
    [unacked.data, feed.events, settings.data, justAcked])

  const value: Console = {
    events: feed.events, waiting, connected: feed.connected,
    settings: settings.data, reloadSettings: settings.reload,
    cameras: cameras.data, camerasError: cameras.error, reloadCameras: cameras.reload,
    cameraName, acknowledge, play: setPlaying, editCamera, zonesVersion, setup,
  }

  return (
    <Ctx.Provider value={value}>
      <header className={`${hs.header} sf-top`}>
        <a href={productHref('/live')} className={hs.brand}>
          <BrandLogo size={34} />
          <div className={hs.brandText}>
            <span className={hs.brandName}><span className={hs.brandAccent}>IRIS</span></span>
            <span className={hs.brandSub}>{t('app.title')}</span>
          </div>
        </a>
        <nav className={`${hs.nav} sf-nav`}>
          {PAGES.map(p => {
            const Icon = PAGE_ICON[p]
            return (
              <a key={p} href={productHref(`/${p}`)} aria-current={route.page === p ? 'page' : undefined}
                className={`${hs.navLink} ${route.page === p ? hs.navLinkActive : ''}`}>
                <Icon width={16} height={16} />
                {t(PAGE_LABEL[p])}
              </a>
            )
          })}
        </nav>
        <div className={`${hs.right} sf-top-end`}>
          <span className={`${hs.badge} ${feed.connected ? hs.badgeLive : hs.badgeError} sf-conn`}>
            <span className={`${hs.liveDot} ${feed.connected ? '' : hs.dotMuted}`} aria-hidden="true" />
            {t(feed.connected ? 'conn.live' : 'conn.down')}
          </span>
          <button type="button" className="iris-btn iris-btn-sm" onClick={() => setLang(lang === 'en' ? 'ar' : 'en')}
            lang={lang === 'en' ? 'ar' : 'en'}>
            {t('lang.other')}
          </button>
        </div>
      </header>
      <Banner waiting={waiting} cameraName={cameraName} onAcknowledge={acknowledge} onPlay={setPlaying} />
      <main>
        {route.page === 'live' && <LivePage />}
        {route.page === 'incidents' && <IncidentsPage query={route.query} />}
        {route.page === 'health' && <HealthPage />}
        {route.page === 'settings' && <SettingsPage />}
      </main>
      <ApplyBar />
      {editing && <CameraEditor cameraId={editing.cameraId} streamId={editing.streamId} onClose={closeEditor} />}
      {playing && <ClipModal event={playing} onClose={() => setPlaying(null)} cameraName={cameraName} />}
    </Ctx.Provider>
  )
}

export function App() {
  return (
    <LangProvider>
      <div className="sf-root">
        <ConsoleRoot />
      </div>
    </LangProvider>
  )
}
