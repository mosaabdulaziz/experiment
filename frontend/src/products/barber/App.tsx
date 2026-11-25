import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from './api/client'
import { rememberCameraNames } from './ui/cameraLabel'
import type { CameraConf, Staff } from './api/types'
import { useLang, type Key } from './i18n'
import { useLive, type Live } from './live/store'
import { Shell } from './ui/Shell'
import { Placeholder } from './ui/Placeholder'
import { LivePage } from './pages/live/LivePage'
import { InsightsPage } from './pages/InsightsPage'
import { ReportsPage } from './pages/ReportsPage'
import { AlertsPage } from './pages/AlertsPage'
import { CustomersPage } from './pages/CustomersPage'
import { StaffChairsPage } from './pages/StaffChairsPage'
import { SetupPage } from './pages/setup/SetupPage'
import { onRouteChange, productHref, productPath } from '../../shared/router'

/** Tiny hash router: '#/insights?visit=…&from=…' -> { path, params }. */
export interface Route {
  path: string
  params: URLSearchParams
}

function parseHash(): Route {
  const raw = productPath()
  const [path, q = ''] = raw.split('?')
  return { path: path || '/', params: new URLSearchParams(q) }
}

/** What App hands every page (later tasks' pages take what they need). */
export interface PageProps {
  live: Live
  staff: Staff[]
  cameras: CameraConf[]
  /** The one-shot camera list fetch: pages wait while 'loading'. */
  camerasState: 'loading' | 'ok' | 'failed'
  reloadStaff: () => Promise<void>
  setDirty: (dirty: boolean) => void
}

export const customerLink = (id: string): string => productHref('/customers?id=' + id)

const PAGES: Record<string, Key> = {
  '/': 'nav.live',
  '/insights': 'nav.insights',
  '/reports': 'nav.reports',
  '/customers': 'nav.customers',
  '/alerts': 'nav.alerts',
  '/staff': 'nav.staff',
  '/setup': 'nav.setup',
}

export function App() {
  const { t } = useLang()
  const [route, setRoute] = useState(parseHash)
  const live = useLive()
  const [staff, setStaff] = useState<Staff[]>([])
  const [cameras, setCameras] = useState<CameraConf[]>([])
  const [camerasState, setCamerasState] = useState<PageProps['camerasState']>('loading')
  // Setup's zone editor is a draft: leaving it with unsaved edits asks first.
  const setupDirty = useRef(false)
  const lastHash = useRef(location.hash)
  const leaveText = useRef('')
  leaveText.current = t('setup.leaveUnsaved')

  useEffect(() => {
    const onChange = () => {
      const setupHash = productHref('/setup')
      const leavingSetup = lastHash.current.startsWith(setupHash) && !location.hash.startsWith(setupHash)
      if (leavingSetup && setupDirty.current) {
        if (!window.confirm(leaveText.current)) {
          history.replaceState(null, '', lastHash.current)
          return
        }
        setupDirty.current = false
      }
      lastHash.current = location.hash
      setRoute(parseHash())
    }
    const onUnload = (e: BeforeUnloadEvent) => {
      if (setupDirty.current) {
        e.preventDefault()
        e.returnValue = ''
      }
    }
    const offRoute = onRouteChange(onChange)
    window.addEventListener('beforeunload', onUnload)
    return () => {
      offRoute()
      window.removeEventListener('beforeunload', onUnload)
    }
  }, [])

  const reloadStaff = useCallback(() => api.staff().then(setStaff).catch(() => {}), [])
  const loadCameras = useCallback(() => {
    api.cameras()
      .then(c => { rememberCameraNames(c); setCameras(c); setCamerasState('ok') })
      .catch(() => setCamerasState(st => (st === 'ok' ? st : 'failed')))
  }, [])
  useEffect(() => {
    reloadStaff()
    loadCameras()
  }, [reloadStaff, loadCameras])
  // If the first fetch failed (barber-service still starting), try again
  // each time the live socket (re)connects, until the cameras are in.
  useEffect(() => {
    if (live.conn === 'connected' && camerasState === 'failed') {
      reloadStaff()
      loadCameras()
    }
  }, [live.conn, camerasState, reloadStaff, loadCameras])
  const setDirty = useCallback((d: boolean) => { setupDirty.current = d }, [])

  const pageProps: PageProps = { live, staff, cameras, camerasState, reloadStaff, setDirty }
  const openAlerts = live.snap?.open_alerts?.length ?? 0
  const page = PAGES[route.path] ?? 'nav.live'

  return (
    <Shell route={route} openAlerts={openAlerts} site={live.snap?.site}>
      {route.path === '/' || !(route.path in PAGES)
        ? <LivePage {...pageProps} />
        : route.path === '/insights'
        ? <InsightsPage key={route.params.toString()} params={route.params} {...pageProps} />
        : route.path === '/reports'
        ? <ReportsPage {...pageProps} />
        : route.path === '/customers'
        ? <CustomersPage params={route.params} {...pageProps} />
        : route.path === '/alerts'
        ? <AlertsPage {...pageProps} />
        : route.path === '/staff'
        ? <StaffChairsPage {...pageProps} />
        : route.path === '/setup'
        ? <SetupPage {...pageProps} />
        : <Placeholder name={t(page)} {...pageProps} />}
    </Shell>
  )
}
