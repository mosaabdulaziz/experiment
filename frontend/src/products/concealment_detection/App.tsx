/** The tab: the hash route, the shell, and the concealment review service's own data. */
import { useCallback, useEffect, useRef, useState } from 'react'
import { navigate, onRouteChange, productPath } from '../../shared/router'
import {
  assessments as fetchAssessments,
  cameras as fetchCameras,
  candidates as fetchCandidates,
  review,
  status as fetchStatus,
  type Assessment,
  type CameraInfo,
  type Candidate,
  type ReviewVerdict,
  type Status,
} from './api'
import { useCandidates } from './alerts/useCandidates'
import { channels } from './alerts/severity'
import { acknowledge as popAlarm, discard, enqueue, type AlarmState } from './alerts/queue'
import { Shell } from './ui/Shell'
import { LivePage } from './pages/LivePage'
import { ReviewPage } from './pages/ReviewPage'

const REFRESH_MS = 5000

export interface Route {
  path: string
  params: URLSearchParams
}

function parseHash(): Route {
  const raw = productPath()
  const [path, q = ''] = raw.split('?')
  return { path: path || '/', params: new URLSearchParams(q) }
}

export interface PageProps {
  status: Status | null
  cameras: CameraInfo[]
  camera: string
  setCamera: (id: string) => void
  reviews: Candidate[]
  assessments: Assessment[]
  error: string
  busy: number | null
  decide: (id: number, verdict: ReviewVerdict, note?: string) => Promise<void>
  reload: () => Promise<void>
  events: Candidate[]
  arrivals: Candidate[]
  notices: Candidate[]
  unseen: number
  markSeen: (id: number) => void
  dismissNotice: (id: number) => void
  alarm: Candidate | null
  alarmQueued: number
  alarmShownAt: number
  ackAlarm: () => Promise<void>
  showCamera: (cam: string) => void
}

export function App() {
  const [route, setRoute] = useState(parseHash)
  const [statusState, setStatusState] = useState<Status | null>(null)
  const [cameras, setCameras] = useState<CameraInfo[]>([])
  const [camera, setCamera] = useState('')
  const [reviews, setReviews] = useState<Candidate[]>([])
  const [assessments, setAssessments] = useState<Assessment[]>([])
  const [error, setError] = useState('')
  const [busy, setBusy] = useState<number | null>(null)

  useEffect(() => onRouteChange(() => setRoute(parseHash())), [])

  const reload = useCallback(async () => {
    try {
      const [s, c, r, a] = await Promise.all([
        fetchStatus(),
        fetchCameras(),
        fetchCandidates({ limit: 50 }),
        fetchAssessments({ limit: 30 }),
      ])
      setStatusState(s)
      setCameras(c)
      setCamera(prev => (prev && c.some(cam => cam.id === prev) ? prev : (c[0]?.id ?? Object.keys(s.cameras ?? {})[0] ?? '')))
      setReviews(r)
      setAssessments(a)
      setError('')
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc))
    }
  }, [])

  useEffect(() => {
    void reload()
    const id = setInterval(() => void reload(), REFRESH_MS)
    return () => clearInterval(id)
  }, [reload])

  const decide = useCallback(async (id: number, verdict: ReviewVerdict, note?: string) => {
    setBusy(id)
    try {
      await review(id, verdict, { note })
      await reload()
      setAlarms(prev => discard(prev, id))
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc))
    } finally {
      setBusy(null)
    }
  }, [reload])

  const feed = useCandidates()

  const [alarms, setAlarms] = useState<AlarmState>({ current: null, queued: [] })
  const [shownAt, setShownAt] = useState(0)
  const carded = useRef<Set<number>>(new Set())

  useEffect(() => {
    const fresh = feed.arrivals.filter(e =>
      channels(e).card &&
      e.status === 'candidate' &&
      !carded.current.has(e.id))
    if (!fresh.length) return
    for (const e of fresh) carded.current.add(e.id)
    setAlarms(prev => [...fresh].reverse().reduce(enqueue, prev))
  }, [feed.arrivals])

  const currentId = alarms.current?.id ?? null

  useEffect(() => {
    if (currentId != null) setShownAt(Date.now())
  }, [currentId])

  const ackAlarm = useCallback(async () => {
    const current = alarms.current
    if (current == null) return
    try {
      await review(current.id, 'acknowledged')
      await reload()
      setAlarms(popAlarm)
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc))
    }
  }, [alarms, reload])

  const showCamera = useCallback((cam: string) => {
    setCamera(cam)
    navigate('/')
  }, [])

  const props: PageProps = {
    status: statusState,
    cameras,
    camera,
    setCamera,
    reviews,
    assessments,
    busy,
    decide,
    reload,
    error: error ? `the concealment service: ${error}` : feed.error ? `the candidates feed: ${feed.error}` : '',
    events: feed.events,
    arrivals: feed.arrivals,
    notices: feed.notices,
    unseen: feed.unseen,
    markSeen: feed.markSeen,
    dismissNotice: feed.dismissNotice,
    alarm: alarms.current,
    alarmQueued: alarms.queued.length,
    alarmShownAt: shownAt,
    ackAlarm,
    showCamera,
  }

  return (
    <Shell route={route} unseen={feed.unseen}>
      {route.path === '/review' ? <ReviewPage {...props} /> : <LivePage {...props} />}
    </Shell>
  )
}
