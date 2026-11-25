/** The tab: the hash route, the shell, and the service's own data.
 *
 * The product's service is polled, not streamed: AC-F33.1 and AC-F34.1 freeze the tab out of
 * sockets and the pub/sub channel, and the platform's own alert path is what makes the fall
 * urgent. This is the operator's surface.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { navigate, onRouteChange, productPath } from '../../shared/router'
import { cameras as fetchCameras, candidates, review, status, type Review, type Status } from './api'
import { useEvents } from './alerts/useEvents'
import { channels, type FallEvent } from './alerts/severity'
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

/** What every page takes: the service's own state. */
export interface PageProps {
  status: Status | null
  cameras: string[]
  camera: string
  setCamera: (id: string) => void
  reviews: Review[]
  error: string
  busy: number | null
  decide: (id: number, verdict: string) => Promise<void>
  reload: () => Promise<void>
  events: FallEvent[]
  arrivals: FallEvent[]
  notices: FallEvent[]
  unseen: number
  markSeen: (id: number) => void
  dismissNotice: (id: number) => void
  alarm: FallEvent | null
  alarmQueued: number
  alarmShownAt: number
  ackAlarm: () => Promise<void>
  showCamera: (cam: string) => void
}

export function App() {
  const [route, setRoute] = useState(parseHash)
  const [statusState, setStatusState] = useState<Status | null>(null)
  const [cameras, setCameras] = useState<string[]>([])
  const [camera, setCamera] = useState('')
  const [reviews, setReviews] = useState<Review[]>([])
  const [error, setError] = useState('')
  const [busy, setBusy] = useState<number | null>(null)

  useEffect(() => onRouteChange(() => setRoute(parseHash())), [])

  const reload = useCallback(async () => {
    try {
      const [s, c, r] = await Promise.all([status(), fetchCameras(), candidates()])
      setStatusState(s)
      setCameras(c)
      setCamera(prev => (prev && c.includes(prev) ? prev : (c[0] ?? '')))
      setReviews(r)
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

  const decide = useCallback(async (id: number, verdict: string) => {
    setBusy(id)
    try {
      await review(id, verdict)
      await reload()
      setAlarms(prev => discard(prev, id))
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc))
    } finally {
      setBusy(null)
    }
  }, [reload])

  const feed = useEvents()

  const [alarms, setAlarms] = useState<AlarmState>({ current: null, queued: [] })
  const [shownAt, setShownAt] = useState(0)
  const carded = useRef<Set<number>>(new Set())

  // A card-worthy event joins the queue the moment it arrives -- and only once.
  useEffect(() => {
    const fresh = feed.arrivals.filter(e =>
      channels(e).card &&
      (e.review_status == null || e.review_status === 'candidate') &&
      !carded.current.has(e.id))
    if (!fresh.length) return
    for (const e of fresh) carded.current.add(e.id)
    setAlarms(prev => [...fresh].reverse().reduce(enqueue, prev))
  }, [feed.arrivals])

  const currentId = alarms.current?.id ?? null

  // Every card arms on its own arrival on screen, not on the queue behind it (R9).
  useEffect(() => {
    if (currentId != null) setShownAt(Date.now())
  }, [currentId])

  // The card leaves only after the verdict is persisted: an ack the service refused must face
  // the operator with the card still up (the card is the remedy), not vanish behind the error bar.
  const ackAlarm = useCallback(async () => {
    const current = alarms.current
    if (current == null) return
    if (current.review_id == null) {
      setAlarms(popAlarm)
      return
    }
    try {
      await review(current.review_id, 'acknowledged')
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
    status: statusState, cameras, camera, setCamera, reviews, busy, decide, reload,
    // One bar, in the pages' words: a dead service outranks a dead feed, since it explains the feed too.
    error: error ? `the fall service: ${error}` : feed.error ? `the events feed: ${feed.error}` : '',
    events: feed.events, arrivals: feed.arrivals, notices: feed.notices, unseen: feed.unseen,
    markSeen: feed.markSeen, dismissNotice: feed.dismissNotice,
    alarm: alarms.current, alarmQueued: alarms.queued.length, alarmShownAt: shownAt,
    ackAlarm, showCamera,
  }

  return (
    <Shell route={route} unseen={feed.unseen}>
      {route.path === '/review' ? <ReviewPage {...props} /> : <LivePage {...props} />}
    </Shell>
  )
}
