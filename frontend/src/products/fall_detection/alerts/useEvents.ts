/** The events feed: what the product has seen, polled.
 *
 * A poll, not a socket: AC-F33.1 and AC-F34.1 freeze the tab out of sockets and the pub/sub
 * channel, and the platform's own alert path is what makes a fall urgent -- this is the
 * operator's surface, two seconds behind is invisible next to the sidecar's cadence.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { events as fetchEvents } from '../api'
import { channels, type FallEvent } from './severity'
import { unseenCount } from './badge'
import { advance } from './cursor'

const POLL_MS = 2000
const KEEP = 100
const NOTICE_MS = 6000

export interface EventsFeed {
  events: FallEvent[]
  arrivals: FallEvent[]
  unseen: number
  notices: FallEvent[]
  error: string
  markSeen: (id: number) => void
  dismissNotice: (id: number) => void
}

export function useEvents(): EventsFeed {
  const [events, setEvents] = useState<FallEvent[]>([])
  const [arrivals, setArrivals] = useState<FallEvent[]>([])
  const [notices, setNotices] = useState<FallEvent[]>([])
  const [seen, setSeen] = useState(0)
  const [error, setError] = useState('')
  const cursor = useRef(0)
  const inFlight = useRef(false)
  const primed = useRef(false)

  const poll = useCallback(async () => {
    if (inFlight.current) return
    inFlight.current = true
    try {
      const fresh = await fetchEvents(cursor.current)
      setError('')
      // The first response is the product's history, not news: it fills the list and sets the
      // badge's baseline, but nothing is announced and the alarm card stays quiet. An empty
      // first response primes too -- on a fresh install the first fall ever is news.
      if (!primed.current) {
        primed.current = true
        if (fresh.length) {
          const newest = advance(cursor.current, fresh)
          cursor.current = newest
          setEvents(prev => [...fresh, ...prev].slice(0, KEEP))
          setSeen(prev => Math.max(prev, newest))
        }
        return
      }
      if (!fresh.length) return
      const newest = advance(cursor.current, fresh)
      cursor.current = newest
      setEvents(prev => [...fresh, ...prev].slice(0, KEEP))
      setArrivals(prev => [...fresh, ...prev].slice(0, KEEP))
      const spoken = fresh.filter(e => channels(e).toast)
      if (spoken.length) setNotices(prev => [...spoken, ...prev].slice(0, 3))
    } catch (exc) {
      // The list keeps its last state, but not silently: a feed that has stopped answering
      // looks exactly like a quiet factory unless the operator is told. The next poll retries,
      // and a success clears this.
      setError(exc instanceof Error ? exc.message : String(exc))
    } finally {
      inFlight.current = false
    }
  }, [])

  useEffect(() => {
    void poll()
    const id = setInterval(() => void poll(), POLL_MS)
    return () => clearInterval(id)
  }, [poll])

  const markSeen = useCallback((id: number) => {
    setSeen(prev => Math.max(prev, id))
  }, [])

  const dismissNotice = useCallback((id: number) => {
    setNotices(prev => prev.filter(e => e.id !== id))
  }, [])

  // A notice is a moment, not a queue: it goes quiet on its own.
  useEffect(() => {
    if (!notices.length) return
    const id = setTimeout(() => setNotices(prev => prev.slice(0, -1)), NOTICE_MS)
    return () => clearTimeout(id)
  }, [notices])

  return useMemo(() => ({
    events, arrivals, notices, unseen: unseenCount(events, seen), error, markSeen, dismissNotice,
  }), [events, arrivals, notices, seen, error, markSeen, dismissNotice])
}
