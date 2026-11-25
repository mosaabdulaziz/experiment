import { useCallback, useEffect, useRef, useState } from 'react'
import { candidates as fetchCandidates, type Candidate } from '../api'
import { unseenCount } from './badge'
import { advance } from './cursor'
import { channels } from './severity'

const POLL_MS = 2000

export type CandidateFeed = {
  events: Candidate[]
  arrivals: Candidate[]
  notices: Candidate[]
  unseen: number
  error: string
  markSeen: (id: number) => void
  dismissNotice: (id: number) => void
}

export function useCandidates(): CandidateFeed {
  const cursor = useRef(0)
  const primed = useRef(false)
  const [list, setList] = useState<Candidate[]>([])
  const [arrivals, setArrivals] = useState<Candidate[]>([])
  const [notices, setNotices] = useState<Candidate[]>([])
  const [seen, setSeen] = useState(0)
  const [error, setError] = useState('')

  useEffect(() => {
    let stopped = false
    const tick = async () => {
      try {
        const batch = await fetchCandidates({ since: cursor.current, limit: 50 })
        if (stopped) return
        const next = advance(cursor.current, batch)
        if (!primed.current) {
          primed.current = true
          cursor.current = next
          setSeen(next)
          setList(batch)
          setArrivals([])
        } else if (batch.length) {
          cursor.current = next
          setList(prev => {
            const ids = new Set(batch.map(b => b.id))
            return [...batch, ...prev.filter(p => !ids.has(p.id))].slice(0, 100)
          })
          setArrivals(batch)
          const toasts = batch.filter(e => channels(e).toast)
          if (toasts.length) setNotices(prev => [...toasts, ...prev].slice(0, 5))
        } else {
          setArrivals([])
        }
        setError('')
      } catch (exc) {
        if (!stopped) setError(exc instanceof Error ? exc.message : String(exc))
      }
    }
    void tick()
    const id = setInterval(() => void tick(), POLL_MS)
    return () => {
      stopped = true
      clearInterval(id)
    }
  }, [])

  const markSeen = useCallback((id: number) => {
    setSeen(prev => (id > prev ? id : prev))
  }, [])

  const dismissNotice = useCallback((id: number) => {
    setNotices(prev => prev.filter(n => n.id !== id))
  }, [])

  return {
    events: list,
    arrivals,
    notices,
    unseen: unseenCount(list, seen),
    error,
    markSeen,
    dismissNotice,
  }
}
