import { useCallback, useEffect, useMemo, useState } from 'react'
import { eventsApi } from '@/api/events'
import { useEventStore } from '@/stores/eventStore'
import type { IrisEvent } from '@/types/event'

const FETCH_LIMIT = 1000

/**
 * Every alert, newest first, seeded from the events API and kept current by the
 * live WebSocket store.
 *
 * There is no clip-listing endpoint and none is needed: event retention (48h)
 * strictly exceeds clip TTL (6h), so every clip that still exists necessarily still
 * has its event row. The store copy wins on merge, which is what makes a
 * `clip_status` transition arriving over the WebSocket reach the grid; updates for
 * fetched events the store does not hold (its `outside` patches) apply over them.
 */
export function useAlertIncidents() {
  const [fetched, setFetched] = useState<IrisEvent[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [reloadKey, setReloadKey] = useState(0)
  const live = useEventStore(s => s.events)
  const outside = useEventStore(s => s.outside)

  const reload = useCallback(() => setReloadKey(k => k + 1), [])

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    eventsApi
      .list(FETCH_LIMIT)
      .then(data => {
        if (!cancelled) setFetched(data)
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Could not load incidents')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [reloadKey])

  const incidents = useMemo(() => {
    const byId = new Map<number, IrisEvent>()
    for (const ev of fetched) byId.set(ev.id, outside[ev.id] ? { ...ev, ...outside[ev.id] } : ev)
    for (const ev of live) byId.set(ev.id, ev)
    return [...byId.values()]
      .filter(ev => ev.event_type === 'alert')
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
  }, [fetched, live, outside])

  return { incidents, loading, error, reload }
}
