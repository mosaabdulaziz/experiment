/** Hooks that keep the console's data current: the live event feed (REST, then /ws), and polled resources. */
import { useCallback, useEffect, useRef, useState } from 'react'
import { FramePoller } from '@/components/streams/framePoll'
import { api } from './api'
import { alertsOnly } from './lib/incidents'
import { applyMessage, resync, type FeedMessage } from './lib/feed'
import type { SafetyCamera, SafetyEvent, SafetyZone } from './lib/types'

/** The recent alerts, newest first, kept current by the app's /ws broadcast (reconnects on its own). */
export function useFeed(): { events: SafetyEvent[]; connected: boolean; patch: (id: number, p: Partial<SafetyEvent>) => void } {
  const [byId, setById] = useState<Map<number, SafetyEvent>>(new Map())
  const [connected, setConnected] = useState(false)

  useEffect(() => {
    let socket: WebSocket | null = null
    let retry: ReturnType<typeof setTimeout> | null = null
    let ping: ReturnType<typeof setInterval> | null = null
    let stopped = false
    let opened = false
    let loaded = false
    // the list, then the socket's changes to it; reloaded on every reconnect, since nothing tells a console that was
    // away what changed meanwhile (an app restart closes its open incidents with nobody connected), and on the
    // socket's first open too when the first load failed
    const load = () => api.events()
      .then(list => { if (!stopped) { loaded = true; setById(prev => resync(prev, list)) } })
      .catch(() => { /* the next open of the socket loads again */ })
    load()
    const connect = () => {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      socket = new WebSocket(`${proto}://${location.host}/ws`)
      socket.onopen = () => {
        setConnected(true)
        if (opened || !loaded) load()
        opened = true
        ping = setInterval(() => socket?.readyState === WebSocket.OPEN && socket.send('ping'), 25000)
      }
      socket.onmessage = m => {
        try {
          const msg = JSON.parse(m.data) as FeedMessage
          if (msg.type === 'history') return          // load() reads more of it than the socket sends
          setById(prev => applyMessage(prev, msg))
        } catch { /* not JSON (a pong): ignore */ }
      }
      socket.onclose = () => {
        setConnected(false)
        if (ping) clearInterval(ping)
        if (!stopped) retry = setTimeout(connect, 3000)
      }
    }
    connect()
    return () => {
      stopped = true
      if (retry) clearTimeout(retry)
      if (ping) clearInterval(ping)
      socket?.close()
    }
  }, [])

  const patch = useCallback((id: number, p: Partial<SafetyEvent>) => {
    setById(prev => {
      const cur = prev.get(id)
      return cur ? new Map(prev).set(id, { ...cur, ...p }) : prev
    })
  }, [])
  const events = alertsOnly([...byId.values()]).sort((a, b) => b.id - a.id)
  return { events, connected, patch }
}

/** `load()` now and every `intervalMs` (0: once); `reload()` on demand. Keeps the last good data on an error. */
export function usePoll<T>(load: () => Promise<T>, intervalMs = 0): {
  data: T | null; error: string | null; reload: () => void
} {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const loadRef = useRef(load)
  loadRef.current = load
  const [tick, setTick] = useState(0)
  const reload = useCallback(() => setTick(t => t + 1), [])
  useEffect(() => {
    let alive = true
    const run = () => loadRef.current()
      .then(d => { if (alive) { setData(d); setError(null) } })
      .catch(e => { if (alive) setError(e instanceof Error ? e.message : String(e)) })
    run()
    const timer = intervalMs ? setInterval(run, intervalMs) : null
    return () => { alive = false; if (timer) clearInterval(timer) }
  }, [intervalMs, tick])
  return { data, error, reload }
}

/** The <img> props for a camera's still (go2rtc, api.ts stillUrl), the next asked for `afterLoadMs` after the last
 *  one loads: a camera no feature runs on, seen every few seconds rather than live. */
export function useStill(url: string, afterLoadMs = 3000, afterErrorMs = 5000) {
  const [seq, setSeq] = useState(0)
  const [poller] = useState(() => new FramePoller(afterLoadMs, afterErrorMs, setSeq))
  useEffect(() => {
    poller.start()
    return () => poller.stop()
  }, [poller])
  return { src: `${url}?t=${seq}`, onLoad: () => poller.loaded(), onError: () => poller.failed() }
}

/** Every camera's saved zones, by camera id: loaded again when the cameras or their features change, when the zone
 *  editor closes (`version`), and every minute. A camera whose zones did not load is left out. */
export function useSiteZones(cameras: SafetyCamera[] | null, version: number): Record<string, SafetyZone[]> {
  const [zones, setZones] = useState<Record<string, SafetyZone[]>>({})
  const [tick, setTick] = useState(0)
  const key = (cameras ?? []).map(c => `${c.id}:${c.features.join('+')}`).join(',')
  useEffect(() => {
    const t = setInterval(() => setTick(n => n + 1), 60000)
    return () => clearInterval(t)
  }, [])
  useEffect(() => {
    let alive = true
    const ids = key ? key.split(',').map(k => k.split(':')[0]) : []
    Promise.all(ids.map(id => api.zones(id).then(z => [id, z] as const).catch(() => null)))
      .then(pairs => { if (alive) setZones(Object.fromEntries(pairs.filter(p => p !== null))) })
    return () => { alive = false }
  }, [key, version, tick])
  return zones
}

