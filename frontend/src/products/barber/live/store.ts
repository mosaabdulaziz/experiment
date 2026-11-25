import { useEffect, useRef, useState, useSyncExternalStore } from 'react'
import type { ChairOcc, LiveSnapshot, Scene } from '../api/types'

const RECONNECT_MS = 3000

/** A tiny external store. The person boxes change several times a second;
 * keeping them out of React state at the app root means only the overlay
 * that draws them re-renders on each update, not every page and chart. */
export interface Store<T> {
  get: () => T
  subscribe: (listener: () => void) => () => void
}

export function createStore<T>(initial: T): Store<T> & { set: (v: T) => void } {
  let value = initial
  const listeners = new Set<() => void>()
  return {
    get: () => value,
    set: v => {
      if (v === value) return
      value = v
      listeners.forEach(l => l())
    },
    subscribe: l => { listeners.add(l); return () => { listeners.delete(l) } },
  }
}

export function useStore<T>(store: Store<T>): T {
  return useSyncExternalStore(store.subscribe, store.get)
}

export type ConnState = 'connecting' | 'connected' | 'reconnecting'
export interface LiveState { snap: LiveSnapshot | null; conn: ConnState; offsetMs: number; scene: Scene; chairs: Record<string, ChairOcc> }
export const initialLive: LiveState = { snap: null, conn: 'connecting', offsetMs: 0, scene: {}, chairs: {} }
type SceneMsg = { type: 'scene'; scene: Scene; chairs?: Record<string, ChairOcc> }
export type LiveEvent = { kind: 'message'; msg: LiveSnapshot | SceneMsg } | { kind: 'open' } | { kind: 'closed' }

/** The socket's messages as pure state transitions:
 *  - `live_snapshot`: everything, on connect, on every change and every 5 s;
 *    it re-syncs the server-clock offset;
 *  - `scene`: just the person boxes (and chair occupancy), up to scene_fps
 *    times a second, so they follow the video between snapshots;
 *  - a closed socket keeps the last data on screen, marked reconnecting. */
export function reduce(s: LiveState, e: LiveEvent, browserNowMs: number): LiveState {
  if (e.kind === 'open') return { ...s, conn: 'connected' }
  if (e.kind === 'closed') return { ...s, conn: 'reconnecting' }
  const m = e.msg
  if (m.type === 'live_snapshot') {
    return { snap: m, conn: 'connected', offsetMs: Date.parse(m.server_time) - browserNowMs,
             scene: m.scene ?? {}, chairs: m.chairs ?? {} }
  }
  if (m.type === 'scene') return { ...s, scene: m.scene ?? {}, chairs: m.chairs ?? s.chairs }
  return s
}

export interface Live {
  snap: LiveSnapshot | null
  conn: ConnState
  /** The server's clock (ms), ticking once a second while connected. It
   * stops while the socket is down, so every timer visibly freezes instead
   * of counting on stale data. */
  now: number
  /** Person boxes per camera, at scene rate. */
  scene: Store<Scene>
  /** Each chair's occupancy, refreshed with the scene. */
  chairs: Store<Record<string, ChairOcc>>
}

/** The one live feed for the whole app (socket /barber/ws). */
export function useLive(): Live {
  const [snap, setSnap] = useState<LiveSnapshot | null>(null)
  const [conn, setConn] = useState<ConnState>('connecting')
  const [now, setNow] = useState(() => Date.now())
  const state = useRef(initialLive)
  const scene = useRef(createStore<Scene>({})).current
  const chairs = useRef(createStore<Record<string, ChairOcc>>({})).current

  useEffect(() => {
    let closed = false
    let timer: ReturnType<typeof setTimeout> | null = null
    let ws: WebSocket | null = null
    const apply = (e: LiveEvent) => {
      const prev = state.current
      const next = reduce(prev, e, Date.now())
      if (next === prev) return
      state.current = next
      setConn(next.conn)
      setSnap(next.snap)
      scene.set(next.scene)
      chairs.set(next.chairs)
      if (next.snap !== prev.snap) setNow(Date.now() + next.offsetMs)
    }
    const connect = () => {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      ws = new WebSocket(`${proto}://${location.host}/barber/ws`)
      ws.onopen = () => apply({ kind: 'open' })
      ws.onmessage = e => {
        let msg: LiveSnapshot | SceneMsg
        try { msg = JSON.parse(e.data) } catch { return /* ignore malformed frames */ }
        if (msg && typeof msg === 'object') apply({ kind: 'message', msg })
      }
      ws.onclose = () => {
        if (closed) return
        apply({ kind: 'closed' })
        timer = setTimeout(connect, RECONNECT_MS)
      }
      ws.onerror = () => ws?.close()
    }
    connect()
    return () => {
      closed = true
      if (timer) clearTimeout(timer)
      ws?.close()
    }
  }, [scene, chairs])

  useEffect(() => {
    if (conn !== 'connected') return
    const id = setInterval(() => setNow(Date.now() + state.current.offsetMs), 1000)
    return () => clearInterval(id)
  }, [conn])

  return { snap, conn, now, scene, chairs }
}
