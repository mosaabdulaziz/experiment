import { useEffect, useMemo, useRef, useState } from 'react'
import { api, eventsApi, type CameraInfo, type ClipRef, type IrisEvent } from './api'
import { camIsLive } from './components'

const FETCH_LIMIT = 500
const RECONNECT_MS = 3000

/** Fire/smoke class, read off the message text -- the alert stream only
 * carries a "fire_smoke" tag (see app/stream_manager.py's count_alert_message),
 * not a separate per-class tag. Message is always "<Subject> detected on
 * <camera>" with Subject in {"Fire", "Smoke", "Fire or smoke"}. */
export type FireKind = 'fire' | 'smoke' | 'unknown'

export function fireKind(ev: IrisEvent): FireKind {
  if (ev.message.startsWith('Fire ')) return 'fire'
  if (ev.message.startsWith('Smoke ')) return 'smoke'
  return 'unknown'
}

/** Every fire/smoke alert, newest first: seeded from /api/events and kept
 * current by iris's own /ws socket (history + event + event_update), the
 * same shape iris-frontend's incidents page consumes. Adapted here without a
 * global store since this app has exactly one consumer. */
export function useFireAlerts() {
  const [fetched, setFetched] = useState<IrisEvent[]>([])
  const [live, setLive] = useState<Map<number, IrisEvent>>(new Map())
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [connected, setConnected] = useState(false)
  const [reloadKey, setReloadKey] = useState(0)
  const liveRef = useRef(live)
  liveRef.current = live

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    eventsApi.list(FETCH_LIMIT)
      .then(data => { if (!cancelled) setFetched(data) })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : 'تعذر تحميل التنبيهات')
      })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [reloadKey])

  useEffect(() => {
    let closed = false
    let timer: ReturnType<typeof setTimeout> | null = null
    let ws: WebSocket | null = null

    const connect = () => {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      ws = new WebSocket(`${proto}://${location.host}/ws`)
      ws.onopen = () => setConnected(true)
      ws.onmessage = e => {
        try {
          // Not a strict discriminated union: TS can't narrow away a generic
          // `{ type: string }` fallback member (every literal is assignable to
          // it), so this stays a loose "any known field might be present" shape
          // and each branch below reaches into only the fields it expects.
          const msg = JSON.parse(e.data) as {
            type: string
            events?: IrisEvent[]
            event?: IrisEvent
            id?: number
            [k: string]: unknown
          }
          if (msg.type === 'history' && msg.events) {
            const events = msg.events
            setLive(prev => {
              const next = new Map(prev)
              for (const ev of events) next.set(ev.id, ev)
              return next
            })
          } else if (msg.type === 'event' && msg.event) {
            const ev = msg.event
            setLive(prev => new Map(prev).set(ev.id, ev))
          } else if (msg.type === 'event_update' && typeof msg.id === 'number') {
            const id = msg.id
            const existing = liveRef.current.get(id) ?? fetched.find(ev => ev.id === id)
            if (!existing) return
            const { type: _type, id: _id, ...patch } = msg
            setLive(prev => new Map(prev).set(id, { ...existing, ...patch }))
          } else if (msg.type === 'cleared' && (msg.tag === undefined || msg.tag === 'fire_smoke')) {
            // Either a global clear (no tag -- iris-frontend's own "clear
            // everything" button) or our own scoped fire_smoke clear: both
            // leave nothing fire_smoke-tagged behind, so drop everything
            // this hook is holding. A clear scoped to some OTHER tag leaves
            // fire/smoke incidents untouched and is ignored here.
            setFetched([])
            setLive(new Map())
          }
        } catch {
          /* ignore malformed frames */
        }
      }
      ws.onclose = () => {
        setConnected(false)
        if (!closed) timer = setTimeout(connect, RECONNECT_MS)
      }
      ws.onerror = () => ws?.close()
    }

    connect()
    return () => {
      closed = true
      if (timer) clearTimeout(timer)
      ws?.close()
    }
    // fetched is read via liveRef/closure fallback only to patch an event_update
    // that arrives before the initial fetch resolves; re-subscribing per fetch
    // would just thrash the socket for no benefit.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const alerts = useMemo(() => {
    const byId = new Map<number, IrisEvent>()
    for (const ev of fetched) byId.set(ev.id, ev)
    for (const ev of live.values()) byId.set(ev.id, ev)
    return [...byId.values()]
      .filter(ev => ev.event_type === 'alert' && ev.tags?.includes('fire_smoke'))
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
  }, [fetched, live])

  return {
    alerts, loading, error, connected, reload: () => setReloadKey(k => k + 1),
    // The one write action the UI offers. No optimistic update: the WS
    // event_update this triggers (same broadcast every other incident-state
    // change already uses) lands within milliseconds and is the single
    // source of truth every consumer already reconciles against.
    acknowledge: (id: number) => eventsApi.acknowledge(id),
    // Fire/Smoke page "Clear" button (2026-09-06): scoped to fire_smoke only
    // (see eventsApi.clearByTag), never the whole event log. Optimistic --
    // unlike acknowledge, an operator clearing the rail wants it empty
    // immediately, not after a round trip; the "cleared" WS broadcast (see
    // above) is what keeps every OTHER open tab/browser in sync.
    clear: async () => {
      await eventsApi.clearByTag('fire_smoke')
      setFetched([])
      setLive(new Map())
    },
  }
}

/** Every fire-role camera that's ACTUALLY streaming right now, polled at the
 * same cadence CamerasPage uses for its own live list. Shared by the live
 * wall and the takeover overlay's "other streams" strip.
 *
 * 2026-09-06: sidecar.yaml can declare a fire-role camera (e.g. cam-8) that
 * simply isn't part of what's currently running (only cam-1/cam-7 streamed
 * this session) -- the sidecar supports every camera it's configured for,
 * but this hook's job is to hand back only the names actually worth
 * showing, so the gate is camIsLive (offset > 0), not just the role tag. A
 * camera that stops streaming disappears from both places entirely instead
 * of lingering as a "غير متصلة" placeholder. */
export function useFireCameras(): CameraInfo[] {
  const [cameras, setCameras] = useState<CameraInfo[]>([])
  useEffect(() => {
    let cancelled = false
    const tick = () => {
      api.listCameras().then(c => {
        if (!cancelled) setCameras(c.filter(cam => cam.roles.includes('fire') && camIsLive(cam)))
      }).catch(() => {})
    }
    tick()
    const id = setInterval(tick, 2000)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [])
  return cameras
}

export const KIND_LABEL: Record<FireKind, string> = {
  fire: 'حريق',
  smoke: 'دخان',
  unknown: 'حريق/دخان',
}

/** Every clip attached to an incident, oldest first -- clips[0] is always the
 * same clip clip_id/clip_status point at (see IrisEvent). Falls back to a
 * synthetic one-clip list built from clip_id/clip_status for an event saved
 * before the `clips` column existed, so a pre-migration row (or a non-fire
 * alert, which never gets more than one clip) still shows its one clip
 * instead of nothing -- the fallback the multi-clip filmstrip is built to
 * preserve. */
export function effectiveClips(ev: IrisEvent): ClipRef[] {
  if (ev.clips.length > 0) return ev.clips
  if (ev.clip_id && ev.clip_status === 'ready') return [{ clip_id: ev.clip_id, status: 'ready' }]
  return []
}

export function fmtDateTime(iso: string): string {
  return new Date(iso).toLocaleString('en-GB', {
    day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit',
  })
}

export function fmtDuration(seconds: number): string {
  const s = Math.max(0, Math.round(seconds))
  const m = Math.floor(s / 60)
  const r = s % 60
  return m > 0 ? `${m}د ${r}ث` : `${r}ث`
}

/** started_at, if present, is authoritative over the record's own save-time
 * timestamp -- it is the true onset (see FireIncidentEvent.onset_age), while
 * timestamp is merely when the alert row was created a few seconds later. */
export function onsetOf(ev: IrisEvent): string {
  return ev.incident_started_at ?? ev.timestamp
}

/** Camera display name: sidecar.yaml's optional label, falling back to the
 * frontend's own CAM_LABELS map, falling back to the raw id. */
export function camLabel(cam: CameraInfo, camLabels: Record<string, string>): string {
  return cam.label ?? camLabels[cam.id] ?? cam.id
}

/** Re-renders every `intervalMs`, returning Date.now() each time -- used to tick a
 * live "ongoing for Xm Ys" duration. Pass null to disable (no incidents ongoing,
 * no point re-rendering on a timer with nothing changing). */
export function useNowTick(intervalMs: number | null): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (intervalMs == null) return
    const id = setInterval(() => setNow(Date.now()), intervalMs)
    return () => clearInterval(id)
  }, [intervalMs])
  return now
}
