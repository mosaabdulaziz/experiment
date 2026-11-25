import { useEffect, useRef } from 'react'
import type { WsMessage } from '@/types/ws'
import { useEventStore } from '@/stores/eventStore'
import { useModeStore } from '@/stores/modeStore'
import { eventsApi } from '@/api/events'
import { forLiveView } from '@/lib/proximity'

const RECONNECT_MS = 3000
const PING_MS = 25_000

export function useWebSocket() {
  const addEvent = useEventStore(s => s.addEvent)
  const updateEvent = useEventStore(s => s.updateEvent)
  const replaceEvent = useEventStore(s => s.replaceEvent)
  const setHistory = useEventStore(s => s.setHistory)
  const setStatus = useEventStore(s => s.setStatus)
  const clear = useEventStore(s => s.clear)
  const setMode = useModeStore(s => s.setMode)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    let pingTimer: ReturnType<typeof setInterval> | null = null
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null
    let closed = false
    let wsHistoryReceived = false

    eventsApi.list(100).then(events => {
      if (!wsHistoryReceived) {
        setHistory(forLiveView(events))
      }
    }).catch(() => {})

    const connect = () => {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      const ws = new WebSocket(`${proto}://${location.host}/ws`)
      wsRef.current = ws

      ws.onopen = () => {
        setStatus('idle')
        pingTimer = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) ws.send('ping')
        }, PING_MS)
      }

      ws.onmessage = e => {
        let msg: WsMessage
        try {
          msg = JSON.parse(e.data) as WsMessage
        } catch {
          return
        }
        switch (msg.type) {
          case 'history':
            wsHistoryReceived = true
            if (msg.mode) setMode(msg.mode)
            setHistory(forLiveView(msg.events))         // never a pair the depth check held back (Ticket 6)
            break
          case 'event':
            if (forLiveView([msg.event]).length) {
              addEvent(msg.event)
            }
            break
          case 'event_update': {
            // Two independent enrichment paths share this message type and carry
            // disjoint fields. Patching the union blindly would clear whichever
            // half this message isn't about — a resolving clip would wipe an
            // already-delivered description.
            // Incidents send the whole record, their clip list, or their status.
            if ('event' in msg) {
              replaceEvent(msg.event)
            } else if ('clip_status' in msg) {
              // a resolving clip carries the whole clip list too (an escalation clip may have come first)
              updateEvent(msg.id, {
                clip_status: msg.clip_status, clip_id: msg.clip_id ?? null, ...(msg.clips ? { clips: msg.clips } : {}),
              })
            } else if ('clips' in msg) {
              updateEvent(msg.id, { clips: msg.clips })
            } else if ('incident_status' in msg) {
              updateEvent(msg.id, { incident_status: msg.incident_status })
            } else if ('acknowledged_at' in msg) {
              // POST /api/events/{id}/acknowledge: before this branch it fell through to the description one and
              // wiped the event's description
              updateEvent(msg.id, { acknowledged_at: msg.acknowledged_at })
            } else {
              updateEvent(msg.id, { description_status: msg.description_status, description: msg.description })
            }
            break
          }
          case 'status':
            setStatus(msg.status)
            break
          case 'mode':
            setMode(msg.mode)
            break
          case 'error':
            setStatus('error')
            break
          case 'cleared':
            clear()
            break
          default: {
            const _exhaustive: never = msg
            void _exhaustive
          }
        }
      }

      ws.onclose = () => {
        if (pingTimer) clearInterval(pingTimer)
        setStatus('error')
        if (!closed) {
          reconnectTimer = setTimeout(connect, RECONNECT_MS)
        }
      }

      ws.onerror = () => ws.close()
    }

    connect()

    return () => {
      closed = true
      if (pingTimer) clearInterval(pingTimer)
      if (reconnectTimer) clearTimeout(reconnectTimer)
      wsRef.current?.close()
    }
  }, [addEvent, updateEvent, replaceEvent, setHistory, setStatus, clear, setMode])
}
