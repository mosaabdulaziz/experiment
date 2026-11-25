import { useEffect, useRef } from 'react'
import type { AttendanceRecord, MonitoringSnapshot } from './api'

const RECONNECT_MS = 3000

interface AttendanceSnapshotMsg {
  type: 'attendance_snapshot'
  day: string
  records: AttendanceRecord[]
}

/** Live attendance feed — adapted from iris-frontend's hook of the same name.
 * attendance-service sends one message shape only: a full snapshot, on
 * connect and whenever a sighting lands (or a leave flag is toggled). */
export function useAttendanceWebSocket(
  onSnapshot: (msg: AttendanceSnapshotMsg) => void,
  onMonitoring?: (msg: MonitoringSnapshot) => void,
) {
  const onSnapshotRef = useRef(onSnapshot)
  onSnapshotRef.current = onSnapshot
  const onMonitoringRef = useRef(onMonitoring)
  onMonitoringRef.current = onMonitoring

  useEffect(() => {
    let closed = false
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null
    let ws: WebSocket | null = null

    const connect = () => {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      ws = new WebSocket(`${proto}://${location.host}/attendance/ws`)
      ws.onmessage = e => {
        try {
          const msg = JSON.parse(e.data) as AttendanceSnapshotMsg | MonitoringSnapshot
          if (msg.type === 'attendance_snapshot') onSnapshotRef.current(msg)
          else if (msg.type === 'monitoring_snapshot') onMonitoringRef.current?.(msg)
        } catch {
          /* ignore malformed frames */
        }
      }
      ws.onclose = () => {
        if (!closed) reconnectTimer = setTimeout(connect, RECONNECT_MS)
      }
      ws.onerror = () => ws?.close()
    }

    connect()
    return () => {
      closed = true
      if (reconnectTimer) clearTimeout(reconnectTimer)
      ws?.close()
    }
  }, [])
}
