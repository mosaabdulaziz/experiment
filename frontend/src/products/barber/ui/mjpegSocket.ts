/** Live video as JPEG frames over go2rtc's WebSocket API. An <img> per camera
 * holds an HTTP connection each and browsers allow 6 per host; WebSockets are
 * not bound by that. Open /go2rtc/api/ws?src=<stream>, send {"type":"mjpeg"},
 * and every binary message is one JPEG. Text messages are go2rtc's echo, or
 * {"type":"error"} when the camera can't be opened (the socket stays open):
 * that is `down`, as is a socket that sent no frame for STALE_MS. Down retries
 * after RETRY_MS. */
import { useEffect, useState } from 'react'
import { videoWsUrl } from '../api/client'

export type MjpegState = 'connecting' | 'live' | 'down'

const RETRY_MS = 4000
export const STALE_MS = 10000

export interface MjpegDeps {
  WebSocket: typeof WebSocket
  createObjectURL: (b: Blob) => string
  revokeObjectURL: (u: string) => void
  setTimeout: typeof setTimeout
  clearTimeout: typeof clearTimeout
  requestFrame: (cb: () => void) => number
  cancelFrame: (id: number) => void
}

const browserDeps = (): MjpegDeps => ({
  WebSocket,
  createObjectURL: b => URL.createObjectURL(b),
  revokeObjectURL: u => URL.revokeObjectURL(u),
  setTimeout: ((fn: () => void, ms: number) => setTimeout(fn, ms)) as typeof setTimeout,
  clearTimeout: (id => clearTimeout(id)) as typeof clearTimeout,
  requestFrame: cb => requestAnimationFrame(cb),
  cancelFrame: id => cancelAnimationFrame(id),
})

export class MjpegSocket {
  private ws: WebSocket | null = null
  private retry: ReturnType<typeof setTimeout> | null = null
  private watchdog: ReturnType<typeof setTimeout> | null = null
  private raf: number | null = null
  private pending: ArrayBuffer | null = null
  private current: string | null = null
  private disposed = false

  constructor(
    private url: string,
    private onFrame: (objectUrl: string) => void,
    private onState: (s: MjpegState) => void,
    private deps: MjpegDeps = browserDeps(),
  ) {
    this.connect()
  }

  private connect() {
    this.onState('connecting')
    const ws = new this.deps.WebSocket(this.url)
    this.ws = ws
    ws.binaryType = 'arraybuffer'
    ws.onopen = () => { if (this.ws === ws) ws.send(JSON.stringify({ type: 'mjpeg' })) }
    ws.onmessage = e => {
      if (this.disposed || this.ws !== ws) return
      if (typeof e.data === 'string') {
        if (isError(e.data)) down()
        return
      }
      this.pending = e.data as ArrayBuffer
      if (this.raf === null) this.raf = this.deps.requestFrame(() => this.paint())
      // armed per message, not per paint: a hidden tab paints nothing but its frames still arrive
      this.stopWatchdog()
      this.watchdog = this.deps.setTimeout(() => { this.watchdog = null; down() }, STALE_MS)
    }
    const down = () => {
      if (this.disposed || this.ws !== ws) return
      this.ws = null
      this.pending = null
      this.stopWatchdog()
      if (this.raf !== null) { this.deps.cancelFrame(this.raf); this.raf = null }
      this.onState('down')
      this.retry = this.deps.setTimeout(() => { this.retry = null; this.connect() }, RETRY_MS)
      try { ws.close() } catch { /* already closed */ }
    }
    ws.onclose = down
    ws.onerror = down
  }

  private stopWatchdog() {
    if (this.watchdog !== null) { this.deps.clearTimeout(this.watchdog); this.watchdog = null }
  }

  private paint() {
    this.raf = null
    if (this.disposed || !this.pending) return
    const url = this.deps.createObjectURL(new Blob([this.pending], { type: 'image/jpeg' }))
    this.pending = null
    const old = this.current
    this.current = url
    this.onFrame(url)
    this.onState('live')
    if (old) this.deps.revokeObjectURL(old)
  }

  dispose() {
    this.disposed = true
    if (this.retry !== null) { this.deps.clearTimeout(this.retry); this.retry = null }
    this.stopWatchdog()
    if (this.raf !== null) { this.deps.cancelFrame(this.raf); this.raf = null }
    const ws = this.ws
    this.ws = null
    if (ws) { try { ws.close() } catch { /* ignore */ } }
    if (this.current) { this.deps.revokeObjectURL(this.current); this.current = null }
  }
}

function isError(text: string): boolean {
  try {
    const m = JSON.parse(text)
    return typeof m === 'object' && m !== null && m.type === 'error'
  } catch {
    return false
  }
}

/** The latest frame (an object URL) and connection state of a go2rtc stream;
 * a null stream opens nothing. */
export function useMjpegSocket(stream: string | null): { frame: string | null; state: MjpegState } {
  const [frame, setFrame] = useState<string | null>(null)
  const [state, setState] = useState<MjpegState>('connecting')
  useEffect(() => {
    setFrame(null)
    if (!stream) return
    const s = new MjpegSocket(videoWsUrl(stream), setFrame, setState)
    return () => s.dispose()
  }, [stream])
  return { frame, state }
}
