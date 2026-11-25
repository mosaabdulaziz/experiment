import { afterEach, describe, expect, it, vi } from 'vitest'
import { MjpegSocket, STALE_MS } from './mjpegSocket'
import { videoWsUrl } from '../api/client'

class FakeWS {
  static all: FakeWS[] = []
  binaryType = ''
  sent: string[] = []
  closed = false
  onopen: (() => void) | null = null
  onmessage: ((e: { data: unknown }) => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  constructor(public url: string) { FakeWS.all.push(this) }
  send(m: string) { this.sent.push(m) }
  close() { this.closed = true }
}

function setup() {
  FakeWS.all = []
  let n = 0
  const timers: { id: number; ms: number; fn: () => void }[] = []
  let frameCb: (() => void) | null = null
  const revoked: string[] = []
  const frames: string[] = []
  const states: string[] = []
  const deps = {
    WebSocket: FakeWS as unknown as typeof WebSocket,
    createObjectURL: () => `blob:${++n}`,
    revokeObjectURL: (u: string) => { revoked.push(u) },
    setTimeout: ((fn: () => void, ms: number) => { timers.push({ id: timers.length + 1, ms, fn }); return timers.length }) as unknown as typeof setTimeout,
    clearTimeout: ((id: number) => { const i = timers.findIndex(t => t.id === id); if (i >= 0) timers.splice(i, 1) }) as unknown as typeof clearTimeout,
    requestFrame: (cb: () => void) => { frameCb = cb; return 1 },
    cancelFrame: () => { frameCb = null },
  }
  const sock = new MjpegSocket('ws://h/x', u => frames.push(u), s => states.push(s), deps)
  const paint = () => { const cb = frameCb; frameCb = null; cb?.() }
  return { sock, timers, revoked, frames, states, paint, hasFrame: () => frameCb !== null }
}
const bin = () => new ArrayBuffer(4)

afterEach(() => vi.unstubAllGlobals())

describe('MjpegSocket', () => {
  it('opens the url and asks for mjpeg on open', () => {
    const s = setup()
    const ws = FakeWS.all[0]
    expect(ws.url).toBe('ws://h/x')
    expect(ws.binaryType).toBe('arraybuffer')
    ws.onopen!()
    expect(ws.sent).toEqual(['{"type":"mjpeg"}'])
    expect(s.states).toEqual(['connecting'])
  })

  it('applies a binary message once per animation frame and revokes the old url', () => {
    const s = setup()
    const ws = FakeWS.all[0]
    ws.onopen!()
    ws.onmessage!({ data: bin() })
    expect(s.frames).toEqual([])
    s.paint()
    expect(s.frames).toEqual(['blob:1'])
    expect(s.states[s.states.length - 1]).toBe('live')
    ws.onmessage!({ data: bin() })
    s.paint()
    expect(s.frames).toEqual(['blob:1', 'blob:2'])
    expect(s.revoked).toEqual(['blob:1'])
  })

  it('coalesces a burst into one frame', () => {
    const s = setup()
    const ws = FakeWS.all[0]
    for (let i = 0; i < 5; i++) ws.onmessage!({ data: bin() })
    s.paint()
    expect(s.frames).toHaveLength(1)
  })

  it('ignores text messages', () => {
    const s = setup()
    FakeWS.all[0].onmessage!({ data: '{"type":"mjpeg"}' })
    expect(s.hasFrame()).toBe(false)
    expect(s.frames).toEqual([])
  })

  it("go2rtc's error text is down and retries; the socket is closed", () => {
    const s = setup()
    const ws = FakeWS.all[0]
    ws.onopen!()
    ws.onmessage!({ data: '{"type":"error","value":"streams: wrong user/pass"}' })
    expect(s.states[s.states.length - 1]).toBe('down')
    expect(ws.closed).toBe(true)
    ws.onclose!()                                               // the close that follows: no second retry
    expect(s.timers.map(t => t.ms)).toEqual([4000])
    s.timers[0].fn()
    expect(FakeWS.all).toHaveLength(2)
  })

  it('other text, even unparsable, is not an error', () => {
    const s = setup()
    FakeWS.all[0].onmessage!({ data: 'not json' })
    FakeWS.all[0].onmessage!({ data: '{"type":"mjpeg"}' })
    expect(s.states).toEqual(['connecting'])
  })

  it(`no frame for ${STALE_MS} ms is down and reconnects; every frame re-arms it`, () => {
    const s = setup()
    const ws = FakeWS.all[0]
    ws.onmessage!({ data: bin() })
    s.paint()
    ws.onmessage!({ data: bin() })
    expect(s.timers.map(t => t.ms)).toEqual([STALE_MS])         // one watchdog, re-armed
    s.timers[0].fn()
    expect(s.states[s.states.length - 1]).toBe('down')
    expect(ws.closed).toBe(true)
    expect(s.timers.map(t => t.ms)).toEqual([STALE_MS, 4000])   // the fired one stays in the fake's list
    s.timers[1].fn()
    expect(FakeWS.all).toHaveLength(2)
  })

  it('a close clears the watchdog', () => {
    const s = setup()
    FakeWS.all[0].onmessage!({ data: bin() })
    FakeWS.all[0].onclose!()
    expect(s.timers.map(t => t.ms)).toEqual([4000])
  })

  it('goes down on close, reconnects after 4000 ms, frames flow again', () => {
    const s = setup()
    FakeWS.all[0].onclose!()
    expect(s.states[s.states.length - 1]).toBe('down')
    expect(s.timers.map(t => t.ms)).toEqual([4000])
    s.timers[0].fn()
    expect(FakeWS.all).toHaveLength(2)
    FakeWS.all[1].onopen!()
    FakeWS.all[1].onmessage!({ data: bin() })
    s.paint()
    expect(s.frames).toHaveLength(1)
    expect(s.states[s.states.length - 1]).toBe('live')
  })

  it('error then close schedules a single retry', () => {
    const s = setup()
    FakeWS.all[0].onerror!()
    FakeWS.all[0].onclose!()
    expect(s.timers).toHaveLength(1)
  })

  it('dispose closes, revokes, cancels retry and goes silent', () => {
    const s = setup()
    const ws = FakeWS.all[0]
    ws.onmessage!({ data: bin() })
    s.paint()
    ws.onmessage!({ data: bin() })
    const n = s.frames.length, st = s.states.length
    s.sock.dispose()
    expect(ws.closed).toBe(true)
    expect(s.revoked).toContain('blob:1')
    expect(s.hasFrame()).toBe(false)
    ws.onclose?.()
    ws.onmessage?.({ data: bin() })
    expect(s.timers).toHaveLength(0)
    expect(FakeWS.all).toHaveLength(1)
    expect(s.frames).toHaveLength(n)
    expect(s.states).toHaveLength(st)
  })

  it('dispose cancels a pending retry', () => {
    const s = setup()
    FakeWS.all[0].onclose!()
    s.sock.dispose()
    expect(s.timers).toHaveLength(0)
  })
})

describe('videoWsUrl', () => {
  it('uses ws on http and wss on https', () => {
    vi.stubGlobal('location', { protocol: 'http:', host: 'a:5190' })
    expect(videoWsUrl('ph-1')).toBe('ws://a:5190/go2rtc/api/ws?src=ph-1')
    vi.stubGlobal('location', { protocol: 'https:', host: 'b' })
    expect(videoWsUrl('ph 1')).toBe('wss://b/go2rtc/api/ws?src=ph%201')
  })
})
