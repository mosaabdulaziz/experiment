/** Live camera frames for the wall and the expanded view: one /stream/<id>/frame request at a time, the next asked for a
 * moment after the previous one loads (longer after it fails), so a wall of tiles stays within the browser's 6
 * connections per host. React starts the poller in an effect and stops it in the cleanup. A frame that loads while
 * the poller is stopped is remembered, and the next start moves on from it: StrictMode stops and restarts each new
 * tile once, after a frame from the browser's cache may already have loaded, and a stream can go off and come back.
 * Without that, the tile kept its first frame for good (D18/D41 froze when the wall's columns changed). */
import { useEffect, useState } from 'react'

export interface TimerDeps {
  setTimeout: typeof setTimeout
  clearTimeout: typeof clearTimeout
}

const browserTimers = (): TimerDeps => ({
  setTimeout: ((fn: () => void, ms: number) => setTimeout(fn, ms)) as typeof setTimeout,
  clearTimeout: (id => clearTimeout(id)) as typeof clearTimeout,
})

export class FramePoller {
  private seq = 0
  private timer: ReturnType<typeof setTimeout> | null = null
  private running = false
  private waitMs: number | null = null   // set once the current frame loaded or failed: the wait before the next

  constructor(
    private afterLoadMs: number,
    private afterErrorMs: number,
    private onNext: (seq: number) => void,
    private deps: TimerDeps = browserTimers(),
  ) {}

  start() {
    this.running = true
    this.arm()
  }

  stop() {
    this.running = false
    if (this.timer !== null) this.deps.clearTimeout(this.timer)
    this.timer = null
  }

  loaded() {
    this.waitMs = this.afterLoadMs
    this.arm()
  }

  failed() {
    this.waitMs = this.afterErrorMs
    this.arm()
  }

  private arm() {
    if (!this.running || this.waitMs === null || this.timer !== null) return
    this.timer = this.deps.setTimeout(() => {
      this.timer = null
      this.waitMs = null
      this.onNext(++this.seq)
    }, this.waitMs)
  }
}

let mounts = 0

/** The <img> props for one camera's live frames. Each mount asks for its own URLs, so a tile that remounts (it moved
 * to another row) fetches a fresh frame instead of the one an earlier tile loaded under the same URL. */
export function useFramePoll(streamId: string, running: boolean, afterLoadMs: number, afterErrorMs = 2000) {
  const [mount] = useState(() => ++mounts)
  const [seq, setSeq] = useState(0)
  const [poller] = useState(() => new FramePoller(afterLoadMs, afterErrorMs, setSeq))
  useEffect(() => {
    if (!running) return
    poller.start()
    return () => poller.stop()
  }, [poller, running])
  return {
    src: `/stream/${streamId}/frame?t=${mount}-${seq}`,
    onLoad: () => poller.loaded(),
    onError: () => poller.failed(),
  }
}
