import { describe, expect, it } from 'vitest'
import { FramePoller } from './framePoll'

function setup(afterLoadMs = 100, afterErrorMs = 2000) {
  const timers: { id: number; ms: number; fn: () => void }[] = []
  let n = 0
  const deps = {
    setTimeout: ((fn: () => void, ms: number) => { timers.push({ id: ++n, ms, fn }); return n }) as unknown as typeof setTimeout,
    clearTimeout: ((id: number) => { const i = timers.findIndex(t => t.id === id); if (i >= 0) timers.splice(i, 1) }) as unknown as typeof clearTimeout,
  }
  const asked: number[] = []
  const poller = new FramePoller(afterLoadMs, afterErrorMs, seq => asked.push(seq), deps)
  const fire = () => { const t = timers.shift(); t?.fn() }
  return { poller, timers, asked, fire }
}

describe('FramePoller', () => {
  it('asks for the next frame a moment after one loads', () => {
    const { poller, timers, asked, fire } = setup()
    poller.start()
    poller.loaded()
    expect(timers.map(t => t.ms)).toEqual([100])
    fire()
    expect(asked).toEqual([1])
  })

  it('waits longer after a frame fails', () => {
    const { poller, timers } = setup()
    poller.start()
    poller.failed()
    expect(timers.map(t => t.ms)).toEqual([2000])
  })

  it('asks for one frame at a time', () => {
    const { poller, timers, asked, fire } = setup()
    poller.start()
    poller.loaded()
    poller.loaded()
    expect(timers).toHaveLength(1)
    fire()
    poller.loaded()
    fire()
    expect(asked).toEqual([1, 2])
  })

  it('asks for nothing while stopped', () => {
    const { poller, timers } = setup()
    poller.start()
    poller.stop()
    poller.loaded()
    expect(timers).toHaveLength(0)
  })

  // A remounted tile whose first frame comes from the browser's cache loads before React runs its effects, and
  // StrictMode then stops and starts it once: the loaded frame must still lead to the next one (D18/D41 froze).
  it('moves on from a frame that loaded before a stop and start', () => {
    const { poller, asked, fire } = setup()
    poller.loaded()
    poller.start()
    poller.stop()
    poller.start()
    fire()
    expect(asked).toEqual([1])
  })

  it('resumes when a stopped stream starts again', () => {
    const { poller, timers, asked, fire } = setup()
    poller.start()
    poller.loaded()
    poller.stop()
    expect(timers).toHaveLength(0)
    poller.start()
    fire()
    expect(asked).toEqual([1])
  })
})
