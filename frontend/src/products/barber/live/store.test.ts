import { describe, expect, it } from 'vitest'
import { initialLive, reduce } from './store'
import type { LiveSnapshot } from '../api/types'

const snap = (server_time: string) => ({ type: 'live_snapshot', server_time, scene: { 'barber-c': [] }, chairs: {} } as unknown as LiveSnapshot)

describe('live reducer', () => {
  it('takes the server clock offset from a snapshot', () => {
    const s = reduce(initialLive, { kind: 'message', msg: snap('2026-09-25T10:00:10') }, Date.parse('2026-09-25T10:00:00'))
    expect(s.offsetMs).toBe(10_000)
    expect(s.snap?.server_time).toBe('2026-09-25T10:00:10')
    expect(s.conn).toBe('connected')
  })
  it('updates only the scene from a scene message', () => {
    const s1 = reduce(initialLive, { kind: 'message', msg: snap('2026-09-25T10:00:00') }, 0)
    const s2 = reduce(s1, { kind: 'message', msg: { type: 'scene', scene: { 'barber-c': [] }, chairs: { L1: { person_id: 'p', session_id: null, sitting: 1 } } } }, 0)
    expect(s2.snap).toBe(s1.snap)
    expect(s2.chairs.L1.sitting).toBe(1)
  })
  it('freezes and resumes', () => {
    const up = reduce(initialLive, { kind: 'message', msg: snap('2026-09-25T10:00:00') }, 0)
    const down = reduce(up, { kind: 'closed' }, 5000)
    expect(down.conn).toBe('reconnecting')
    expect(down.snap).toBe(up.snap)                      // last data stays on screen
    const back = reduce(down, { kind: 'message', msg: snap('2026-09-25T10:05:00') }, 6000)
    expect(back.conn).toBe('connected')
    expect(back.offsetMs).toBe(Date.parse('2026-09-25T10:05:00') - 6000)
  })
  it('ignores unknown messages', () => {
    expect(reduce(initialLive, { kind: 'message', msg: { type: 'other' } as never }, 0)).toBe(initialLive)
  })
})
