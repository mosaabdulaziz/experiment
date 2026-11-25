import { describe, expect, it } from 'vitest'
import type { Staff } from '../api/types'
import { chairOptions, createLatestGuard } from './staffChairs'

const s = (id: string, name: string, active: boolean): Staff => ({ id, name, active, created_at: '2026-09-01T00:00:00' })

describe('chairOptions', () => {
  const staff = [s('1', 'Ahmed', true), s('2', 'Karim', true), s('3', 'Samy', false)]

  it('lists active staff only when the chair has no inactive barber', () => {
    expect(chairOptions(staff, null)).toEqual([
      { id: '1', name: 'Ahmed', disabled: false },
      { id: '2', name: 'Karim', disabled: false },
    ])
  })

  it('leaves an already-active current barber alone (no duplicate)', () => {
    expect(chairOptions(staff, '1')).toEqual([
      { id: '1', name: 'Ahmed', disabled: false },
      { id: '2', name: 'Karim', disabled: false },
    ])
  })

  it('appends the current barber, disabled, when they have since gone inactive', () => {
    expect(chairOptions(staff, '3')).toEqual([
      { id: '1', name: 'Ahmed', disabled: false },
      { id: '2', name: 'Karim', disabled: false },
      { id: '3', name: 'Samy', disabled: true },
    ])
  })

  it('drops a current barber id that no longer exists', () => {
    expect(chairOptions(staff, 'ghost')).toEqual([
      { id: '1', name: 'Ahmed', disabled: false },
      { id: '2', name: 'Karim', disabled: false },
    ])
  })
})

describe('createLatestGuard', () => {
  it('says only the most recently started token is current', () => {
    const g = createLatestGuard()
    const a = g.start()
    const b = g.start()
    expect(g.isCurrent(a)).toBe(false)
    expect(g.isCurrent(b)).toBe(true)
  })

  it('is current for a lone request', () => {
    const g = createLatestGuard()
    const a = g.start()
    expect(g.isCurrent(a)).toBe(true)
  })

  it('lets an out-of-order resolution be told apart from an in-order one', () => {
    // Simulates: request A starts, then B starts before A resolves; A's
    // result arrives late (out of order) and must be dropped, B's is kept.
    const g = createLatestGuard()
    const a = g.start()
    const b = g.start()
    const results: string[] = []
    if (g.isCurrent(b)) results.push('B applied')
    if (g.isCurrent(a)) results.push('A applied') // must not run
    expect(results).toEqual(['B applied'])
  })

  it('tracks a third, later request as the new latest', () => {
    const g = createLatestGuard()
    const a = g.start()
    const b = g.start()
    const c = g.start()
    expect([a, b, c].map(g.isCurrent)).toEqual([false, false, true])
  })
})
