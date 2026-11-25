import { describe, expect, it } from 'vitest'
import { unseenCount } from './badge'

describe('unseen badge count', () => {
  const rows = [
    { id: 1, severity: 'warning' as const, status: 'candidate' as const },
    { id: 2, severity: 'info' as const, status: 'candidate' as const },
    { id: 3, severity: 'warning' as const, status: 'acknowledged' as const },
    { id: 4, severity: 'warning' as const, status: 'candidate' as const },
  ]

  it('counts unreviewed warning and info candidates newer than seenCursor', () => {
    expect(unseenCount(rows, 0)).toBe(3)
    expect(unseenCount(rows, 1)).toBe(2)
  })

  it('drops to zero when seenCursor catches up to the newest id', () => {
    expect(unseenCount(rows, 4)).toBe(0)
  })
})
