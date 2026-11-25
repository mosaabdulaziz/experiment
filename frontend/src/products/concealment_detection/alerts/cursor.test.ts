import { describe, expect, it } from 'vitest'
import { advance } from './cursor'

describe('incremental poll cursor', () => {
  it('advances monotonically to the highest candidate id in the batch', () => {
    expect(advance(0, [{ id: 3 }, { id: 7 }, { id: 5 }])).toBe(7)
    expect(advance(7, [{ id: 4 }, { id: 7 }])).toBe(7)
    expect(advance(7, [])).toBe(7)
  })
})
