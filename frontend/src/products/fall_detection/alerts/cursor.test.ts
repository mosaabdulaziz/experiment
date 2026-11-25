import { describe, expect, it } from 'vitest'
import { advance } from './cursor'

describe('the feed cursor', () => {
  it('leaves the cursor unchanged on an empty batch', () => {
    expect(advance(7, [])).toBe(7)
  })

  it('moves the cursor to the batch\'s newest id when the batch is newer', () => {
    expect(advance(4, [{ id: 5 }, { id: 7 }, { id: 6 }])).toBe(7)
  })

  it('never moves the cursor back for a batch entirely older than it', () => {
    expect(advance(9, [{ id: 6 }, { id: 8 }])).toBe(9)
  })

  it('leaves the cursor where it is for a replayed batch whose newest id equals it', () => {
    expect(advance(5, [{ id: 5 }, { id: 3 }])).toBe(5)
  })
})
