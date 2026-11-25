import { describe, expect, it } from 'vitest'
import { dragWidth, keyWidth } from './resize'

const limits = { min: 300, max: 760 }

describe('dragging the inner edge of a side panel', () => {
  it('widens a right-hand panel as its edge moves left, and narrows it as it moves right', () => {
    expect(dragWidth(460, 1000, 900, false, limits)).toBe(560)
    expect(dragWidth(460, 1000, 1080, false, limits)).toBe(380)
  })
  it('mirrors in right-to-left pages, where the panel sits on the left', () => {
    expect(dragWidth(460, 400, 500, true, limits)).toBe(560)
  })
  it('stops at the limits', () => {
    expect(dragWidth(460, 1000, 0, false, limits)).toBe(760)
    expect(dragWidth(460, 1000, 2000, false, limits)).toBe(300)
  })
})

describe('the keyboard on the edge', () => {
  it('moves the edge with the arrows (wider away from the panel) and jumps with Home / End', () => {
    expect(keyWidth(460, 'ArrowLeft', false, limits)).toBe(484)
    expect(keyWidth(460, 'ArrowRight', false, limits)).toBe(436)
    expect(keyWidth(460, 'ArrowLeft', true, limits)).toBe(436)
    expect(keyWidth(460, 'Home', false, limits)).toBe(300)
    expect(keyWidth(460, 'End', false, limits)).toBe(760)
  })
  it('ignores other keys', () => {
    expect(keyWidth(460, 'Enter', false, limits)).toBeNull()
  })
})
