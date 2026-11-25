import { describe, expect, it } from 'vitest'
import { pointIn } from './kidsCarPick'

describe('pointIn', () => {
  const rect = { x: 10, y: 20, w: 200, h: 100 }          // the picture inside its box
  const box = { left: 0, top: 0 }
  it('gives the normalised point on the picture', () => {
    expect(pointIn(rect, 110, 70, box)).toEqual([0.5, 0.5])
    expect(pointIn(rect, 10, 20, { left: 0, top: 0 })).toEqual([0, 0])
  })
  it('refuses a click beside the picture', () => {
    expect(pointIn(rect, 5, 70, box)).toBeNull()
    expect(pointIn(rect, 110, 125, box)).toBeNull()
  })
  it('accounts for where the box sits on the page', () => {
    expect(pointIn(rect, 160, 120, { left: 50, top: 50 })).toEqual([0.5, 0.5])
  })
})
