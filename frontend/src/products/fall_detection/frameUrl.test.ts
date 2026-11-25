// AC-F33.2: the fall tab and the dashboard wall draw the SAME engine-baked frame endpoint,
// so a condition the engine bakes onto a box appears identically in both (Phase 2 audit rewording).
import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { frameUrl } from './live/frameUrl'

// The wall's tiles (StreamTile, ExpandedStream) take their frame URL from useFramePoll.
const wallTile = readFileSync(new URL('../iris/components/streams/framePoll.ts', import.meta.url), 'utf8')

describe('the frame endpoint is the wall’s own', () => {
  it('the dashboard wall asks for /stream/<cam>/frame', () => {
    expect(wallTile).toMatch(/`\/stream\/\$\{[^}]+\}\/frame/)
  })

  it('the fall tab asks for the same endpoint', () => {
    expect(frameUrl('cam-1')).toMatch(/^\/stream\/cam-1\/frame/)
  })
})
