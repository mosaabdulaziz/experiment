import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { frameUrl } from './live/frameUrl'

const wallPoller = readFileSync(new URL('../iris/components/streams/framePoll.ts', import.meta.url), 'utf8')

describe('the frame endpoint is the wall’s own', () => {
  it('the dashboard wall asks for /stream/<cam>/frame', () => {
    expect(wallPoller).toMatch(/`\/stream\/\$\{[^}]+\}\/frame/)
  })

  it('the concealment tab asks for the same endpoint', () => {
    expect(frameUrl('cam-1')).toMatch(/^\/stream\/cam-1\/frame/)
  })
})
