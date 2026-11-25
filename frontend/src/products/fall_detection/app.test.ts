// The fall tab is the house: the tokens are the house's, copied, not re-invented. If this
// drifts, the tab is a different product wearing the same colours.
import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const ours = readFileSync(new URL('./app.css', import.meta.url), 'utf8')
const house = readFileSync(new URL('../barber/app.css', import.meta.url), 'utf8')

function tokens(css: string): Record<string, string> {
  const block = css.slice(css.indexOf(':root'), css.indexOf('}', css.indexOf(':root')))
  const out: Record<string, string> = {}
  for (const line of block.split('\n')) {
    const m = /^\s*(--[a-z0-9-]+)\s*:\s*(.+);\s*$/.exec(line)
    if (m) out[m[1]] = m[2].trim()
  }
  return out
}

const SHARED = [
  '--green-900', '--green-800', '--green-700', '--green-600', '--green-500', '--green-50',
  '--ground', '--card', '--ink', '--ink-soft', '--ink-mute', '--line', '--line-strong',
  '--red-500', '--red-50', '--amber-600', '--amber-500', '--amber-50',
  '--radius', '--shadow',
]

// The house's blocks this product copies, as the house's own line numbers. Each one must appear
// in our stylesheet exactly as it appears in the house's -- copied, not retyped.
const HOUSE_BLOCKS: Array<[number, number]> = [
  [10, 45], [47, 59], [61, 64], [68, 138], [140, 165], [166, 166], [171, 171],
  [176, 195], [206, 213], [215, 242], [244, 260], [357, 368], [470, 488], [533, 559],
  [648, 653], [759, 763], [845, 857], [858, 859],
]

describe('the fall tab uses the house tokens', () => {
  it('every shared token is the house value, character for character', () => {
    const a = tokens(ours)
    const b = tokens(house)
    for (const name of SHARED) {
      expect(a[name], `${name} must be the house's value`).toBeDefined()
      expect(a[name], `${name} must be the house's value`).toBe(b[name])
    }
  })

  it('every token the house declares has the house value, character for character', () => {
    const a = tokens(ours)
    const b = tokens(house)
    for (const name of Object.keys(b)) {
      expect(a[name], `${name} must be the house's value`).toBe(b[name])
    }
  })

  it('the stylesheet declares every token the house declares', () => {
    expect(Object.keys(tokens(ours))).toEqual(expect.arrayContaining(Object.keys(tokens(house))))
  })
})

describe('the fall tab copies the house blocks, it does not retype them', () => {
  const houseLines = house.split('\n')

  for (const [from, to] of HOUSE_BLOCKS) {
    it(`the house's lines ${from}-${to} are in ours, byte for byte`, () => {
      const block = houseLines.slice(from - 1, to).join('\n')
      expect(ours, `barber/app.css:${from}-${to} must appear verbatim`).toContain(block)
    })
  }
})

// 2026-10-29 (user): the frame around the page is the IRIS surveillance product's own background
// -- its mesh over near-black -- not the house's dark green. That is a deliberate departure from
// one house block, so the block above stays copied verbatim and this product's override follows
// it; the later rule is the one the browser applies.
const iris = readFileSync(new URL('../iris/styles/tokens.css', import.meta.url), 'utf8')

function declared(css: string, name: string): string {
  const from = css.indexOf(`${name}:`)
  return from < 0 ? '' : css.slice(from + name.length + 1, css.indexOf(';', from))
}

const squash = (s: string) => s.replace(/\s+/g, ' ').trim()

describe('the frame is the IRIS glow, not the house green', () => {
  const blocks = [...ours.matchAll(/(?:^|\n)html,\s*body\s*\{([\s\S]*?)\}/g)].map(m => m[1])
  const applied = squash(blocks[blocks.length - 1] ?? '')

  it('the frame sits on the IRIS near-black, not on the green', () => {
    expect(blocks.length, 'the product declares its own html/body rule after the house one')
      .toBeGreaterThan(1)
    expect(applied).toContain(`background-color: ${declared(iris, '--bg-deep').trim()}`)
    expect(applied).not.toContain('--green-900')
  })

  it('the frame carries the IRIS mesh, value for value', () => {
    const stops = squash(declared(iris, '--mesh'))
      .split(/(?=radial-gradient\()/)
      .filter(Boolean)
    expect(stops.length).toBe(4)
    for (const stop of stops) expect(applied).toContain(stop)
  })
})
