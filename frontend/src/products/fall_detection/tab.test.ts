// AC-F33.1: the tab is registered in the products registry and adds no Redis contract, no
// WebSocket and no box math — it draws the engine-baked JPEG.
// AC-F34.1: the first release has no WebSocket, and the JPEG layer is its own module separate
// from the page, so the (future) chip layer lands without changing the page.
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { join, relative, sep } from 'node:path'
import { describe, expect, it } from 'vitest'
import { PRODUCTS } from '../../shared/products'

const here = fileURLToPath(new URL('.', import.meta.url))
const sharedProducts = readFileSync(new URL('../../shared/products.ts', import.meta.url), 'utf8')

function sources(dir: string): string[] {
  return readdirSync(dir).flatMap(name => {
    const p = join(dir, name)
    if (statSync(p).isDirectory()) return sources(p)
    return /\.(ts|tsx)$/.test(name) && !/\.test\.tsx?$/.test(name) ? [p] : []
  })
}

// Every source file in the product, with forward-slash paths so the guards below read the same
// on Windows and on Linux.
const files = sources(here)
const texts = files.map(p => ({ p: relative(here, p).split(sep).join('/'), text: readFileSync(p, 'utf8') }))

describe('the product is registered', () => {
  it('PRODUCTS carries a fall_detection entry loading this folder', () => {
    expect(PRODUCTS.some(p => p.id === 'fall_detection')).toBe(true)
    expect(sharedProducts).toMatch(/import\('\.\.\/products\/fall_detection'\)/)
  })
})

describe('the tab adds no contracts of its own (AC-F33.1, AC-F34.1)', () => {
  it('there are source files to judge', () => {
    expect(files.length).toBeGreaterThan(0)
  })

  it('no WebSocket anywhere in the product', () => {
    expect(texts.filter(t => t.text.includes('WebSocket')).map(t => t.p)).toEqual([])
  })

  it('no Redis contract: no stream key, no XADD', () => {
    expect(texts.filter(t => /iris:|XADD/.test(t.text)).map(t => t.p)).toEqual([])
  })

  it('no box math: nothing draws on a canvas', () => {
    expect(texts.filter(t => /<canvas|getContext\(/.test(t.text)).map(t => t.p)).toEqual([])
  })
})

describe('the JPEG layer is separate from the page (AC-F34.1)', () => {
  it('a frame-layer module owns the endpoint', () => {
    expect(existsSync(join(here, 'live/FrameLayer.tsx'))).toBe(true)
    expect(readFileSync(join(here, 'live/frameUrl.ts'), 'utf8')).toContain('/frame')
  })

  it('only the frame module knows the endpoint, whatever else the product grows', () => {
    const knowing = texts.filter(t => t.text.includes('/frame')).map(t => t.p).sort()
    expect(knowing).toEqual(['live/FrameLayer.tsx', 'live/frameUrl.ts'])
  })

  it('a page composes the layer', () => {
    const pages = texts.filter(t => !t.p.startsWith('live/') && t.text.includes('FrameLayer'))
    expect(pages.length).toBeGreaterThan(0)
  })
})
