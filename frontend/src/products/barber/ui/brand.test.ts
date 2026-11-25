import { describe, expect, it } from 'vitest'
import { shopName, tabTitle } from './brand'

describe('shopName', () => {
  it("spells the site's place as words (2026-10-18)", () => {
    expect(shopName({ place: 'BekkyBarber', branch: 'Palm Hills' })).toBe('Bekky Barber')
    expect(shopName({ place: 'Bekky Barber', branch: null })).toBe('Bekky Barber')
  })
  it('is null without a site (an older backend)', () => {
    expect(shopName(undefined)).toBeNull()
    expect(shopName({ place: '', branch: null })).toBeNull()
  })
})

describe('tabTitle', () => {
  it('names the shop, its branch and IRIS', () => {
    expect(tabTitle({ place: 'BekkyBarber', branch: 'Palm Hills' })).toBe('Bekky Barber · Palm Hills · IRIS')
    expect(tabTitle({ place: 'BekkyBarber', branch: null })).toBe('Bekky Barber · IRIS')
    expect(tabTitle(undefined)).toBe('IRIS · Barber Ops')
  })
})
