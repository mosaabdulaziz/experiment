import { describe, expect, it } from 'vitest'
import { retentionMonths, visitCount } from './customers'

describe('customers', () => {
  it('says the retention in months', () => {
    expect(retentionMonths(365)).toBe(12)
    expect(retentionMonths(180)).toBe(6)
    expect(retentionMonths(30)).toBe(1)
  })
  it('counts visits with the right plural', () => {
    expect([1, 2, 3, 10, 11, 25].map(n => visitCount('en', n)))
      .toEqual(['1 visit', '2 visits', '3 visits', '10 visits', '11 visits', '25 visits'])
    expect([1, 2, 3, 10, 11, 1200].map(n => visitCount('ar', n)))
      .toEqual(['زيارة واحدة', 'زيارتان', '3 زيارات', '10 زيارات', '11 زيارة', '1,200 زيارة'])
  })
})
