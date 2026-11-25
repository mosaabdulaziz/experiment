import { describe, expect, it } from 'vitest'
import { photoReportUrl } from '../api/client'
import { photoLinks } from './reportPhotos'

describe('photoReportUrl', () => {
  it('builds the photo report URL', () => {
    expect(photoReportUrl({ start: '2026-10-16', lang: 'ar', format: 'pdf' }))
      .toBe('/barber/api/barber/report/photos?start=2026-10-16&lang=ar&format=pdf')
  })
})

describe('photoLinks', () => {
  it('shows the links for a daily report only', () => {
    expect(photoLinks('day', '2026-10-16', 'en', true).show).toBe(true)
    expect(photoLinks('week', '2026-10-16', 'en', true).show).toBe(false)
    expect(photoLinks('month', '2026-10', 'en', true).show).toBe(false)
  })
  it('enables them on the shop network only, not while unknown', () => {
    expect(photoLinks('day', '2026-10-16', 'en', true).enabled).toBe(true)
    expect(photoLinks('day', '2026-10-16', 'en', false).enabled).toBe(false)
    expect(photoLinks('day', '2026-10-16', 'en', null).enabled).toBe(false)
  })
  it('opens the HTML and downloads the PDF of that day and language', () => {
    const l = photoLinks('day', '2026-10-16', 'ar', true)
    expect(l.html).toBe('/barber/api/barber/report/photos?start=2026-10-16&lang=ar&format=html')
    expect(l.pdf).toBe('/barber/api/barber/report/photos?start=2026-10-16&lang=ar&format=pdf')
  })
})
