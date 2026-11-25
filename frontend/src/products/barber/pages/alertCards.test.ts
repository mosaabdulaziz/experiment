import { describe, expect, it } from 'vitest'
import { alertCard, alertSections, cropView, openAlertsKey, pageSections, type AlertCardModel } from './alertCards'
import type { Alert } from '../api/types'

const base: Alert = { id: 'a1', kind: 'excessive_wait', camera: 'shop', visit_id: 'w1', session_id: null,
  started_at: '2026-10-05T12:00:00', ended_at: null, open: true, chair_id: null, barber_id: null,
  has_snapshot: true, subject_box: [0.5, 0.4, 0.1, 0.2] }

/** the region of the frame the card shows, in frame pixels */
function region(v: ReturnType<typeof cropView>, W: number, H: number, A = 4 / 3) {
  const rw = 1 / (v.widthPct / 100) // fraction of frame width
  const rh = rw * W / (H * A)       // uniform scaling => height follows from the card aspect
  return { rw, rh, left: -v.leftPct / 100 * rw, top: -v.topPct / 100 * rh, aspect: (rw * W) / (rh * H) }
}

describe('cropView', () => {
  it('shows a typical person box uniformly at the card aspect, centred on it', () => {
    const v = cropView([0.5, 0.4, 0.1, 0.2], 1920, 1080)
    const r = region(v, 1920, 1080)
    expect(r.aspect).toBeCloseTo(4 / 3, 6)
    expect(r.left + r.rw / 2).toBeCloseTo(0.55, 6)
    expect(r.top + r.rh / 2).toBeCloseTo(0.5, 6)
    // grown height 0.32 * 1080 = 345.6 px is the long side: region is 460.8 x 345.6
    expect(r.rw * 1920).toBeCloseTo(460.8, 4)
    expect(r.rh * 1080).toBeCloseTo(345.6, 4)
  })
  it('stays inside the frame at an edge', () => {
    const v = cropView([0.95, 0.0, 0.1, 0.1], 1920, 1080)
    const r = region(v, 1920, 1080)
    expect(r.left).toBeGreaterThanOrEqual(-1e-9)
    expect(r.left + r.rw).toBeLessThanOrEqual(1 + 1e-9)
    expect(r.top).toBeGreaterThanOrEqual(-1e-9)
    expect(r.top + r.rh).toBeLessThanOrEqual(1 + 1e-9)
    expect(r.aspect).toBeCloseTo(4 / 3, 6)
  })
  it('a box larger than the frame shows the largest 4:3 region that fits, uniformly', () => {
    const v = cropView([0, 0, 1, 1], 1920, 1080)
    const r = region(v, 1920, 1080)
    expect(r.rh).toBeCloseTo(1, 6)
    expect(r.aspect).toBeCloseTo(4 / 3, 6)
    expect(r.left).toBeGreaterThanOrEqual(0)
    expect(r.left + r.rw).toBeLessThanOrEqual(1 + 1e-9)
  })
  it('a zero-size box or frame gives finite numbers', () => {
    for (const v of [cropView([0.5, 0.5, 0, 0], 1920, 1080), cropView([0.5, 0.5, 0.1, 0.1], 0, 0)])
      expect(Object.values(v).every(Number.isFinite)).toBe(true)
  })
})

describe('alertCard', () => {
  it('an open wait alert counts live and has a snapshot url', () => {
    const m = alertCard(base, Date.parse('2026-10-05T12:01:30'), {}, () => null)
    expect(m.kindKey).toBe('alert.excessive_wait')
    expect(m.open).toBe(true)
    expect(m.durationS).toBe(90)
    expect(m.imageUrl).toContain('/api/barber/alerts/a1/snapshot')
  })
  it('alertCard without a snapshot has no image url', () => {
    expect(alertCard({ ...base, has_snapshot: false }, 0, {}, () => null).imageUrl).toBeNull()
  })
  it('a closed haircut alert names its chair and barber and its length', () => {
    const m = alertCard({ ...base, kind: 'excessive_service', open: false, ended_at: '2026-10-05T12:10:00',
      chair_id: 'R1', barber_id: 'staff-1' }, 0, { 'staff-1': 'Hassan' }, id => (id === 'R1' ? 'R1' : null))
    expect([m.chair, m.barber, m.durationS, m.open]).toEqual(['R1', 'Hassan', 600, false])
  })
})

describe('alert card ticket and refetch key', () => {
  it('carries the ticket, null when absent', () => {
    expect(alertCard({ ...base, ticket: 7 }, 0, {}, () => null).ticket).toBe(7)
    expect(alertCard(base, 0, {}, () => null).ticket).toBeNull()
  })
  it('the open key changes when a picture is saved or an alert opens', () => {
    const a = [{ id: 'a1', has_snapshot: false }, { id: 'a2', has_snapshot: true }]
    expect(openAlertsKey(a)).toBe('a1,a2*')
    expect(openAlertsKey([{ id: 'a1', has_snapshot: true }, a[1]])).not.toBe(openAlertsKey(a))
    expect(openAlertsKey(undefined)).toBe('')
  })
})

describe('alertSections (2026-10-18: open first, then closed by day)', () => {
  const card = (id: string, startedAt: string, open = false) =>
    ({ id, startedAt, open, kindKey: 'alert.excessive_wait', durationS: 0, chair: null, barber: null, ticket: null,
       imageUrl: null, box: null }) as AlertCardModel
  const ids = (cs: AlertCardModel[]) => cs.map(c => c.id)

  it('puts every open alert on top, newest first, even one from yesterday', () => {
    const s = alertSections([card('c1', '2026-10-18T11:00:00'), card('o1', '2026-10-17T22:00:00', true),
                             card('o2', '2026-10-18T10:00:00', true)], 5)
    expect(s[0]).toEqual({ kind: 'open', cards: [expect.objectContaining({ id: 'o2' }), expect.objectContaining({ id: 'o1' })] })
    expect(s.slice(1).map(x => x.kind)).toEqual(['day'])
  })
  it('groups the closed ones by shop day across the day start, newest day and newest alert first', () => {
    const s = alertSections([card('a', '2026-10-17T12:00:00'), card('b', '2026-10-18T02:00:00'),
                             card('c', '2026-10-18T06:00:00'), card('d', '2026-10-18T09:00:00')], 5)
    expect(s.map(x => (x.kind === 'day' ? [x.day, ids(x.cards)] : ['open', ids(x.cards)]))).toEqual([
      ['2026-10-18', ['d', 'c']],
      ['2026-10-17', ['b', 'a']],                     // 02:00 belongs to the 17th's shop day (it starts at 05:00)
    ])
  })
  it('has no open section when nothing is open, and nothing for no alerts', () => {
    expect(alertSections([card('a', '2026-10-17T12:00:00')], 5)[0].kind).toBe('day')
    expect(alertSections([], 5)).toEqual([])
  })
  it('pages over the ordered cards, keeping the heading of a section a page cuts', () => {
    const s = alertSections([card('o', '2026-10-18T10:00:00', true), card('a', '2026-10-18T09:00:00'),
                             card('b', '2026-10-18T08:00:00'), card('c', '2026-10-17T08:00:00')], 5)
    expect(pageSections(s, 0, 2).map(x => [x.kind, ids(x.cards)])).toEqual([['open', ['o']], ['day', ['a']]])
    expect(pageSections(s, 1, 2).map(x => [x.kind, ids(x.cards)])).toEqual([['day', ['b']], ['day', ['c']]])
    expect(pageSections(s, 1, 2)[0]).toMatchObject({ day: '2026-10-18' })
  })
})
