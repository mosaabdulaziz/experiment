import { describe, expect, it } from 'vitest'
import type { ChairOcc, SceneEntry, Zone } from '../api/types'
import {
  CHAIR_STATE_COLOR, RULE_COLOR, clipBelow, drawBox, maskMarks, occupancyLayer, occupancyMarks, outlinePoints, personColor,
  aboveZone, labelSince, pushTrail, ruleLabel, ruleStyle, seatMarks, trackLabel, type TrailBuf,
} from './overlayModel'

const entry = (p: Partial<SceneEntry>): SceneEntry => ({
  track_id: 1, box: [0, 0, 0.1, 0.2], person_id: null, ticket: null, state: null, staff: null,
  areas: [], seat: null, still: true, ...p,
})

describe('occupancyMarks', () => {
  it('draws a filled mark at the anchor when counted', () => {
    const marks = occupancyMarks([entry({
      track_id: 5, anchor: [0.4, 0.6], occupancy: { chair: 'L1', why: 'counted' },
    })])
    expect(marks).toEqual([{ track_id: 5, anchor: [0.4, 0.6], foot: undefined, seatLine: undefined, counted: true, reason: null }])
  })

  it('draws a hollow mark with the reason when not counted', () => {
    const marks = occupancyMarks([entry({
      track_id: 6, anchor: [0.3, 0.5], foot: [0.31, 0.9], seat_line: [[0.1, 0.8], [0.5, 0.8]],
      occupancy: { chair: 'L1', why: 'not_still' },
    })])
    expect(marks).toEqual([{
      track_id: 6, anchor: [0.3, 0.5], foot: [0.31, 0.9], seatLine: [[0.1, 0.8], [0.5, 0.8]],
      counted: false, reason: 'not_still',
    }])
  })

  it('draws nothing for an entry missing anchor/occupancy (older backend) and never throws', () => {
    expect(() => occupancyMarks([entry({})])).not.toThrow()
    expect(occupancyMarks([entry({})])).toEqual([])
    // anchor without occupancy, and occupancy without anchor, are both skipped too
    expect(occupancyMarks([entry({ anchor: [0.1, 0.1] })])).toEqual([])
    expect(occupancyMarks([entry({ occupancy: { chair: null, why: null } })])).toEqual([])
  })

  it('is empty for an empty scene', () => {
    expect(occupancyMarks([])).toEqual([])
  })
})

describe('personColor', () => {
  it('gives the same person the same colour on every camera', () => {
    const a = personColor('person-abc')
    const b = personColor('person-abc')
    expect(a).toBe(b)
    expect(a).toMatch(/^hsl\(\d+, 70%, 55%\)$/)
  })

  it('gives different people different colours (in general)', () => {
    expect(personColor('person-abc')).not.toBe(personColor('person-xyz'))
  })

  it('is white for an unbound (null) person', () => {
    expect(personColor(null)).toBe('#ffffff')
  })
})

describe('trackLabel', () => {
  it('labels an unbound track with just its number', () => {
    expect(trackLabel(42, null)).toBe('#42')
  })
  it('labels a bound track with the first 6 chars of the person id', () => {
    expect(trackLabel(42, 'person-abcdef-extra')).toBe('#42 → person')
  })
})

describe('pushTrail', () => {
  it('keeps the anchors of the last 10s, per camera per track_id', () => {
    let buf: TrailBuf = {}
    buf = pushTrail(buf, 'cam1', [entry({ track_id: 1, anchor: [0.1, 0.1] })], 0)
    buf = pushTrail(buf, 'cam1', [entry({ track_id: 1, anchor: [0.2, 0.2] })], 4000)
    buf = pushTrail(buf, 'cam1', [entry({ track_id: 1, anchor: [0.3, 0.3] })], 9000)
    expect(buf.cam1[1]).toEqual([
      { x: 0.1, y: 0.1, t: 0 },
      { x: 0.2, y: 0.2, t: 4000 },
      { x: 0.3, y: 0.3, t: 9000 },
    ])
    // a point older than keepMs (10s) ago drops off
    buf = pushTrail(buf, 'cam1', [entry({ track_id: 1, anchor: [0.4, 0.4] })], 11500)
    expect(buf.cam1[1]).toEqual([
      { x: 0.2, y: 0.2, t: 4000 },
      { x: 0.3, y: 0.3, t: 9000 },
      { x: 0.4, y: 0.4, t: 11500 },
    ])
  })

  it('keeps separate buffers per camera and per track_id', () => {
    let buf: TrailBuf = {}
    buf = pushTrail(buf, 'cam1', [entry({ track_id: 1, anchor: [0, 0] })], 0)
    buf = pushTrail(buf, 'cam2', [entry({ track_id: 1, anchor: [0.9, 0.9] })], 0)
    buf = pushTrail(buf, 'cam1', [entry({ track_id: 2, anchor: [0.5, 0.5] })], 0)
    expect(buf.cam1[1]).toEqual([{ x: 0, y: 0, t: 0 }])
    expect(buf.cam1[2]).toEqual([{ x: 0.5, y: 0.5, t: 0 }])
    expect(buf.cam2[1]).toEqual([{ x: 0.9, y: 0.9, t: 0 }])
  })

  it('ignores entries without an anchor, and ages a track fully out once its last point expires', () => {
    let buf: TrailBuf = {}
    buf = pushTrail(buf, 'cam1', [entry({ track_id: 1, anchor: [0.1, 0.1] })], 0)
    buf = pushTrail(buf, 'cam1', [entry({ track_id: 1 })], 500) // no anchor this tick
    expect(buf.cam1[1]).toEqual([{ x: 0.1, y: 0.1, t: 0 }])
    buf = pushTrail(buf, 'cam1', [], 11000)
    expect(buf.cam1[1]).toBeUndefined()
  })

  it('never throws given an empty scene', () => {
    expect(() => pushTrail({}, 'cam1', [], 0)).not.toThrow()
  })
})

// a 0.2 x 0.4 rectangle person, top at y=0.4, bottom at y=0.8
const RECT_OUTLINE = [0.4, 0.4, 0.6, 0.4, 0.6, 0.8, 0.4, 0.8]

describe('outlinePoints', () => {
  it('pairs a flat outline into points', () => {
    expect(outlinePoints(RECT_OUTLINE)).toEqual([[0.4, 0.4], [0.6, 0.4], [0.6, 0.8], [0.4, 0.8]])
  })
  it('is null for a missing, short or malformed outline, and never throws', () => {
    for (const o of [undefined, null, [], [0.1, 0.2, 0.3, 0.4], 'x', {}, [0.1, 'a', 0.2, 0.3, 0.4, 0.5], [0.1, NaN, 0.2, 0.3, 0.4, 0.5]]) {
      expect(outlinePoints(o)).toBeNull()
    }
  })
  it('drops an odd trailing value', () => {
    expect(outlinePoints([0, 0, 1, 0, 1, 1, 0.5])).toEqual([[0, 0], [1, 0], [1, 1]])
  })
})

describe('clipBelow', () => {
  const pts = outlinePoints(RECT_OUTLINE)!
  it('keeps the part of the polygon below the cut line', () => {
    expect(clipBelow(pts, 0.66)).toEqual([[0.6, 0.66], [0.6, 0.8], [0.4, 0.8], [0.4, 0.66]])
  })
  it('keeps the whole polygon when the line is above it, nothing when below it', () => {
    expect(clipBelow(pts, 0.1)).toEqual(pts)
    expect(clipBelow(pts, 0.9)).toEqual([])
  })
  it('clips a triangle across its slanted edges', () => {
    const out = clipBelow([[0, 0], [1, 1], [0, 1]], 0.5)
    expect(out).toEqual([[0.5, 0.5], [1, 1], [0, 1], [0, 0.5]])
  })
})

describe('ruleStyle / ruleLabel', () => {
  it('colours by why: occupied green, staff cover blue, staff grey, no reason label', () => {
    expect(ruleStyle('occupied')).toEqual({ color: RULE_COLOR.occupied, reason: null })
    expect(ruleStyle('staff_cover')).toEqual({ color: RULE_COLOR.staff_cover, reason: null })
    expect(ruleStyle('staff')).toEqual({ color: RULE_COLOR.staff, reason: null })
  })
  it('draws every other verdict orange with its reason', () => {
    for (const why of ['low_conf', 'no_mask', 'below_min', 'other_chair', 'no_chair', 'other_view']) {
      expect(ruleStyle(why)).toEqual({ color: RULE_COLOR.other, reason: why })
    }
  })
  it('draws an unknown or missing why orange without a reason', () => {
    expect(ruleStyle('something_new')).toEqual({ color: RULE_COLOR.other, reason: null })
    expect(ruleStyle(undefined)).toEqual({ color: RULE_COLOR.other, reason: null })
  })
  it('labels "chair · share%" only when both are known', () => {
    expect(ruleLabel({ chair: 'R3', share: 0.234, why: 'occupied' })).toBe('R3 · 23%')
    expect(ruleLabel({ chair: 'R3', share: 1.4, why: 'occupied' })).toBe('R3 · 100%')
    expect(ruleLabel({ chair: null, share: 0.5, why: 'no_chair' })).toBeNull()
    expect(ruleLabel({ chair: 'R3', share: null, why: 'no_mask' })).toBeNull()
    expect(ruleLabel(undefined)).toBeNull()
    expect(ruleLabel(null)).toBeNull()
  })
})

describe('maskMarks / occupancyLayer', () => {
  it('draws the outline, the shaded lower part and the rule colour and label', () => {
    const [m] = maskMarks([entry({ track_id: 3, outline: RECT_OUTLINE, lower_y: 0.66,
                                   rule: { chair: 'R3', share: 0.23, why: 'below_min' } })])
    expect(m).toEqual({
      track_id: 3, points: [[0.4, 0.4], [0.6, 0.4], [0.6, 0.8], [0.4, 0.8]],
      lower: [[0.6, 0.66], [0.6, 0.8], [0.4, 0.8], [0.4, 0.66]], lowerY: 0.66,
      color: RULE_COLOR.other, reason: 'below_min', label: 'R3 · 23%',
    })
  })
  it('draws a plain white outline without a rule, and no shading without a cut line', () => {
    const [m] = maskMarks([entry({ outline: RECT_OUTLINE, lower_y: null })])
    expect(m).toMatchObject({ color: '#ffffff', reason: null, label: null, lower: [], lowerY: null })
  })
  it('falls back to the old anchor marks only for entries without an outline', () => {
    const layer = occupancyLayer([
      entry({ track_id: 1, outline: RECT_OUTLINE, anchor: [0.5, 0.5], occupancy: { chair: 'L1', why: 'counted' } }),
      entry({ track_id: 2, anchor: [0.2, 0.2], occupancy: { chair: 'L1', why: 'not_still' } }),
    ])
    expect(layer.masks.map(m => m.track_id)).toEqual([1])
    expect(layer.legacy.map(m => m.track_id)).toEqual([2])
  })
  it('draws nothing for an older backend entry with neither, and never throws', () => {
    expect(occupancyLayer([entry({})])).toEqual({ masks: [], legacy: [] })
    expect(() => maskMarks([entry({ outline: [1, 2] as number[], rule: null, lower_y: undefined })])).not.toThrow()
  })
})

describe('seatMarks', () => {
  const zone = (z: Partial<Zone>): Zone => ({ id: 'z', name: '', type: 'chair', points: [[0, 0], [1, 0], [1, 1]], ...z })
  const zones: Zone[] = [
    zone({ id: 'c1', name: 'chair right 3', type: 'chair', pair: 'R3', owner: true }),
    zone({ id: 's1', type: 'seat', seat_of: 'c1' }),
    zone({ id: 'L1', name: 'L1', type: 'chair' }),
    zone({ id: 's2', type: 'seat', seat_of: 'L1' }),
  ]
  const occ = (p: Partial<ChairOcc>): ChairOcc => ({ person_id: null, session_id: null, sitting: 0, ...p })
  const now = Date.parse('2026-10-05T12:00:00')

  it('colours each seat by its chair state, with the time in the state', () => {
    const marks = seatMarks(zones, 'cam1', {
      R3: occ({ state: 'sitting', since: '2026-10-05T11:47:26', camera: 'cam1' }),
      L1: occ({ state: 'empty', since: null, camera: 'cam1' }),
    }, now)
    expect(marks).toEqual([
      { id: 's1', chair: 'c1', name: 'R3', points: zones[1].points, state: 'sitting', color: CHAIR_STATE_COLOR.sitting,
        dashed: false, seconds: 754, decision: null, checking: false },
      { id: 's2', chair: 'L1', name: 'L1', points: zones[3].points, state: 'empty', color: CHAIR_STATE_COLOR.empty,
        dashed: false, seconds: null, decision: null, checking: false },
    ])
  })
  it('times a resumed haircut from the haircut start, not the sitting (2026-10-18)', () => {
    const [m] = seatMarks(zones.slice(0, 2), 'cam1', {
      R3: occ({ state: 'sitting', since: '2026-10-05T11:58:00', haircut_since: '2026-10-05T11:40:00', camera: 'cam1' }),
    }, now)
    expect(m.seconds).toBe(20 * 60)
  })
  it('carries the re-ID check: checking while it runs, then its answer for 5 minutes (2026-10-18)', () => {
    const d = { resumed: false, sim: 0.41, threshold: 0.62, gap_s: 60, at: '2026-10-05T11:58:00', ticket: 9 }
    const mark = (p: Partial<ChairOcc>) => seatMarks(zones.slice(0, 2), 'cam1', {
      R3: occ({ state: 'sitting', since: '2026-10-05T11:57:00', camera: 'cam1', ...p }) }, now)[0]
    expect(mark({ resume: d })).toMatchObject({ decision: d, checking: false })
    expect(mark({ resume: d, checking: '2026-10-05T11:59:50' })).toMatchObject({ decision: null, checking: true })
    expect(mark({ resume: { ...d, at: '2026-10-05T11:54:00' } }).decision).toBeNull()      // 6 min old
    expect(mark({ state: 'away', resume: d }).decision).toBeNull()
  })
  it('keeps the time away from the away start', () => {
    const [m] = seatMarks(zones.slice(0, 2), 'cam1', {
      R3: occ({ state: 'away', since: '2026-10-05T11:59:00', haircut_since: '2026-10-05T11:40:00', camera: 'cam1' }),
    }, now)
    expect(m.seconds).toBe(60)
  })
  it('counts up the time away when away, and dashes no_view', () => {
    const [away] = seatMarks(zones, 'cam1', { R3: occ({ state: 'away', since: '2026-10-05T11:58:00',
                                                         resume_until: '2026-10-05T12:03:00', camera: 'cam1' }) }, now)
    expect(away).toMatchObject({ state: 'away', seconds: 120, dashed: false })
    const [nv] = seatMarks(zones, 'cam1', { L1: occ({ state: 'no_view', since: null, camera: 'cam1' }) }, now)
    expect(nv).toMatchObject({ state: 'no_view', dashed: true, color: CHAIR_STATE_COLOR.no_view, seconds: null })
  })
  it('debounces starting: empty under 3 s (or without since), starting from 3 s', () => {
    const st = (since?: string) => seatMarks(zones, 'cam1', { R3: occ({ state: 'starting', since, camera: 'cam1' }) }, now)[0]
    expect(st('2026-10-05T11:59:58')).toMatchObject({ state: 'empty', color: CHAIR_STATE_COLOR.empty, seconds: null })
    expect(st(undefined)).toMatchObject({ state: 'empty' })
    expect(st('2026-10-05T11:59:57')).toMatchObject({ state: 'starting', color: CHAIR_STATE_COLOR.starting, seconds: 3 })
  })
  it('draws only on the camera owning the chair', () => {
    expect(seatMarks(zones, 'cam2', { R3: occ({ state: 'sitting', camera: 'cam1' }) }, now)).toEqual([])
    // without camera: the zone owning the pair (or an unpaired zone) decides
    expect(seatMarks(zones, 'cam2', { R3: occ({ state: 'starting' }) }, now).map(m => m.name)).toEqual(['R3'])
    const copy = [zone({ id: 'c9', type: 'chair', pair: 'R3' }), zone({ id: 's9', type: 'seat', seat_of: 'c9' })]
    expect(seatMarks(copy, 'cam2', { R3: occ({ state: 'starting' }) }, now)).toEqual([])
  })
  it('draws nothing for an older backend (no state) or bad data, and never throws', () => {
    expect(seatMarks(zones, 'cam1', { R3: occ({}) }, now)).toEqual([])
    expect(seatMarks(zones, 'cam1', {}, now)).toEqual([])
    expect(seatMarks(zones, 'cam1', undefined, now)).toEqual([])
    expect(seatMarks(zones, 'cam1', { R3: occ({ state: 'weird' as never, camera: 'cam1' }) }, now)).toEqual([])
    expect(seatMarks([zone({ id: 's', type: 'seat', seat_of: 'missing' })], 'cam1', { missing: occ({ state: 'sitting' }) }, now)).toEqual([])
    expect(seatMarks(zones, 'cam1', { R3: occ({ state: 'sitting', since: 'garbage', camera: 'cam1' }) }, now)[0].seconds).toBeNull()
  })
})

describe('drawBox', () => {
  const outline = [0, 0, 1, 0, 1, 1]
  it('hides the box only when occupancy is on and the outline is usable', () => {
    expect(drawBox(entry({ outline }), true)).toBe(false)
    expect(drawBox(entry({ outline }), false)).toBe(true)
    expect(drawBox(entry({}), true)).toBe(true)
    expect(drawBox(entry({ outline: [0, 0, 1, 1] }), true)).toBe(true)
  })
})

describe('labelSince', () => {
  it('counts a haircut from its start, so a resume goes on (2026-10-18)', () => {
    expect(labelSince({ state: 'haircut', state_since: '2026-10-18T11:58:00', haircut_started_at: '2026-10-18T11:40:00' }))
      .toBe('2026-10-18T11:40:00')
  })
  it('counts any other state from its own start', () => {
    expect(labelSince({ state: 'waiting', state_since: '2026-10-18T11:58:00', haircut_started_at: null })).toBe('2026-10-18T11:58:00')
    expect(labelSince({ state: 'washing', state_since: '2026-10-18T11:58:00', haircut_started_at: '2026-10-18T11:40:00' }))
      .toBe('2026-10-18T11:58:00')
  })
})

describe('aboveZone', () => {
  // a 1000 x 500 picture at (10, 20); the zone's box spans x 0.4-0.5 -> px 410-510
  const rect = { x: 10, y: 20, w: 1000, h: 500 }
  const pts = (y0: number, y1: number) => [[0.4, y0], [0.5, y0], [0.5, y1], [0.4, y1]]
  it('puts the tag right on top of the zone, centred on it, off the chair (2026-10-18, user)', () => {
    // y 0.4-0.6 -> px 220-320: the tag's bottom 3 px above the zone's top
    expect(aboveZone(pts(0.4, 0.6), rect, 20)).toEqual({ x: 460, y: 197, anchor: 'middle' })
  })
  it('goes just under the zone when the picture ends above it', () => {
    // y 0.02-0.2 -> px 30-120: no 23 px above it inside the picture
    expect(aboveZone(pts(0.02, 0.2), rect, 20)).toEqual({ x: 460, y: 123, anchor: 'middle' })
  })
})
