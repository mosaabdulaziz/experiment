import { describe, expect, it } from 'vitest'
import {
  byDay, durationSeconds, filtersFromQuery, isOpen, matches, needsAck, newIncident, NO_FILTERS, sortIncidents,
  toneOf, zonesOf, type Filters,
} from './incidents'
import type { SafetyEvent, Settings } from './types'

const settings = {
  features: {
    proximity: { ack_required: false, notify: true, min_severity: 'critical' },
    zones: { ack_required: false, notify: false, min_severity: 'warning' },
    ppe: { ack_required: false, notify: false, min_severity: 'warning' },
    fall: { ack_required: true, notify: true, min_severity: 'warning' },
    fire: { ack_required: true, notify: true, min_severity: 'warning' },
    health: { ack_required: false, notify: true, min_severity: 'warning' },
  },
  telegram: { enabled: false, chat_id: '', token_set: false },
  language: 'en',
} as Settings

function ev(over: Partial<SafetyEvent>): SafetyEvent {
  return {
    id: 1, timestamp: '2026-11-09T10:00:00Z', event_type: 'alert', message: 'm', tags: [], camera: 'bay-a@ppe',
    camera_id: 'bay-a', feature: 'ppe', severity: 'warning', incident_status: null, incident_started_at: null,
    incident_duration_seconds: null, clip_id: null, clip_status: null, clips: [], acknowledged_at: null,
    details: null, ...over,
  }
}

const all: Filters = { features: [], camera: null, zone: null, severity: null, status: 'all' }

describe('filters', () => {
  it('passes everything with no filter', () => {
    expect(matches(ev({}), all, settings)).toBe(true)
  })
  it('narrows by feature, camera and severity', () => {
    expect(matches(ev({ feature: 'fire' }), { ...all, features: ['ppe'] }, settings)).toBe(false)
    expect(matches(ev({ camera_id: 'store' }), { ...all, camera: 'bay-a' }, settings)).toBe(false)
    expect(matches(ev({ severity: 'critical' }), { ...all, severity: 'warning' }, settings)).toBe(false)
  })
  it('narrows by zone: the name the alert fired in, whatever its feature', () => {
    const inDock = ev({ details: { zone_name: 'Dock' } })
    expect(matches(inDock, { ...all, zone: 'Dock' }, settings)).toBe(true)
    expect(matches(inDock, { ...all, zone: 'Lane' }, settings)).toBe(false)
    expect(matches(ev({}), { ...all, zone: 'Dock' }, settings)).toBe(false)
    expect(filtersFromQuery({ zone: 'Dock' }).zone).toBe('Dock')
    expect(zonesOf([inDock, ev({ details: { zone_name: 'Lane' } }), ev({}), inDock])).toEqual(['Dock', 'Lane'])
  })
  it('open means an incident still going', () => {
    expect(matches(ev({ incident_status: 'ongoing' }), { ...all, status: 'open' }, settings)).toBe(true)
    expect(matches(ev({ incident_status: 'resolved' }), { ...all, status: 'open' }, settings)).toBe(false)
  })
  it('waiting means a feature that needs an acknowledgement and has none, ended or not', () => {
    const waiting = { ...all, status: 'unacked' as const }
    expect(matches(ev({ feature: 'fall', incident_status: 'resolved' }), waiting, settings)).toBe(true)
    expect(matches(ev({ feature: 'fall', acknowledged_at: '2026-11-09T10:01:00Z' }), waiting, settings)).toBe(false)
    expect(matches(ev({ feature: 'ppe' }), waiting, settings)).toBe(false)
  })
})

describe('incident state', () => {
  it('needs an acknowledgement only for the features set to require one', () => {
    expect(needsAck(ev({ feature: 'fire' }), settings)).toBe(true)
    expect(needsAck(ev({ feature: 'zones' }), settings)).toBe(false)
    expect(needsAck(ev({ feature: null }), settings)).toBe(false)
  })
  it('an ongoing incident lasts until now, an ended one its recorded duration', () => {
    const now = Date.parse('2026-11-09T10:05:00Z')
    expect(durationSeconds(ev({ incident_status: 'ongoing', incident_started_at: '2026-11-09T10:00:00Z' }), now)).toBe(300)
    expect(durationSeconds(ev({ incident_status: 'resolved', incident_duration_seconds: 42 }), now)).toBe(42)
    expect(durationSeconds(ev({}), now)).toBeNull()
    expect(isOpen(ev({ incident_status: 'ongoing' }))).toBe(true)
  })
  it('lists open incidents first, then the newest', () => {
    const list = [ev({ id: 1 }), ev({ id: 2, incident_status: 'ongoing' }), ev({ id: 3 })]
    expect(sortIncidents(list).map(e => e.id)).toEqual([2, 3, 1])
  })
})

describe('the banner list', () => {
  it('merges the server list and live events, keeps the live version, drops what was just acknowledged', async () => {
    const { waitingList } = await import('./incidents')
    const server = [ev({ id: 1, feature: 'fall' }), ev({ id: 2, feature: 'fire' })]
    const live = [ev({ id: 2, feature: 'fire', acknowledged_at: '2026-11-09T10:01:00Z' }), ev({ id: 3, feature: 'fire' }),
      ev({ id: 4, feature: 'ppe' })]
    expect(waitingList(server, live, settings, new Set([1])).map(e => e.id)).toEqual([3])
  })
})

describe('byDay', () => {
  it('groups by local day of the start, today and yesterday named', () => {
    const now = new Date(2026, 10, 9, 15, 0).getTime()
    const at = (d: number, h: number) => new Date(2026, 10, d, h, 0).toISOString()
    const groups = byDay([
      ev({ id: 4, incident_started_at: at(9, 14), timestamp: at(9, 14) }),
      ev({ id: 3, timestamp: at(9, 1) }),
      ev({ id: 2, incident_started_at: at(8, 23), timestamp: at(9, 0) }),
      ev({ id: 1, timestamp: at(6, 9) }),
    ], now)
    expect(groups.map(g => [g.day, g.events.map(e => e.id)])).toEqual([
      ['today', [4, 3]], ['yesterday', [2]], ['2026-11-06', [1]],
    ])
  })
})

describe('filtersFromQuery', () => {
  it('reads the status a link asks for and ignores what it does not know', () => {
    expect(filtersFromQuery({ status: 'unacked' })).toEqual({ ...NO_FILTERS, status: 'unacked' })
    expect(filtersFromQuery({ status: 'bogus', feature: 'fire,nope', camera: 'bay-a' }))
      .toEqual({ ...NO_FILTERS, features: ['fire'], camera: 'bay-a' })
  })
})

describe('when a tile flashes for a new incident', () => {
  it('only for an incident newer than any it has seen: not on first sight, not when the newest one ends', () => {
    expect(newIncident(null, 40)).toEqual({ flash: false, seen: 40 })       // the tile appears with one going on
    expect(newIncident(40, 41)).toEqual({ flash: true, seen: 41 })          // a new one starts
    expect(newIncident(41, 40)).toEqual({ flash: false, seen: 41 })         // 41 ended, 40 still going on
    expect(newIncident(41, null)).toEqual({ flash: false, seen: 41 })       // all ended
    expect(newIncident(41, 41)).toEqual({ flash: false, seen: 41 })
  })
})

describe('the colour a card takes', () => {
  it('is red for every fall incident, sure or for review, like fire', () => {
    expect(toneOf(ev({ feature: 'fall', severity: 'warning' }))).toBe('critical')
    expect(toneOf(ev({ feature: 'fall', severity: 'critical' }))).toBe('critical')
    expect(toneOf(ev({ feature: 'fall', severity: null }))).toBe('critical')
    expect(toneOf(ev({ feature: 'fire', severity: 'critical' }))).toBe('critical')
  })
  it('is the severity for every other feature, grey when there is none', () => {
    expect(toneOf(ev({ feature: 'proximity', severity: 'warning' }))).toBe('warning')
    expect(toneOf(ev({ feature: 'proximity', severity: 'info' }))).toBe('info')
    expect(toneOf(ev({ feature: 'zones', severity: null }))).toBe('info')
    expect(toneOf(ev({ feature: 'ppe', severity: 'high' }))).toBe('info')       // a word the console does not know
  })
})
