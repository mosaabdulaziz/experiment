import { describe, expect, it } from 'vitest'
import { entryRows, servedChairs, visitRows } from './customerEntries'

describe('entryRows', () => {
  it('keeps the API order and splits date and time', () => {
    const rows = entryRows([{ id: 'e2', at: '2026-10-05T17:20:05' }, { id: 'e1', at: '2026-10-01T09:03:00' }], 'en')
    expect(rows.map(r => r.id)).toEqual(['e2', 'e1'])
    expect(rows[0].time).toBe('17:20')
  })
  it('is empty without entries', () => { expect(entryRows([], 'en')).toEqual([]) })
})

// 2026-10-18 (user): every visit says at which chair he was served
const NAMES: Record<string, string> = { 'ph5-R1': 'R2', L6: 'L6' }
const name = (id: string) => NAMES[id] ?? id

describe('visitRows', () => {
  it('lists the visits newest first with their chairs by the shop name, each once', () => {
    const rows = visitRows([
      { id: 'v1', entered_at: '2026-10-01T09:03:00', haircut_sessions: [{ chair_id: 'L6' }] },
      { id: 'v2', entered_at: '2026-10-05T17:20:05',
        haircut_sessions: [{ chair_id: 'ph5-R1' }, { chair_id: 'L7' }, { chair_id: 'ph5-R1' }] },
      { id: 'v3', entered_at: '2026-10-03T12:00:00', haircut_sessions: [] },
    ], 'en', name)
    expect(rows.map(r => [r.id, r.time, r.chairs])).toEqual([['v2', '17:20', ['R2', 'L7']], ['v3', '12:00', []],
                                                             ['v1', '09:03', ['L6']]])
  })
  it('is empty without visits (an older backend sends none)', () => {
    expect(visitRows(undefined, 'en', name)).toEqual([])
  })
})

describe('servedChairs', () => {
  it("names each served visit's chairs, newest first, and skips visits not served", () => {
    expect(servedChairs([{ at: 'b', chairs: ['ph5-R1', 'L7'] }, { at: 'a', chairs: [] }, { at: '0', chairs: ['L6'] }],
                        name)).toEqual(['R2 + L7', 'L6'])
    expect(servedChairs(undefined, name)).toEqual([])
  })
})
