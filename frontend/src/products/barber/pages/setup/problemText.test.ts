import { describe, expect, it } from 'vitest'
import type { ZoneProblem } from '../../api/types'
import { buildAreas } from '../../floor/areas'
import { translate, type Key } from '../../i18n'
import { problemText } from './problemText'

const en = (key: Key, vars?: Record<string, string | number>) => translate('en', key, vars)
const ar = (key: Key, vars?: Record<string, string | number>) => translate('ar', key, vars)
const areas = buildAreas({})

const PROBLEMS: [ZoneProblem, string][] = [
  [{ code: 'face_area_unknown_chair', camera: 'barber-c', zone: 'R1 face' }, 'Face area R1 face on Centre names a chair that does not exist'],
  [{ code: 'pair_single_camera', pair: 'L3', camera: 'barber-c' }, 'L3 is drawn only in Centre'],
  [{ code: 'pair_twice_in_camera', pair: 'R2' }, 'R2 is drawn twice in one camera'],
  [{ code: 'pair_owner_count', pair: 'L1', count: 2 }, 'L1 needs exactly one owner camera (has 2)'],
  [{ code: 'chair_without_seat', camera: 'barber-r', zone: 'L3' }, 'Chair L3 in Right has no seat'],
  [{ code: 'door_area_ignored', camera: 'barber-l' }, 'The door area in Left is ignored: only the camera with the entrance line detects faces'],
]

describe('problemText', () => {
  it.each(PROBLEMS)('reads %o in English', (p, text) => {
    expect(problemText(p, en, areas)).toBe(text)
  })
  it('names cameras by their label, else by id', () => {
    expect(problemText({ code: 'pair_single_camera', pair: 'L3', camera: 'barber-l' }, en, areas)).toBe('L3 is drawn only in Left')
    expect(problemText({ code: 'pair_single_camera', pair: 'L3', camera: 'barber-9' }, en, areas)).toBe('L3 is drawn only in barber-9')
  })
  it('uses the area name when the area has one', () => {
    const named = buildAreas({ 'barber-c': [{ id: 'c-w', name: 'Sink', type: 'washing', pair: 'wash', owner: true, points: [[0, 0], [1, 0], [1, 1]] }] })
    expect(problemText({ code: 'pair_twice_in_camera', pair: 'wash' }, en, named)).toBe('Sink is drawn twice in one camera')
  })
  it.each(PROBLEMS)('leaves no placeholder in Arabic for %o', p => {
    const text = problemText(p, ar, areas)
    expect(text).not.toContain('{')
    expect(text.length).toBeGreaterThan(0)
  })
})
