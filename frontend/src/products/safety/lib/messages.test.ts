import { describe, expect, it } from 'vitest'
import { translate, type Key } from '../i18n'
import { messageOf } from './messages'
import type { SafetyEvent } from './types'

const ev = (over: Partial<SafetyEvent>): SafetyEvent => ({
  id: 1, timestamp: '2026-11-10T10:00:00Z', event_type: 'alert', message: 'server words', tags: [],
  camera: 'bay-a@ppe', camera_id: 'bay-a', feature: 'ppe', severity: 'warning', incident_status: null,
  incident_started_at: null, incident_duration_seconds: null, clip_id: null, clip_status: null, clips: [],
  acknowledged_at: null, details: null, ...over,
})
const ar = (e: SafetyEvent) => messageOf(e, 'ar', (k: Key, v?: Record<string, string | number>) => translate('ar', k, v),
  id => ({ 'bay-a': 'Bay A', store: 'Store room', office: 'Office' }[id ?? ''] ?? String(id)))

describe('an incident message in the console language', () => {
  it('in English it is the app own words', () => {
    expect(messageOf(ev({ details: { missing: 'helmet', track_id: '12' } }), 'en', k => k, id => String(id)))
      .toBe('server words')
  })
  it('names the camera where the app wrote its stream key (office@zones), in English and in a fallback', () => {
    const names = (id: string | null) => ({ office: 'Office', store: 'Store room' }[id ?? ''] ?? String(id))
    expect(messageOf(ev({ feature: 'zones', camera: 'office@zones', camera_id: 'office', tags: ['zone_overcrowding'],
      message: 'Overcrowding detected on office@zones — Zone 1 (4 people)' }), 'en', k => k, names))
      .toBe('Overcrowding detected on Office — Zone 1 (4 people)')
    expect(messageOf(ev({ feature: 'fire', camera: 'store@fire', camera_id: 'store', message: 'Something on store@fire' }),
      'ar', k => k, names)).toBe('Something on Store room')
    expect(messageOf(ev({ camera: 'store', camera_id: 'store', message: 'No video from store' }), 'en', k => k, names))
      .toBe('No video from store')
  })
  it('PPE: who, and what they are missing, the duplicate hint too', () => {
    expect(ar(ev({ details: { missing: 'helmet,vest', track_id: '12' } }))).toBe('الشخص #١٢ بدون خوذة وسترة عاكسة')
    expect(ar(ev({ details: { missing: 'helmet', track_id: '7', possible_duplicate_of: '5' },
      message: 'Person #7 without helmet on bay-a (possibly #5 again, alerted 9 s earlier)' })))
      .toBe('الشخص #٧ بدون خوذة (ربما #٥ مجدداً، قبل ٩ ث)')
  })
  it('falls: the evaluation, sure or to be reviewed, and the confidence', () => {
    expect(ar(ev({ feature: 'fall', details: { evaluation: 'falling', confidence: '0.82', review: '0' } })))
      .toBe('سقط شخص (الثقة ٨٢٪)')
    expect(ar(ev({ feature: 'fall', details: { evaluation: 'lying', confidence: '0.1', review: '1' } })))
      .toBe('ربما شخص ممدد على الأرض: يحتاج إلى مراجعة (الثقة ١٠٪)')
  })
  it('fire and smoke, by the class the event keeps (or the app sentence for an older one)', () => {
    expect(ar(ev({ feature: 'fire', camera_id: 'store', details: { class: 'smoke' } }))).toBe('رُصد دخان في Store room')
    expect(ar(ev({ feature: 'fire', camera_id: 'store', message: 'Fire detected on store@fire' })))
      .toBe('رُصد حريق في Store room')
  })
  it('zones: who entered or left which zone, and overcrowding with its count', () => {
    expect(ar(ev({ feature: 'zones', tags: ['zone_intrusion'],
      details: { object_id: '12', object_label: 'person', zone_name: 'Lab door' } }))).toBe('دخل شخص #١٢ المنطقة Lab door')
    expect(ar(ev({ feature: 'zones', tags: ['zone_exit'], details: { object_label: 'car', zone_name: 'Yard' } })))
      .toBe('خرج سيارة من المنطقة Yard')
    expect(ar(ev({ feature: 'zones', tags: ['zone_overcrowding'], camera_id: 'office',
      message: 'Overcrowding detected on office@zones — Zone 1 (4 people)' }))).toBe('ازدحام في Zone 1 (٤ أشخاص)')
  })
  it('proximity: the gap, the vehicle and the zone; a pair the depth check held back says so', () => {
    const d = { gap: '0.55', vehicle_class: 'truck', zone_name: 'Zone 1' }
    expect(ar(ev({ feature: 'proximity', details: d }))).toBe('شخص على بعد ٠٫٥٥ من طول الشخص من شاحنة متحركة في Zone 1')
    expect(ar(ev({ feature: 'proximity', tags: ['proximity', 'depth-suppressed'], details: { ...d, suppressed: 'depth' } })))
      .toBe('استبعده فحص العمق: شخص على بعد ٠٫٥٥ من طول الشخص من شاحنة متحركة في Zone 1')
  })
  it('camera health: the camera and its problem', () => {
    expect(ar(ev({ feature: 'health', details: { problem: 'offline' } }))).toBe('Bay A: لا يوجد فيديو')
  })
  it('keeps the app words when the event keeps too little to say it', () => {
    expect(ar(ev({ feature: 'fall', details: { person_ids: '3' } }))).toBe('server words')
    expect(ar(ev({ feature: 'ppe', details: null }))).toBe('server words')
  })
})
