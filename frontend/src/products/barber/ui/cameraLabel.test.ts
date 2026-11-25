import { describe, expect, it } from 'vitest'
import { translate } from '../i18n'
import { cameraLabel, rememberCameraNames } from './cameraLabel'

describe('cameraLabel', () => {
  it('names the three shop cameras and falls back to the id', () => {
    const en = (k: Parameters<typeof translate>[1]) => translate('en', k)
    const ar = (k: Parameters<typeof translate>[1]) => translate('ar', k)
    expect(['barber-c', 'barber-l', 'barber-r', 'cam-9'].map(c => cameraLabel(c, en))).toEqual(['Centre', 'Left', 'Right', 'cam-9'])
    expect(cameraLabel('barber-c', ar)).toBe('الوسط')
  })
})

it('uses the configured camera name, then the translated id, then the id', () => {
  const t = (k: string) => `T:${k}`
  rememberCameraNames([{ id: 'ph-1', name: 'Wash area' }, { id: 'ph-2', name: null }])
  expect(cameraLabel('ph-1', t as never)).toBe('Wash area')
  expect(cameraLabel('ph-2', t as never)).toBe('ph-2')
  expect(cameraLabel('barber-c', t as never)).toBe('T:cam.centre')
  rememberCameraNames([])
  expect(cameraLabel('ph-1', t as never)).toBe('ph-1')
})
