import { describe, expect, it } from 'vitest'
import {
  assessmentReasonLabel,
  durationSeconds,
  pct,
  seconds,
  severityChip,
  shortTimestamp,
  stageChainLabel,
  statusChip,
  targetLabel,
  tierChip,
  timestamp,
} from './format'

describe('format helpers', () => {
  it('formats frame PTS offsets, ISO timestamps, and durations cleanly without 1970 epoch leaks', () => {
    expect(seconds(0)).toBe('+00:00')
    expect(seconds(127.4)).toBe('+02:07')
    expect(seconds(3665)).toBe('+01:01:05')
    expect(seconds(1_763_055_655)).toBe('2025-11-13 17:40:55')
    expect(seconds(null)).toBe('—')
    expect(timestamp('2026-11-13T17:40:55.123456+00:00', 12.3)).toBe('2026-11-13 17:40:55')
    expect(timestamp('1970-01-01T00:06:35.750000+00:00', 395.75)).toBe('+06:35')
    expect(shortTimestamp('2026-11-13T17:40:55.123456+00:00', 12.3)).toBe('17:40:55')
    expect(durationSeconds(3.24)).toBe('3.2s')
    expect(durationSeconds(null)).toBe('—')
  })

  it('formats assessment reason codes into readable wrapping phrases', () => {
    expect(assessmentReasonLabel('low_core_keypoint_confidence')).toBe('low core keypoint confidence')
    expect(assessmentReasonLabel('insufficient_visible_keypoints,low_core_keypoint_confidence')).toBe(
      'insufficient visible keypoints, low core keypoint confidence',
    )
    expect(assessmentReasonLabel('')).toBe('insufficient evidence')
  })

  it('formats score to two decimals', () => {
    expect(pct(0.836)).toBe('0.84')
    expect(pct(null)).toBe('—')
  })

  it('maps severity, tier, and status to house chip classes', () => {
    expect(severityChip('critical')).toBe('chipRed')
    expect(severityChip('warning')).toBe('chipAmber')
    expect(severityChip('info')).toBe('chipGreen')
    expect(tierChip('high')).toBe('chip chipRed')
    expect(tierChip('medium')).toBe('chip chipAmber')
    expect(tierChip('low')).toBe('chip')
    expect(statusChip('candidate')).toBe('chip chipAmber')
    expect(statusChip('concern_noted')).toBe('chip chipRed')
    expect(statusChip('acknowledged')).toBe('chip chipGreen')
    expect(statusChip('dismissed')).toBe('chip')
  })

  it('formats target and stage progression clearly', () => {
    expect(targetLabel('personal_bag', null)).toBe('personal bag')
    expect(targetLabel('clothing', 'under_arm')).toBe('clothing (under arm)')
    expect(
      stageChainLabel(
        [
          { stage: 'DWELL', entered_pts: 1.0 },
          { stage: 'REACH', entered_pts: 2.0 },
          { stage: 'AT_BODY', entered_pts: 3.1 },
        ],
        'RETURNED',
      ),
    ).toBe('DWELL -> REACH -> AT_BODY -> RETURNED')
  })
})
