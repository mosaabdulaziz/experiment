import { describe, expect, it } from 'vitest'
import { evaluationLabel, pct, seconds, severityChip, statusChip } from './format'

describe('the formatters', () => {
  it('a clock is the sidecar\'s frame time', () => {
    expect(seconds(0)).toBe('1970-01-01 00:00:00')
    expect(seconds(null)).toBe('—')
  })

  it('a score keeps two decimals', () => {
    expect(pct(0.812)).toBe('0.81')
    expect(pct(null)).toBe('—')
    expect(pct(undefined)).toBe('—')
  })

  it('the severity picks the house chip', () => {
    expect(severityChip('critical')).toBe('chipRed')
    expect(severityChip('warning')).toBe('chipAmber')
    expect(severityChip(null)).toBe('chipGreen')
  })

  it('the verdict picks the chip that tells the queue apart', () => {
    expect(statusChip('candidate')).toBe('chip chipAmber')
    expect(statusChip('confirmed')).toBe('chip chipGreen')
    // A rejection must not look like a confirmation -- that was the defect: both drew green.
    expect(statusChip('dismissed')).toBe('chip')
    // Seen without a verdict is no verdict either.
    expect(statusChip('acknowledged')).toBe('chip')
  })

  it('the evaluation reads as a word', () => {
    expect(evaluationLabel('falling')).toBe('falling')
    expect(evaluationLabel('upright')).toBe('upright')
  })
})
