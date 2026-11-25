import { afterEach, describe, expect, it, vi } from 'vitest'
import { assessments, cameras, candidates, review, status } from './api'

type Call = { url: string; init?: RequestInit }

function mockFetch(body: unknown, code = 200): Call[] {
  const calls: Call[] = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({ url, init })
      return new Response(code === 204 ? null : JSON.stringify(body), {
        status: code,
        headers: { 'Content-Type': 'application/json' },
      })
    }),
  )
  return calls
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('concealment_detection api client', () => {
  it('calls /concealment_detection/api/status', async () => {
    const calls = mockFetch({
      running: true,
      ready: true,
      cameras: { aisle_4: { camera_ok: true, reason: 'ok' } },
      capabilities: {
        cart_detection: false,
        bag_context_requires_boxed_detections: true,
        durable_clips: false,
      },
    })
    const s = await status()
    expect(s.ready).toBe(true)
    expect(calls[0].url).toBe('/concealment_detection/api/status')
  })

  it('normalizes top-level or wrapped /cameras lists', async () => {
    const calls = mockFetch([
      { id: 'aisle_4', location: 'Aisle 4 Cosmetics' },
      'cam_2',
    ])
    const cams = await cameras()
    expect(cams).toEqual([
      { id: 'aisle_4', location: 'Aisle 4 Cosmetics' },
      { id: 'cam_2', location: 'cam_2' },
    ])
    expect(calls[0].url).toBe('/concealment_detection/api/cameras')
  })

  it('passes camera, review_status, since, limit, and before to /candidates', async () => {
    const calls = mockFetch([])
    await candidates({
      camera: 'aisle_4',
      review_status: 'candidate',
      since: 12,
      limit: 25,
      before: 99,
    })
    expect(calls[0].url).toBe(
      '/concealment_detection/api/candidates?camera=aisle_4&review_status=candidate&since=12&limit=25&before=99',
    )
  })

  it('calls /assessments with optional camera and limit', async () => {
    const calls = mockFetch([])
    await assessments({ camera: 'aisle_4', limit: 20 })
    expect(calls[0].url).toBe('/concealment_detection/api/assessments?camera=aisle_4&limit=20')
  })

  it('posts non-accusatory review verdicts to /review/:id', async () => {
    const calls = mockFetch({ id: 7, status: 'concern_noted' })
    await review(7, 'concern_noted', { note: 'verify clip', reviewer: 'op_1' })
    expect(calls[0].url).toBe('/concealment_detection/api/review/7')
    expect(calls[0].init?.method).toBe('POST')
    expect(JSON.parse(String(calls[0].init?.body))).toEqual({
      status: 'concern_noted',
      note: 'verify clip',
      reviewer: 'op_1',
    })
  })

  it('surfaces backend detail on HTTP errors', async () => {
    mockFetch({ detail: 'candidate not found' }, 404)
    await expect(review(99, 'dismissed')).rejects.toThrow('candidate not found')
  })
})
