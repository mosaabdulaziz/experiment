import { afterEach, describe, expect, it, vi } from 'vitest'
import { candidates, cameras, events, review, status } from './api'

afterEach(() => { vi.unstubAllGlobals() })

type Call = { url: string; init: RequestInit }

function stubFetch(): Call[] {
  const calls: Call[] = []
  vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
    calls.push({ url, init: init ?? {} })
    return { ok: true, status: 200, json: async () => [] } as unknown as Response
  })
  return calls
}

describe('the calls the tab makes', () => {
  it('each one asks the product\'s own service under its own prefix', async () => {
    const calls = stubFetch()
    await status(); await cameras(); await candidates()
    expect(calls.map(c => c.url)).toEqual([
      '/fall_detection/api/status',
      '/fall_detection/api/cameras',
      '/fall_detection/api/candidates',
    ])
    // a refused read must name the path and the status
    vi.stubGlobal('fetch', async () => (
      { ok: false, status: 503, json: async () => ({}) } as unknown as Response
    ))
    await expect(status()).rejects.toThrow(/\/status.*503/)
  })

  it('the events call carries the cursor', async () => {
    const calls = stubFetch()
    await events(12)
    expect(calls.map(c => c.url)).toEqual(['/fall_detection/api/events?since=12'])
  })

  it('the events call names the camera when there is one', async () => {
    const calls = stubFetch()
    await events(12, 'cam 1')
    expect(calls.map(c => c.url)).toEqual(['/fall_detection/api/events?since=12&camera=cam%201'])
  })

  it('the verdict is posted to the review row', async () => {
    const calls = stubFetch()
    await review(7, 'acknowledged')
    expect(calls.map(c => c.url)).toEqual(['/fall_detection/api/review/7'])
    // the write is a POST carrying the verdict itself, as JSON
    const init = calls[0].init
    expect(init.method).toBe('POST')
    expect(JSON.parse(String(init.body))).toEqual({ status: 'acknowledged' })
    expect((init.headers as Record<string, string> | undefined)?.['Content-Type']).toBe('application/json')
    // an aborted write must not read as a plain failure: the verdict may already be committed
    vi.stubGlobal('fetch', async () => {
      throw Object.assign(new Error('the request timed out'), { name: 'TimeoutError' })
    })
    await expect(review(7, 'acknowledged')).rejects.toThrow(/may have landed/)
  })
})
