import { describe, expect, it } from 'vitest'
import { parseRoute } from './route'

describe('routes', () => {
  it('splits the page from its query, live by default', () => {
    expect(parseRoute('/incidents?status=unacked')).toEqual({ page: 'incidents', query: { status: 'unacked' } })
    expect(parseRoute('/')).toEqual({ page: 'live', query: {} })
    expect(parseRoute('/nowhere')).toEqual({ page: 'live', query: {} })
    expect(parseRoute('/health')).toEqual({ page: 'health', query: {} })
  })
  it('the old Cameras page lands on Live, where each tile configures its camera', () => {
    expect(parseRoute('/cameras')).toEqual({ page: 'live', query: {} })
  })
})
