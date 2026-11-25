import { describe, expect, it } from 'vitest'
import { insightsKpis } from './insightsKpis'

describe('insightsKpis', () => {
  it('shows served customers, average haircut, average wait and visits', () => {
    const k = insightsKpis({ served_customers: 9, average_haircut_s: 693, average_wait_s: 120, visitors: 13 } as never,
                           n => String(n), s => (s == null ? '—' : `${s}s`))
    expect(k.map(x => [x.label, x.value])).toEqual([
      ['kpi.served', '9'], ['kpi.avgCut', '693s'], ['kpi.avgWait', '120s'], ['kpi.visits', '13'],
    ])
  })
  it('dashes without stats', () => {
    expect(insightsKpis(null, n => String(n), () => '—').map(x => x.value)).toEqual(['—', '—', '—', '—'])
  })
})

describe('insightsKpis with a barber chosen', () => {
  it('takes the average wait and visits from the shop-wide stats and notes it', () => {
    const mine = { served_customers: 3, average_haircut_s: 600, average_wait_s: null, visitors: 0 } as never
    const shop = { served_customers: 9, average_haircut_s: 693, average_wait_s: 120, visitors: 13 } as never
    const k = insightsKpis(mine, n => String(n), s => (s == null ? '—' : `${s}s`), { shop })
    expect(k.map(x => [x.label, x.value, x.note])).toEqual([
      ['kpi.served', '3', undefined], ['kpi.avgCut', '600s', undefined],
      ['kpi.avgWait', '120s', 'kpi.allBarbers'], ['kpi.visits', '13', 'kpi.allBarbers'],
    ])
  })
  it('dashes the shop-wide figures until they load', () => {
    const k = insightsKpis({ served_customers: 3 } as never, n => String(n), () => '—', { shop: null })
    expect([k[2].value, k[3].value]).toEqual(['—', '—'])
  })
})
