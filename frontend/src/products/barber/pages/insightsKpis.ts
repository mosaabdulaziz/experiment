import type { Stats } from '../api/types'
import type { Key } from '../i18n'

export interface Kpi { label: Key; value: string; note?: Key }

/** The four KPIs. With a barber chosen, the haircut figures follow the barber but the average wait and the
 * visits are shop-wide (waits and door crossings have no barber): they come from `shop` (the unfiltered stats)
 * and carry the "all barbers" note. */
export function insightsKpis(s: Stats | null, num: (n: number) => string, timer: (s: number | null | undefined) => string,
                             barber: { shop: Stats | null } | null = null): Kpi[] {
  const n = (v: number | undefined) => (v === undefined ? '—' : num(v))
  const w = barber ? barber.shop : s
  const note = barber ? { note: 'kpi.allBarbers' as Key } : {}
  return [
    { label: 'kpi.served', value: n(s?.served_customers) },
    { label: 'kpi.avgCut', value: s ? timer(s.average_haircut_s) : '—' },
    { label: 'kpi.avgWait', value: w ? timer(w.average_wait_s) : '—', ...note },
    { label: 'kpi.visits', value: n(w?.visitors), ...note },
  ]
}
