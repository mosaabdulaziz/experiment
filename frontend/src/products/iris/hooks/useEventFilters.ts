import { useMemo, useState } from 'react'
import type { IrisEvent, EventType } from '@/types/event'

export type FilterTypeValue = 'all' | EventType
export type FilterTimeValue = 'all' | '1h' | '6h' | '24h'

export interface EventFilterState {
  type: FilterTypeValue
  query: string
  timeRange: FilterTimeValue
}

const TIME_WINDOW_MS: Record<FilterTimeValue, number | null> = {
  all: null,
  '1h': 60 * 60 * 1000,
  '6h': 6 * 60 * 60 * 1000,
  '24h': 24 * 60 * 60 * 1000,
}

export function useEventFilters(events: IrisEvent[]) {
  const [filters, setFilters] = useState<EventFilterState>({
    type: 'all',
    query: '',
    timeRange: 'all',
  })

  const filtered = useMemo(() => {
    const cutoff = TIME_WINDOW_MS[filters.timeRange]
    const cutoffTs = cutoff ? Date.now() - cutoff : null
    const q = filters.query.trim().toLowerCase()

    return events.filter(e => {
      if (filters.type !== 'all' && e.event_type !== filters.type) return false
      if (cutoffTs !== null) {
        const ts = new Date(e.timestamp).getTime()
        if (Number.isNaN(ts) || ts < cutoffTs) return false
      }
      if (q) {
        const hay = `${e.message} ${e.tags.join(' ')}`.toLowerCase()
        if (!hay.includes(q)) return false
      }
      return true
    })
  }, [events, filters])

  return { filters, setFilters, filtered }
}
