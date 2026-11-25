import type { EventFilterState, FilterTimeValue, FilterTypeValue } from '@/hooks/useEventFilters'
import { IconSearch } from '@/components/ui/icons'
import styles from './EventFilters.module.css'

interface Props {
  filters: EventFilterState
  setFilters: (
    update: EventFilterState | ((prev: EventFilterState) => EventFilterState),
  ) => void
  total: number
  visible: number
}

const TYPE_OPTIONS: { value: FilterTypeValue; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'comment', label: 'Comments' },
  { value: 'alert', label: 'Alerts' },
]

const TIME_OPTIONS: { value: FilterTimeValue; label: string }[] = [
  { value: 'all', label: 'All time' },
  { value: '1h', label: 'Last hour' },
  { value: '6h', label: 'Last 6h' },
  { value: '24h', label: 'Last 24h' },
]

export function EventFilters({ filters, setFilters, total, visible }: Props) {
  const setType = (type: FilterTypeValue) => setFilters(prev => ({ ...prev, type }))
  const setQuery = (query: string) => setFilters(prev => ({ ...prev, query }))
  const setTimeRange = (timeRange: FilterTimeValue) =>
    setFilters(prev => ({ ...prev, timeRange }))

  return (
    <div className={styles.bar}>
      <div className={styles.row}>
        <div className={styles.chips} role="tablist" aria-label="Filter events by type">
          {TYPE_OPTIONS.map(opt => (
            <button
              key={opt.value}
              type="button"
              className={`${styles.chip} ${filters.type === opt.value ? styles.chipActive : ''}`}
              onClick={() => setType(opt.value)}
              role="tab"
              aria-selected={filters.type === opt.value}
            >
              {opt.label}
            </button>
          ))}
        </div>

        <select
          className="iris-select"
          value={filters.timeRange}
          onChange={e => setTimeRange(e.target.value as FilterTimeValue)}
          aria-label="Filter events by time range"
        >
          {TIME_OPTIONS.map(o => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      </div>

      <div className={styles.row}>
        <div className={styles.search}>
          <IconSearch className={styles.searchIcon} />
          <input
            type="text"
            placeholder="Search messages and tags…"
            className={styles.searchInput}
            value={filters.query}
            onChange={e => setQuery(e.target.value)}
            spellCheck={false}
          />
        </div>
        <span className={styles.count}>
          {visible} / {total}
        </span>
      </div>
    </div>
  )
}
