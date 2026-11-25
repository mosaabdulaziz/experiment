import { useCallback, useMemo, useState } from 'react'
import type { IrisEvent } from '@/types/event'
import type { ClipInfo } from '@/types/chat'
import type { AlertKind } from '@/lib/clips'
import { groupByDay } from '@/lib/clips'
import { matchesIncidentFilters } from '@/lib/incidentFilters'
import { isDepthSuppressed } from '@/lib/proximity'
import { useAlertIncidents } from '@/hooks/useAlertIncidents'
import { useConfigStore } from '@/stores/configStore'
import { IncidentCard } from '@/components/incidents/IncidentCard'
import { IncidentFilters } from '@/components/incidents/IncidentFilters'
import { ClipPreviewModal } from '@/components/chat/ClipPreviewModal'
import styles from './IncidentsPage.module.css'

function toggle<T>(list: T[], value: T): T[] {
  return list.includes(value) ? list.filter(v => v !== value) : [...list, value]
}

export function IncidentsPage() {
  const { incidents, loading, error, reload } = useAlertIncidents()
  const ttl = useConfigStore(s => s.clipTtlSeconds)
  const [activeCameras, setActiveCameras] = useState<string[]>([])
  const [activeKinds, setActiveKinds] = useState<AlertKind[]>([])
  const [footageOnly, setFootageOnly] = useState(false)
  const [heldBack, setHeldBack] = useState(false)
  const [previewClip, setPreviewClip] = useState<ClipInfo | null>(null)

  const cameras = useMemo(() => {
    const set = new Set<string>()
    for (const ev of incidents) if (ev.camera) set.add(ev.camera)
    return [...set].sort()
  }, [incidents])

  // An empty selection means "no filter", so the page opens showing every alert (lib/incidentFilters.ts); pairs the
  // depth check held back only with their own filter on.
  const filtered = useMemo(
    () =>
      incidents.filter(ev =>
        matchesIncidentFilters(ev, { cameras: activeCameras, kinds: activeKinds, footageOnly, heldBack }, ttl)),
    [incidents, activeCameras, activeKinds, footageOnly, heldBack, ttl],
  )
  const pool = useMemo(() => incidents.filter(ev => isDepthSuppressed(ev) === heldBack).length, [incidents, heldBack])

  const groups = useMemo(() => groupByDay(filtered), [filtered])
  const filtersActive = activeCameras.length > 0 || activeKinds.length > 0 || footageOnly || heldBack

  const handlePreview = useCallback((event: IrisEvent) => {
    if (!event.clip_id || !event.camera) return
    setPreviewClip({ id: event.clip_id, cameras: [event.camera] })
  }, [])

  const clearFilters = () => {
    setActiveCameras([])
    setActiveKinds([])
    setFootageOnly(false)
    setHeldBack(false)
  }

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <div>
          <div className={styles.title}>Incidents</div>
          <div className={styles.subtitle}>
            {loading
              ? 'Loading…'
              : `${filtered.length} of ${pool} ${heldBack ? 'held back by depth' : `alert${pool === 1 ? '' : 's'}`}`}
          </div>
        </div>
      </div>

      {error && (
        <div className={styles.error}>
          <span>{error}</span>
          <button type="button" className={styles.retry} onClick={reload}>
            Retry
          </button>
        </div>
      )}

      <IncidentFilters
        cameras={cameras}
        activeCameras={activeCameras}
        activeKinds={activeKinds}
        footageOnly={footageOnly}
        heldBack={heldBack}
        onToggleCamera={c => setActiveCameras(prev => toggle(prev, c))}
        onToggleKind={k => setActiveKinds(prev => toggle(prev, k))}
        onToggleFootageOnly={() => setFootageOnly(v => !v)}
        onToggleHeldBack={() => setHeldBack(v => !v)}
      />

      {!loading && incidents.length === 0 && (
        <div className={styles.empty}>No incidents recorded yet.</div>
      )}

      {!loading && incidents.length > 0 && filtered.length === 0 && (
        <div className={styles.empty}>
          <span>No incidents match these filters.</span>
          {filtersActive && (
            <button type="button" className={styles.retry} onClick={clearFilters}>
              Clear filters
            </button>
          )}
        </div>
      )}

      {groups.map(group => (
        <section key={group.key} className={styles.daySection}>
          <div className={styles.dayHeader}>
            <span className={styles.dayLabel}>{group.label}</span>
            <span className={styles.dayRule} />
            <span className={styles.dayCount}>{group.items.length}</span>
          </div>
          <div className={styles.grid}>
            {group.items.map(ev => (
              <IncidentCard key={ev.id} event={ev} onPreview={handlePreview} />
            ))}
          </div>
        </section>
      ))}

      <ClipPreviewModal
        clip={previewClip}
        basePath="/api/alerts/clip"
        title="Incident preview"
        onClose={() => setPreviewClip(null)}
      />
    </div>
  )
}
