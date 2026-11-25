import { useCallback, useEffect, useRef, useState } from 'react'
import type { IrisEvent } from '@/types/event'
import type { ClipInfo } from '@/types/chat'
import { EventCard } from './EventCard'
import { ClipPreviewModal } from '@/components/chat/ClipPreviewModal'
import { IconEye } from '@/components/ui/icons'
import styles from './EventLog.module.css'

interface Props {
  events: IrisEvent[]
  filtered: IrisEvent[]
}

export function EventLog({ events, filtered }: Props) {
  const listRef = useRef<HTMLUListElement>(null)
  const prevLenRef = useRef(events.length)
  const [previewClip, setPreviewClip] = useState<ClipInfo | null>(null)

  // One modal for the whole list rather than one per card — only ever one is open,
  // and a stable callback keeps EventCard's memo effective.
  const handlePreview = useCallback((event: IrisEvent) => {
    if (!event.clip_id || !event.camera) return
    setPreviewClip({ id: event.clip_id, cameras: [event.camera] })
  }, [])

  // Auto-scroll to bottom on new events
  useEffect(() => {
    if (events.length > prevLenRef.current) {
      const el = listRef.current
      if (el) el.scrollTop = el.scrollHeight
    }
    prevLenRef.current = events.length
  }, [events.length])

  if (events.length === 0) {
    return (
      <div className={styles.view}>
        <div className={styles.empty}>
          <div className={styles.emptyIcon} aria-hidden="true">
            <IconEye width={36} height={36} stroke="#f9996c" opacity={0.4} />
          </div>
          <p className={styles.emptyTitle}>Watching…</p>
          <p className={styles.emptyBody}>
            The AI will post observations
            <br />
            as events occur in the feed.
          </p>
        </div>
      </div>
    )
  }

  if (filtered.length === 0) {
    return (
      <div className={styles.view}>
        <div className={styles.empty}>
          <p className={styles.emptyTitle}>No matches</p>
          <p className={styles.emptyBody}>
            Try adjusting your filters
            <br />
            or clearing the search.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className={styles.view}>
      <ul ref={listRef} className={styles.list}>
        {filtered.map(e => (
          <EventCard key={e.id} event={e} onPreview={handlePreview} />
        ))}
      </ul>
      <ClipPreviewModal
        clip={previewClip}
        basePath="/api/alerts/clip"
        title="Incident preview"
        onClose={() => setPreviewClip(null)}
      />
    </div>
  )
}
