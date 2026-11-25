import { useEffect } from 'react'
import { useStreamsStore } from '@/stores/streamsStore'
import { useEventsForCamera } from '@/hooks/useEventsForCamera'
import { EventLog } from '@/components/events/EventLog'
import { formatOffset } from '@/lib/format'
import { IconClose } from '@/components/ui/icons'
import { useFramePoll } from './framePoll'
import styles from './ExpandedStream.module.css'

export function ExpandedStream() {
  const expandedId = useStreamsStore(s => s.expandedId)
  const streams = useStreamsStore(s => s.streams)
  const setExpanded = useStreamsStore(s => s.setExpanded)
  const camEvents = useEventsForCamera(expandedId)

  // Close on Escape while the overlay is open.
  useEffect(() => {
    if (!expandedId) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setExpanded(null)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [expandedId, setExpanded])

  const stream = streams.find(s => s.id === expandedId)
  const live = stream?.running ?? false
  const showTime = stream?.is_file && stream?.offset != null

  const frames = useFramePoll(expandedId ?? '', live, 50)

  if (!expandedId || !stream) return null

  return (
    <div className={styles.backdrop} onClick={() => setExpanded(null)}>
      <div
        className={styles.panel}
        role="dialog"
        aria-modal="true"
        aria-label={`${stream.id} detail`}
        onClick={e => e.stopPropagation()}
      >
        <header className={styles.header}>
          <div className={styles.headLeft}>
            <span className={styles.id}>{stream.id}</span>
            <span className={`${styles.status} ${live ? styles.statusLive : styles.statusOff}`}>
              <span className={styles.dot} />
              {live ? 'Live' : 'Off'}
            </span>
            <span className={styles.source}>{stream.source}</span>
          </div>
          <button
            type="button"
            className={styles.close}
            onClick={() => setExpanded(null)}
            aria-label="Close detail view"
          >
            <IconClose /> Close
          </button>
        </header>

        <div className={styles.body}>
          <div className={styles.video}>
        {/* No key on the <img>: remounting the <img> every poll blanks the view
            while the next JPEG decodes; swapping src keeps the last frame. */}
        <img
          className={styles.stream}
          src={frames.src}
          alt={`${stream.id} feed`}
          onLoad={frames.onLoad}
          onError={frames.onError}
        />
            {showTime && <div className={styles.timeBadge}>{formatOffset(stream.offset as number)}</div>}
            {!live && (
              <div className={styles.offScrim}>
                <span>Stream offline</span>
              </div>
            )}
          </div>
          <aside className={styles.events}>
            <div className={styles.eventsHead}>Events · {stream.id}</div>
            <EventLog events={camEvents} filtered={camEvents} />
          </aside>
        </div>
      </div>
    </div>
  )
}
