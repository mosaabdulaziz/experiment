import { memo } from 'react'
import { SEVERITY_LABEL } from '@/lib/proximity'
import type { IrisEvent } from '@/types/event'
import { formatOffsetRange, formatTime } from '@/lib/format'
import { alertKind, clipState } from '@/lib/clips'
import { useConfigStore } from '@/stores/configStore'
import { IconPlay } from '@/components/ui/icons'
import styles from './EventCard.module.css'

interface Props {
  event: IrisEvent
  onPreview?: (event: IrisEvent) => void
}

function EventCardBase({ event, onPreview }: Props) {
  const isAlert = event.event_type === 'alert'
  const offsetRange = formatOffsetRange(event.stream_offset_start, event.stream_offset_end)
  const descStatus = event.description_status ?? 'none'
  const showSplit = isAlert && descStatus !== 'none'
  const ttl = useConfigStore(s => s.clipTtlSeconds)
  const clip = clipState(event, ttl)

  return (
    <li className={`${styles.card} ${isAlert ? styles.alert : styles.comment}`}>
      <div className={styles.header}>
        <span
          className={`${styles.typeBadge} ${
            isAlert ? styles.typeBadgeAlert : styles.typeBadgeComment
          }`}
        >
          {event.event_type}
        </span>
        {event.camera && <span className={styles.cameraChip}>{event.camera}</span>}
        {alertKind(event) === 'proximity' && (
          <span className={`${styles.cameraChip} ${styles[`sev_${event.severity ?? 'warning'}`]}`}>
            Proximity · {SEVERITY_LABEL[event.severity ?? 'warning']}
          </span>
        )}
        <span className={styles.meta}>
          {offsetRange && <span className={styles.offset}>{offsetRange}</span>}
          <span className={styles.time}>{formatTime(event.timestamp)}</span>
        </span>
      </div>
      {showSplit ? (
        <>
          <div className={styles.title} dir="auto">
            {event.message}
          </div>
          <div className={styles.message} dir="auto">
            {descStatus === 'pending' && (
              <span className={styles.generating}>
                <span className={styles.badgeDot} aria-hidden="true" />
                Generating description…
              </span>
            )}
            {descStatus === 'ready' && (event.description || 'No description available.')}
          </div>
        </>
      ) : (
        <div className={styles.message} dir="auto">
          {event.message}
        </div>
      )}
      {event.tags.length > 0 && (
        <div className={styles.tags}>
          {event.tags.map(t => (
            <span key={t} className={styles.tag}>
              {t}
            </span>
          ))}
        </div>
      )}
      {clip !== 'none' && (
        <div className={styles.clipRow}>
          {clip === 'ready' && (
            <button
              type="button"
              className={styles.previewBtn}
              onClick={() => onPreview?.(event)}
              title="Play the footage around this alert"
            >
              <IconPlay /> Preview incident
            </button>
          )}
          {clip === 'pending' && (
            <span className={styles.clipPending}>
              <span className={styles.badgeDot} aria-hidden="true" />
              Preparing clip…
            </span>
          )}
          {clip === 'unavailable' && <span className={styles.clipNote}>No footage saved</span>}
          {clip === 'expired' && <span className={styles.clipNote}>Footage expired</span>}
        </div>
      )}
    </li>
  )
}

export const EventCard = memo(EventCardBase)
