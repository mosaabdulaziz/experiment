import { useEffect, useState } from 'react'
import type { IrisEvent } from '@/types/event'
import { ALERT_KIND_LABEL, alertKind, clipState } from '@/lib/clips'
import { useConfigStore } from '@/stores/configStore'
import { useInView } from '@/hooks/useInView'
import { alertClipsApi } from '@/api/alertClips'
import { formatTime } from '@/lib/format'
import { IconFilm, IconPlay } from '@/components/ui/icons'
import { incidentStatusLine, proximityFacts, SEVERITY_LABEL } from '@/lib/proximity'
import { TriggerTick } from './TriggerTick'
import styles from './IncidentCard.module.css'

interface Props {
  event: IrisEvent
  onPreview: (event: IrisEvent) => void
}

interface Placement {
  /** Seconds from the start of the clip to the fired moment. */
  offset: number
  /** Same, as a percentage of clip length. */
  pct: number
}

export function IncidentCard({ event, onPreview }: Props) {
  const ttl = useConfigStore(s => s.clipTtlSeconds)
  // An incident whose first clip failed but whose escalation clip was stored still plays the clip it has.
  const firstClip = event.clip_id ?? event.clips?.[0]?.clip_id ?? null
  const state = event.clip_id || !firstClip
    ? clipState(event, ttl)
    : clipState({ ...event, clip_status: 'ready', clip_id: firstClip }, ttl)
  const kind = alertKind(event)
  const facts = kind === 'proximity' ? proximityFacts(event) : null
  const statusLine = facts ? incidentStatusLine(event, facts) : null
  // Typed as HTMLElement, not HTMLDivElement — the ref lands on an <article>, whose
  // JSX ref type is Ref<HTMLElement>, and RefObject is not covariant.
  const [ref, inView] = useInView<HTMLElement>()
  const [placement, setPlacement] = useState<Placement | null>(null)
  const [failed, setFailed] = useState(false)
  // a snapshot that will not load (expired, a network blip) falls back to the clip, which may still play
  const [snapshotFailed, setSnapshotFailed] = useState(false)
  const [depthFailed, setDepthFailed] = useState(false)
  const [expanded, setExpanded] = useState(false)

  const clipId = firstClip
  const camera = event.camera
  const snapshotSrc = facts && event.snapshot_id && camera && !snapshotFailed
    ? `/api/alerts/snapshot/${event.snapshot_id}/${camera}/${event.id}` : null
  const playable = state === 'ready' && !!clipId && !!camera && !failed

  // Metadata is fetched with the card's first approach to the viewport, alongside
  // the video mount, so an off-screen card costs nothing. The poster seek and the
  // tick both come from this response rather than being assumed from config.
  useEffect(() => {
    if (!inView || !playable || !clipId) return
    let cancelled = false
    alertClipsApi
      .meta(clipId)
      .then(res => {
        if (cancelled) return
        const range = res.meta?.range
        const anchor = res.meta?.anchor
        if (!range || typeof anchor !== 'number') return
        const span = range[1] - range[0]
        if (!(span > 0)) return
        const offset = Math.min(Math.max(anchor - range[0], 0), span)
        setPlacement({ offset, pct: (offset / span) * 100 })
      })
      .catch(() => {
        // Pruned between render and scroll — fall through to the unavailable face.
        if (!cancelled) setFailed(true)
      })
    return () => {
      cancelled = true
    }
  }, [inView, playable, clipId])

  const triggerTime = formatTime(event.timestamp)

  return (
    <article className={styles.card} ref={ref}>
      {state !== 'none' && (
        <div className={styles.media}>
          {playable ? (
            <button
              type="button"
              className={styles.playSurface}
              onClick={() => onPreview({ ...event, clip_id: clipId })}
              aria-label={`Preview incident at ${triggerTime} on ${camera}`}
            >
              {snapshotSrc ? (
                // like the video, fetched only once the card nears the viewport (a snapshot is a full-size JPEG)
                inView && (
                  <img className={styles.video} src={snapshotSrc} alt={`Closest moment of the incident on ${camera}`}
                    onError={() => setSnapshotFailed(true)} />
                )
              ) : inView && (
                <video
                  className={styles.video}
                  src={`/api/alerts/clip/${clipId}/${camera}#t=${(placement?.offset ?? 0).toFixed(2)}`}
                  preload="metadata"
                  muted
                  playsInline
                  onError={() => setFailed(true)}
                />
              )}
              {facts?.depthImage && inView && !depthFailed && (
                <img className={styles.depthInset} src={`/api/alerts/depth/${event.id}`}
                  alt={`Depth map of the pair on ${camera}`} onError={() => setDepthFailed(true)} />
              )}
              <span className={styles.playBadge} aria-hidden="true">
                <IconPlay width={14} height={14} />
              </span>
            </button>
          ) : (
            <div className={styles.placeholder}>
              <IconFilm
                stroke={state === 'pending' ? 'var(--yellow)' : 'var(--text-muted)'}
                opacity={0.7}
              />
              <span
                className={state === 'pending' ? styles.placeholderPending : styles.placeholderNote}
              >
                {state === 'pending' && 'Preparing clip…'}
                {state === 'unavailable' && 'No footage saved'}
                {state === 'expired' && 'Footage expired'}
                {state === 'ready' && 'Footage unavailable'}
              </span>
            </div>
          )}
        </div>
      )}

      {playable && placement && <TriggerTick positionPct={placement.pct} label={triggerTime} />}

      <div className={styles.body}>
        <div className={styles.stamp}>{triggerTime}</div>
        <div className={styles.chips}>
          {event.camera && <span className={styles.chip}>{event.camera}</span>}
          <span className={`${styles.chip} ${styles[kind]}`}>{ALERT_KIND_LABEL[kind]}</span>
        </div>
        <div className={styles.title} dir="auto">
          {event.message}
        </div>
        {facts && (
          <div className={styles.facts}>
            <div className={styles.chips}>
              {facts.heldBack ? (
                // a pair the depth check held back (Ticket 6): not an alert, so no severity of its own
                <>
                  <span className={`${styles.chip} ${styles.flag}`}>Held back by depth</span>
                  {facts.wouldBe && (
                    <span className={`${styles.chip} ${styles[`sev_${facts.wouldBe}`]}`}>
                      would have been {SEVERITY_LABEL[facts.wouldBe] ?? facts.wouldBe}
                    </span>
                  )}
                </>
              ) : event.severity && (
                <span className={`${styles.chip} ${styles[`sev_${event.severity}`]}`}>
                  {SEVERITY_LABEL[event.severity]}
                </span>
              )}
              {facts.unconfirmed && <span className={`${styles.chip} ${styles.flag}`}>Feet not visible — unconfirmed</span>}
              {facts.depthConfirmed && <span className={styles.chip}>Depth-confirmed</span>}
            </div>
            {statusLine && <div className={styles.status}>{statusLine}</div>}
            <dl className={styles.factList}>
              {facts.zone && (<><dt>Zone</dt><dd>{facts.zone}</dd></>)}
              {facts.vehicleClass && (<><dt>Vehicle</dt><dd>{facts.vehicleClass}</dd></>)}
              {facts.gap !== null && (
                <><dt>Minimum gap</dt><dd>{facts.gap.toFixed(2)} person heights (≈ {facts.gapCm} cm)</dd></>
              )}
              {facts.depthGap !== null && (
                <><dt>Depth gap</dt><dd>{facts.depthGap.toFixed(2)}{facts.heldBack ? ' — apart' : ''}
                  {facts.depthMaxGap !== null ? ` (same depth at ≤ ${facts.depthMaxGap.toFixed(2)})` : ''}</dd></>
              )}
              {facts.closeFor !== null && (<><dt>Close for</dt><dd>{facts.closeFor.toFixed(1)} s</dd></>)}
              {facts.persons.length > 0 && (<><dt>Person</dt><dd>#{facts.persons.join(', #')}</dd></>)}
              {facts.vehicles.length > 0 && (<><dt>Vehicle track</dt><dd>#{facts.vehicles.join(', #')}</dd></>)}
            </dl>
            {(event.clips?.length ?? 0) > 1 && camera && (
              <div className={styles.clipChoice}>
                {event.clips!.map((c, i) => (
                  <button key={c.clip_id} type="button" className={styles.clipButton}
                    onClick={() => onPreview({ ...event, clip_id: c.clip_id })}>
                    {i === 0 ? 'First alert' : 'Escalation'}
                  </button>
                ))}
              </div>
            )}
            {snapshotSrc && !facts.snapshotExact && (
              <div className={styles.note}>Snapshot: closest available frame</div>
            )}
          </div>
        )}
        {event.description_status === 'pending' && (
          <div className={styles.descPending}>Generating description…</div>
        )}
        {event.description_status === 'ready' && event.description && (
          // Clamped to 3 lines, click to expand. A button rather than a <p> so the
          // toggle is reachable by keyboard.
          <button
            type="button"
            className={`${styles.description} ${expanded ? styles.descriptionOpen : ''}`}
            onClick={() => setExpanded(v => !v)}
            aria-expanded={expanded}
            dir="auto"
          >
            {event.description}
          </button>
        )}
      </div>
    </article>
  )
}
