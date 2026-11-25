import { useEffect, useRef, useState } from 'react'
import type { StreamInfo } from '@/types/camera'
import { useStreamsStore } from '@/stores/streamsStore'
import { useEventStore } from '@/stores/eventStore'
import { useElementSize } from '@/hooks/useElementSize'
import { formatOffset } from '@/lib/format'
import { IconAlert, IconExpand, IconZone } from '@/components/ui/icons'
import { ZoneOverlay } from './ZoneOverlay'
import { useFramePoll } from './framePoll'
import styles from './StreamTile.module.css'

interface Props {
  stream: StreamInfo
}

export function StreamTile({ stream }: Props) {
  const setExpanded = useStreamsStore(s => s.setExpanded)
  const setZoneEdit = useStreamsStore(s => s.setZoneEdit)
  const paused = useStreamsStore(s => s.pipelineDesired === 'paused')
  const showZones = useStreamsStore(s => s.showZones)
  // Only live flashes (a new alert, or an incident raised in place), never historical setHistory loads.
  const flashSeq = useEventStore(s => s.flashSeqByCamera[stream.id] ?? 0)
  const [flash, setFlash] = useState(false)
  // Feed's intrinsic size (from <img> onLoad) + the tile box size, for the zone overlay.
  const [natural, setNatural] = useState<{ width: number; height: number } | null>(null)
  const { ref: videoRef, width: videoW, height: videoH } = useElementSize<HTMLDivElement>()
  // Seed from current store value so remounts don't re-flash the last known alert.
  const lastFlashedRef = useRef(flashSeq)

  useEffect(() => {
    if (flashSeq === lastFlashedRef.current) return
    lastFlashedRef.current = flashSeq
    setFlash(true)
    const t = setTimeout(() => setFlash(false), 2600)
    return () => clearTimeout(t)
  }, [flashSeq])

  // The next frame 100 ms after the last one loads, never two at once (framePoll.ts): a
  // setInterval changing src blindly overran the browser's 6 connections per host across
  // 14 tiles, aborting requests in flight and dropping playback below 1 fps.
  const frames = useFramePoll(stream.id, stream.running, 100)

  const handleLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    setNatural({ width: e.currentTarget.naturalWidth, height: e.currentTarget.naturalHeight })
    frames.onLoad()
  }

  const live = stream.running
  const showTime = stream.is_file && stream.offset != null

  return (
    <div
      role="button"
      tabIndex={0}
      className={`${styles.tile} ${flash ? styles.flash : ''}`}
      onClick={() => setExpanded(stream.id)}
      onKeyDown={e => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          setExpanded(stream.id)
        }
      }}
      aria-label={`Expand ${stream.id} — ${stream.source}`}
    >
      <div className={styles.video} ref={videoRef}>
        {/* No key on the <img>: remounting the <img> every poll would blank the
            tile while the next JPEG decodes (flicker). Swapping src keeps the
            last frame visible until the new one loads. */}
        <img
          className={styles.stream}
          src={frames.src}
          alt={`${stream.id} feed`}
          onLoad={handleLoad}
          onError={frames.onError}
        />
        {!live && <div className={styles.offScrim} />}
        {showZones && (
          <ZoneOverlay streamId={stream.id} width={videoW} height={videoH} natural={natural} />
        )}
      </div>

      <div className={styles.topBar}>
        <span className={styles.id}>{stream.id}</span>
        <span className={`${styles.status} ${live ? styles.statusLive : styles.statusOff}`}>
          <span className={styles.dot} />
          {live ? 'Live' : 'Off'}
        </span>
      </div>

      <div className={styles.bottomBar}>
        <span className={styles.source}>{stream.source}</span>
        {showTime && <span className={styles.time}>{formatOffset(stream.offset as number)}</span>}
      </div>

      <div className={styles.alertBanner} aria-hidden="true">
        <IconAlert width={14} height={14} strokeWidth={2.5} />
        Alert
      </div>

      {!paused && (
        <span className={styles.expandHint} aria-hidden="true">
          <IconExpand />
        </span>
      )}

      <button
        type="button"
        className={`${styles.zoneBtn} ${paused ? styles.zoneBtnCentered : styles.zoneBtnCorner}`}
        onClick={e => {
          e.stopPropagation()
          setZoneEdit(stream.id)
        }}
        aria-label={`Edit zones for ${stream.id}`}
      >
        <IconZone width={14} height={14} />
        <span>Zones</span>
      </button>
    </div>
  )
}
