import { useRef, useState } from 'react'
import { useStreamsStore } from '@/stores/streamsStore'
import { useElementSize } from '@/hooks/useElementSize'
import { IconCamera } from '@/components/ui/icons'
import { streamsApi } from '@/api/streams'
import { StreamTile } from './StreamTile'
import { ProximityPreviewSettings } from './ProximityPreviewSettings'
import { scrollGridRows } from './layout'
import styles from './StreamGrid.module.css'

/** Shown while a saved zone edit is still waiting for the sidecar to hot-reload
 *  the ROI config — covers acks that land after the zone editor was closed. */
function ZonesApplyingBadge() {
  const zonesStatus = useStreamsStore(s => s.zonesStatus)
  const applying =
    zonesStatus !== null && zonesStatus.enabled && zonesStatus.version > zonesStatus.appliedVersion
  if (!applying) return null
  return (
    <span className={styles.zonesApplying}>
      <span className={styles.zonesApplyingDot} />
      Zones applying…
    </span>
  )
}

function PipelineButton() {
  const desired = useStreamsStore(s => s.pipelineDesired)
  const setPipelineDesired = useStreamsStore(s => s.setPipelineDesired)
  const [busy, setBusy] = useState(false)

  const isPlaying = desired === 'playing'

  const toggle = async () => {
    setBusy(true)
    try {
      if (isPlaying) {
        await streamsApi.stopPipeline()
        setPipelineDesired('paused')
      } else {
        await streamsApi.startPipeline()
        setPipelineDesired('playing')
      }
    } catch {
      /* leave state as-is; the polling loop will correct it */
    } finally {
      setBusy(false)
    }
  }

  return (
    <button
      className={`${styles.pipelineBtn} ${isPlaying ? styles.pipelineBtnStop : styles.pipelineBtnStart}`}
      onClick={toggle}
      disabled={busy}
      title={isPlaying ? 'Stop stream processing' : 'Start stream processing'}
    >
      {busy ? '…' : isPlaying ? '⏹ Stop' : '▶ Start'}
    </button>
  )
}

function ShowZonesButton() {
  const showZones = useStreamsStore(s => s.showZones)
  const setShowZones = useStreamsStore(s => s.setShowZones)
  return (
    <button
      type="button"
      role="switch"
      aria-checked={showZones}
      className={styles.zonesSwitch}
      onClick={() => setShowZones(!showZones)}
      title={showZones ? 'Hide zone outlines on the wall' : 'Show zone outlines on the wall'}
    >
      <span className={styles.zonesSwitchLabel}>Show zones</span>
      <span className={`${styles.zonesSwitchTrack} ${showZones ? styles.zonesSwitchTrackOn : ''}`}>
        <span className={styles.zonesSwitchKnob} />
      </span>
    </button>
  )
}

/** Tiles per row for the scrollable wall, from container width. */
function gridCols(width: number): number {
  return Math.max(2, Math.min(6, Math.floor(width / 300)))
}

const ROW_GAP = 12
const VISIBLE_ROWS = 2 // rows on screen at once (~8 tiles at 4 columns)
const WINDOW_BUFFER = 1 // extra rows rendered just above/below the viewport

export function StreamGrid() {
  const streams = useStreamsStore(s => s.streams)
  const desired = useStreamsStore(s => s.pipelineDesired)
  const { ref, width, height } = useElementSize<HTMLElement>()
  const [scrollTop, setScrollTop] = useState(0)
  const scrollRaf = useRef<number | null>(null)

  const handleScroll = () => {
    const el = ref.current
    if (!el) return
    if (scrollRaf.current != null) return
    scrollRaf.current = requestAnimationFrame(() => {
      scrollRaf.current = null
      setScrollTop(el.scrollTop)
    })
  }

  if (streams.length === 0) {
    return (
      <section className={styles.empty}>
        <div className={styles.emptyIcon} aria-hidden="true">
          <IconCamera width={40} height={40} stroke="var(--accent)" />
        </div>
        <p className={styles.emptyTitle}>
          {desired === 'paused' ? 'Pipeline paused' : 'No streams yet'}
        </p>
        <p className={styles.emptyBody}>
          {desired === 'paused'
            ? 'Click Start to begin stream processing.'
            : 'Waiting for the sidecar to publish camera feeds.\nCheck that the DeepStream sidecar is running.'}
        </p>
        <PipelineButton />
      </section>
    )
  }

  // Scrollable wall: fixed tile grid (~8 visible), rows windowed so only the
  // on-screen tiles are mounted — off-screen tiles stop polling /frame
  // entirely (their timers die with the unmount).
  const cols = gridCols(width)
  const rows = scrollGridRows(streams, cols)
  const rowH =
    height > 0 ? Math.max(140, (height - ROW_GAP * (VISIBLE_ROWS - 1)) / VISIBLE_ROWS) : 200
  const rowStride = rowH + ROW_GAP
  const startRow = Math.max(0, Math.floor(scrollTop / rowStride) - WINDOW_BUFFER)
  const endRow = Math.min(
    rows.length - 1,
    Math.ceil((scrollTop + height) / rowStride) + WINDOW_BUFFER
  )

  return (
    <section className={styles.wallWrapper}>
      <div className={styles.wallToolbar}>
        <ZonesApplyingBadge />
        <ShowZonesButton />
        <ProximityPreviewSettings />
        <PipelineButton />
      </div>
      <section ref={ref} className={styles.wall} onScroll={handleScroll}>
        {rows.map((row, i) => (
          <div key={i} className={styles.row} style={{ height: rowH, flex: '0 0 auto' }}>
            {i >= startRow && i <= endRow
              ? row.map(s => <StreamTile key={s.id} stream={s} />)
              : null}
          </div>
        ))}
      </section>
    </section>
  )
}
