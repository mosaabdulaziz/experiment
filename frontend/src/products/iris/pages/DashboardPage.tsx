import type { CSSProperties } from 'react'
import { CameraFeed } from '@/components/camera/CameraFeed'
import { RightPanel } from '@/components/events/RightPanel'
import { StreamGrid } from '@/components/streams/StreamGrid'
import { ExpandedStream } from '@/components/streams/ExpandedStream'
import { ZoneEditorOverlay } from '@/components/streams/ZoneEditorOverlay'
import { ResizeHandle } from '@/components/ui/ResizeHandle'
import { usePanelWidth } from '@/hooks/usePanelWidth'
import { useStreams } from '@/hooks/useStreams'
import { useStreamsStore } from '@/stores/streamsStore'
import styles from './DashboardPage.module.css'

// The alerts panel: as wide as the operator drags it, between these (and never more than 62% of the row).
const ALERTS_DEFAULT = 460
const ALERTS_LIMITS = { min: 300, max: 860 }

export function DashboardPage() {
  const mode = useStreams()
  const zoneEditId = useStreamsStore(s => s.zoneEditId)
  const [alertsWidth, setAlertsWidth] = usePanelWidth('iris:alertsWidth', ALERTS_DEFAULT, ALERTS_LIMITS)

  return (
    <>
      <main className={styles.main} style={{ '--alerts-w': `${alertsWidth}px` } as CSSProperties}>
        {mode === 'loading' ? (
          <div className={styles.loading}>Connecting to streams…</div>
        ) : mode === 'deepstream' ? (
          <StreamGrid />
        ) : (
          <CameraFeed />
        )}
        <div className={styles.side}>
          <ResizeHandle width={alertsWidth} onChange={setAlertsWidth} limits={ALERTS_LIMITS}
            defaultWidth={ALERTS_DEFAULT} label="Resize the alerts panel" />
          <RightPanel />
        </div>
      </main>
      {mode === 'deepstream' && <ExpandedStream />}
      {mode === 'deepstream' && zoneEditId && <ZoneEditorOverlay key={zoneEditId} />}
    </>
  )
}
