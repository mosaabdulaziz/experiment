import { useEffect, useState } from 'react'
import { useCameraStore } from '@/stores/cameraStore'
import { useEventStore } from '@/stores/eventStore'
import { useStreamTime } from '@/hooks/useStreamTime'
import { formatOffset, formatTime } from '@/lib/format'
import { cameraApi } from '@/api/camera'
import { CameraControls } from './CameraControls'
import styles from './CameraFeed.module.css'

export function CameraFeed() {
  const running = useCameraStore(s => s.running)
  const isFile = useCameraStore(s => s.isFile)
  const source = useCameraStore(s => s.source)
  const lastAlertId = useEventStore(s => s.lastAlertId)
  const lastEvent = useEventStore(s => s.events[s.events.length - 1])

  const offset = useStreamTime(running && isFile)
  const [streamKey, setStreamKey] = useState(0)
  const [alertMessage, setAlertMessage] = useState<string | null>(null)

  // Cache-bust stream when camera transitions to running
  useEffect(() => {
    if (running) setStreamKey(k => k + 1)
  }, [running])

  // Show alert overlay when a new alert event arrives
  useEffect(() => {
    if (!lastAlertId) return
    const ev = lastEvent
    if (!ev || ev.id !== lastAlertId) return
    setAlertMessage(ev.message)
    const timer = setTimeout(() => setAlertMessage(null), 8000)
    return () => clearTimeout(timer)
  }, [lastAlertId, lastEvent])

  const handleStreamError = () => {
    if (running) setTimeout(() => setStreamKey(k => k + 1), 2000)
  }

  const handleRestart = async () => {
    try {
      const data = await cameraApi.useWebcam()
      useCameraStore.getState().setStatus({
        running: data.status === 'started',
        source: data.source ?? source,
        isFile: !!data.is_file,
        paused: false,
      })
    } catch (err) {
      console.error('Restart camera failed:', err)
    }
  }

  const lastAnalysisText = lastEvent
    ? `Last: ${formatTime(lastEvent.timestamp)} — ${lastEvent.event_type.toUpperCase()}`
    : 'Waiting for first analysis…'

  return (
    <section className={styles.panel}>
      <div className={styles.title}>
        <div className={styles.titleLeft}>
          <span className={styles.label}>Camera Feed</span>
          <span className={styles.sourceChip}>{source}</span>
        </div>
        <CameraControls />
      </div>

      <div className={styles.wrapper}>
        <img
          key={streamKey}
          className={styles.stream}
          src={running ? `/stream?t=${streamKey}` : '/stream'}
          alt="Camera stream"
          onError={handleStreamError}
        />

        {running && isFile && offset != null && (
          <div className={styles.timeBadge}>{formatOffset(offset)}</div>
        )}

        {!running && (
          <div className={styles.offOverlay}>
            <div className={styles.offIcon} aria-hidden="true">
              <svg
                width="32"
                height="32"
                viewBox="0 0 24 24"
                fill="none"
                stroke="#f9996c"
                strokeWidth="1.5"
                opacity="0.5"
              >
                <rect x="2" y="6" width="15" height="12" rx="2" />
                <path d="M17 9l5-3v12l-5-3" />
                <line x1="12" y1="2" x2="12" y2="22" stroke="#ef4444" strokeWidth="2" />
              </svg>
            </div>
            <span className={styles.offLabel}>Camera stopped</span>
            <button
              type="button"
              onClick={handleRestart}
              className="iris-btn iris-btn-success iris-btn-sm"
            >
              Start Camera
            </button>
          </div>
        )}

        {alertMessage && (
          <div className={styles.alertOverlay}>
            <div className={styles.alertIcon}>⚠</div>
            <span>{alertMessage}</span>
          </div>
        )}
      </div>

      <div className={styles.footer}>{lastAnalysisText}</div>
    </section>
  )
}
