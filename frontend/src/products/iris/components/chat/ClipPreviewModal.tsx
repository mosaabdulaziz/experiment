import { useEffect, useState } from 'react'
import { IconClose } from '@/components/ui/icons'
import type { ClipInfo } from '@/types/chat'
import styles from './ClipPreviewModal.module.css'

const pad = (n: number) => String(n).padStart(2, '0')
function fmtClock(sec: number): string {
  const t = Math.max(Math.round(sec), 0)
  return `${pad(Math.floor(t / 3600))}:${pad(Math.floor((t % 3600) / 60))}:${pad(t % 60)}`
}

export function ClipPreviewModal({
  clip,
  onClose,
  basePath = '/api/chat/clip',
  title = 'Match preview',
}: {
  clip: ClipInfo | null
  onClose: () => void
  /** Endpoint root for playback. Chat FOUND clips and fired-alert clips are served
   *  by two different routes with the same response shape. */
  basePath?: string
  title?: string
}) {
  const [active, setActive] = useState(0)
  const [failed, setFailed] = useState(false)

  // Reset camera tab + error state whenever a different clip is opened.
  useEffect(() => {
    setActive(0)
    setFailed(false)
  }, [clip?.id])

  // Close on Escape while open.
  useEffect(() => {
    if (!clip) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [clip, onClose])

  if (!clip) return null
  const cam = clip.cameras[active] ?? clip.cameras[0]
  const range = clip.range ? `${fmtClock(clip.range[0])} – ${fmtClock(clip.range[1])}` : null

  return (
    <div className={styles.backdrop} onClick={onClose}>
      <div
        className={styles.panel}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        onClick={e => e.stopPropagation()}
      >
        <header className={styles.header}>
          <div className={styles.headLeft}>
            <span className={styles.title}>{title}</span>
            {range && <span className={styles.range}>{range}</span>}
          </div>
          <button type="button" className={styles.close} onClick={onClose} aria-label="Close preview">
            <IconClose /> Close
          </button>
        </header>

        {clip.cameras.length > 1 && (
          <div className={styles.tabs}>
            {clip.cameras.map((c, i) => (
              <button
                key={c}
                type="button"
                className={`${styles.tab} ${i === active ? styles.tabActive : ''}`}
                onClick={() => setActive(i)}
              >
                {c}
              </button>
            ))}
          </div>
        )}

        <div className={styles.body}>
          {failed ? (
            <div className={styles.fallback}>Preview unavailable (clip expired).</div>
          ) : (
            <video
              key={cam}
              className={styles.video}
              src={`${basePath}/${clip.id}/${cam}`}
              controls
              autoPlay
              muted
              loop
              playsInline
              onError={() => setFailed(true)}
            />
          )}
        </div>
      </div>
    </div>
  )
}
