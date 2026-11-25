import { useEffect } from 'react'
import styles from './UploadOverlay.module.css'

export type UploadPhase = 'uploading' | 'processing' | 'done' | 'error'

interface Props {
  filename: string
  phase: UploadPhase
  percent: number
  errorMessage?: string | null
  onDismiss?: () => void
}

export function UploadOverlay({ filename, phase, percent, errorMessage, onDismiss }: Props) {
  // Lock body scroll while overlay is open
  useEffect(() => {
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.body.style.overflow = prev
    }
  }, [])

  // ESC dismiss (only when in a dismissible state)
  useEffect(() => {
    if (phase !== 'error' || !onDismiss) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onDismiss()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [phase, onDismiss])

  const title =
    phase === 'uploading'
      ? 'Uploading video'
      : phase === 'processing'
      ? 'Preparing stream'
      : phase === 'done'
      ? 'Stream ready'
      : 'Upload failed'

  const subtitle =
    phase === 'uploading'
      ? `${percent}% complete`
      : phase === 'processing'
      ? 'Spinning up the analysis pipeline…'
      : phase === 'done'
      ? 'Streaming will start momentarily'
      : 'Please try again'

  const cardClass = `${styles.card} ${
    phase === 'error' ? styles.error : phase === 'done' ? styles.done : ''
  }`

  return (
    <div className={styles.backdrop} role="dialog" aria-modal="true" aria-label={title}>
      <div className={cardClass}>
        <div className={styles.header}>
          {phase === 'error' ? (
            <div className={styles.iconError} aria-hidden="true">!</div>
          ) : phase === 'done' ? (
            <div className={styles.iconDone} aria-hidden="true">✓</div>
          ) : (
            <div className={styles.spinner} aria-hidden="true" />
          )}
          <div className={styles.titleBlock}>
            <div className={styles.title}>{title}</div>
            <div className={styles.filename} title={filename}>{filename}</div>
          </div>
        </div>

        {phase === 'error' ? (
          <div className={styles.errorBody}>{errorMessage ?? 'Something went wrong.'}</div>
        ) : (
          <>
            <div className={styles.barWrap}>
              {phase === 'uploading' ? (
                <div className={styles.bar} style={{ width: `${percent}%` }} />
              ) : phase === 'done' ? (
                <div className={styles.bar} style={{ width: '100%' }} />
              ) : (
                <div className={styles.barIndeterminate} />
              )}
            </div>
            <div className={styles.meta}>
              <span>{subtitle}</span>
              <span className={styles.percent}>
                {phase === 'uploading' ? `${percent}%` : phase === 'done' ? '100%' : '···'}
              </span>
            </div>
          </>
        )}

        {phase === 'error' && onDismiss && (
          <div className={styles.actions}>
            <button type="button" className="iris-btn iris-btn-ghost" onClick={onDismiss}>
              Dismiss
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

