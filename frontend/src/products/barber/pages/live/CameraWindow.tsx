/** One camera enlarged in a window over the page: the same live picture and
 * overlays as its tile (or, for an alert's preview, a still). Modal: focus moves in on open and stays inside,
 * Escape / × / a click on the backdrop closes, the page behind does not
 * scroll. The caller gives focus back to the tile. */
import { useEffect, useRef, type KeyboardEvent, type ReactNode } from 'react'
import { useLang } from '../../i18n'
import { Feed, type FeedBadge, type Rect } from '../../ui/Feed'

export function CameraWindow({ src, live, title, caption, onClose, children, badge }: {
  src?: string | null
  live?: string | null
  title: string
  caption: ReactNode
  onClose: () => void
  children: (rect: Rect) => ReactNode
  /** the live / recording tag on the picture (none for a still) */
  badge?: FeedBadge | null
}) {
  const { t } = useLang()
  const closeRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    closeRef.current?.focus()
    const body = document.body.style
    const before = body.overflow
    body.overflow = 'hidden'
    return () => { body.overflow = before }
  }, [])

  useEffect(() => {
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key !== 'Escape') return
      e.preventDefault()
      onClose()
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose])

  // Tab cycles through the window's own buttons (close, and any in the caption).
  const trapTab = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== 'Tab') return
    e.preventDefault()
    const btns = [...e.currentTarget.querySelectorAll<HTMLButtonElement>('button:not(:disabled)')]
    if (!btns.length) return
    const i = btns.indexOf(document.activeElement as HTMLButtonElement)
    btns[(i + (e.shiftKey ? -1 : 1) + btns.length) % btns.length].focus()
  }

  return (
    <div className="camWindowScrim" onClick={e => { if (e.target === e.currentTarget) onClose() }}>
      <div className="camWindow" role="dialog" aria-modal="true" aria-label={title} onKeyDown={trapTab}>
        <Feed src={src} live={live} caption={caption} className="camWindowFeed" badge={badge}
              action={
                <button ref={closeRef} type="button" className="panelClose camWindowClose"
                        aria-label={t('cam.close')} title={t('cam.close')} onClick={onClose}>
                  <span aria-hidden="true">×</span>
                </button>
              }>
          {children}
        </Feed>
      </div>
    </div>
  )
}
