/** A camera picture with an SVG drawn over the rect the picture actually
 * occupies (object-fit: contain), so overlays in normalized frame
 * coordinates stay on the floor whatever the camera's aspect ratio.
 * Live video is go2rtc MJPEG frames over a WebSocket (`live`), shown only
 * while frames flow (a dropped or stale socket shows the notice, not its
 * last frame); a still (`src`) is an <img> that retries every 4 s when it
 * fails. */
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { useLang } from '../i18n'
import { contentRect } from './geometry'
import { useMjpegSocket } from './mjpegSocket'

export interface Rect { x: number; y: number; w: number; h: number }

const RETRY_MS = 4000

/** Size of the element and of the picture inside it. */
export function useFeedRect(natural: { w: number; h: number }) {
  const ref = useRef<HTMLDivElement | null>(null)
  const [box, setBox] = useState({ w: 0, h: 0 })
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const update = () => setBox({ w: el.clientWidth, h: el.clientHeight })
    const ro = new ResizeObserver(update)
    ro.observe(el)
    update()
    return () => ro.disconnect()
  }, [])
  const rect = contentRect(box.w, box.h, natural.w || 16, natural.h || 9)
  return { ref, box, rect }
}

/** On-picture tag (2026-10-18, user): "Live" while frames come in, or a recording's position. */
export type FeedBadge = { kind: 'live'; text: string } | { kind: 'clip'; text: string; label: string }

/** `caption` sits above the video (in the page's direction); the video and
 * its overlay are spatial and always left-to-right. `children` receives the
 * picture rect. `action` (e.g. a button covering the tile) goes last in the
 * figure. `badge` shows in the picture's top corner only while it plays. */
export function Feed({ src, live, caption, className = '', children, action, badge }: {
  src?: string | null
  /** A go2rtc stream name: live video over the WebSocket instead of `src`. */
  live?: string | null
  caption?: ReactNode
  className?: string
  children?: (rect: Rect) => ReactNode
  action?: ReactNode
  badge?: FeedBadge | null
}) {
  const { t } = useLang()
  const [natural, setNatural] = useState({ w: 0, h: 0 })
  const [failed, setFailed] = useState(false)
  const [attempt, setAttempt] = useState(0)
  const { ref, box, rect } = useFeedRect(natural)
  useEffect(() => {
    if (!failed) return
    const id = setTimeout(() => { setFailed(false); setAttempt(a => a + 1) }, RETRY_MS)
    return () => clearTimeout(id)
  }, [failed])
  const sock = useMjpegSocket(live ?? null)
  const url = live ? sock.frame : src ? `${src}${src.includes('?') ? '&' : '?'}r=${attempt}` : null
  const noVideo = live ? sock.state === 'down' : failed || !url
  const playing = !!url && (live ? sock.state === 'live' : !failed)
  return (
    <figure className={`feedTile ${className}`}>
      {caption && <figcaption className="feedCaption">{caption}</figcaption>}
      <div ref={ref} className="feed">
        {playing && (
          <img
            src={url}
            alt=""
            onLoad={e => {
              // live frames load ~20 times a second: re-render only when the size changes
              const w = e.currentTarget.naturalWidth, h = e.currentTarget.naturalHeight
              setNatural(n => (n.w === w && n.h === h ? n : { w, h }))
            }}
            onError={live ? undefined : () => setFailed(true)}
          />
        )}
        {noVideo && <div className="feedNotice" role="status">{t('cam.noVideo')}</div>}
        {children && <svg width={box.w} height={box.h} aria-hidden="true">{box.w > 0 && children(rect)}</svg>}
        {badge && playing && (badge.kind === 'live'
          ? <span className="feedBadge feedLive"><i aria-hidden="true" />{badge.text}</span>
          : <span className="feedBadge feedReplay" role="img" aria-label={badge.label}>
              <span aria-hidden="true">▶</span>{badge.text}
            </span>)}
      </div>
      {action}
    </figure>
  )
}
