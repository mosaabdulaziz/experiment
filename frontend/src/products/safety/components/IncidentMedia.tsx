/** An incident's picture, IRIS's way (its incident card's media and stylesheet): the snapshot when it has one, else its
 *  clip paused at the alert moment, with a play badge that opens the clip; or why there is no picture. A proximity
 *  incident the depth model checked also shows the depth model's crop of its pair in the corner, as IRIS's card does.
 *  Fetched only once the card nears the viewport. */
import { useEffect, useState } from 'react'
import { alertClipsApi } from '@/api/alertClips'
import ic from '@/components/incidents/IncidentCard.module.css'
import { IconFilm, IconPlay } from '@/components/ui/icons'
import { useInView } from '@/hooks/useInView'
import { useLang } from '../i18n'
import { mediaOf } from '../lib/media'
import type { SafetyEvent } from '../lib/types'

interface Props {
  event: SafetyEvent
  onPlay: (ev: SafetyEvent) => void
  className?: string
}

export function IncidentMedia({ event, onPlay, className }: Props) {
  const { t } = useLang()
  const [ref, inView] = useInView<HTMLDivElement>()
  const media = mediaOf(event, Date.now())
  const [offset, setOffset] = useState<number | null>(null)
  const [failed, setFailed] = useState(false)
  const [depthFailed, setDepthFailed] = useState(false)
  // the depth model's image of the pair (app: details.depth_image, served by /api/alerts/depth/<event id>)
  const depth = Boolean(event.details?.depth_image) && !depthFailed
  const clipId = media.kind === 'clip' ? media.clipId : null

  // the moment the alert fired, inside its clip: where the paused clip shows
  useEffect(() => {
    if (!inView || !clipId) return
    let cancelled = false
    alertClipsApi.meta(clipId)
      .then(res => {
        const range = res.meta?.range
        const anchor = res.meta?.anchor
        if (!cancelled && range && typeof anchor === 'number') setOffset(Math.max(0, Math.min(anchor - range[0], range[1] - range[0])))
      })
      .catch(() => { if (!cancelled) setFailed(true) })
    return () => { cancelled = true }
  }, [inView, clipId])

  const playable = !failed && ((media.kind === 'snapshot' && media.clip) || media.kind === 'clip')
  return (
    <div ref={ref} className={`${ic.media} ${className ?? ''}`}>
      {(media.kind === 'snapshot' || media.kind === 'clip') && !failed ? (
        <button type="button" className={ic.playSurface} disabled={!playable}
          onClick={e => { e.stopPropagation(); onPlay(event) }} aria-label={t('incident.clip')}>
          {inView && (media.kind === 'snapshot'
            ? <img className={ic.video} src={media.src} alt="" onError={() => setFailed(true)} />
            : <video className={ic.video} src={`${media.clip}#t=${(offset ?? 0).toFixed(2)}`} preload="metadata" muted
                playsInline onError={() => setFailed(true)} />)}
          {depth && inView && (
            <img className={ic.depthInset} src={`/api/alerts/depth/${event.id}`} alt=""
              onError={() => setDepthFailed(true)} />
          )}
          {playable && <span className={ic.playBadge} aria-hidden="true"><IconPlay width={14} height={14} /></span>}
        </button>
      ) : (
        <div className={ic.placeholder}>
          <IconFilm stroke={media.kind === 'pending' ? 'var(--yellow)' : 'var(--text-muted)'} opacity={0.7} />
          <span className={media.kind === 'pending' ? ic.placeholderPending : ic.placeholderNote}>
            {media.kind === 'pending' ? t('incident.clipPending')
              : media.kind === 'expired' ? t('media.expired') : t('media.unavailable')}
          </span>
        </div>
      )}
    </div>
  )
}
