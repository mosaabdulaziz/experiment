/** An alert's Preview: a still from the camera that owns where the customer
 * is, taken when Preview (or Refresh) is pressed, with one mark on it -- a
 * box around the waiting customer, or a long haircut's chair zone outline;
 * no other zones, no other people. Shown in the enlarged-camera window. */
import { stillUrl } from '../../api/client'
import type { Alert, CameraConf } from '../../api/types'
import type { PreviewTarget } from '../../live/alertPreview'
import { useLang } from '../../i18n'
import { cameraLabel } from '../../ui/cameraLabel'
import { fmtClockSeconds } from '../../ui/format'
import { Num, NumIn } from '../../ui/Num'
import { CameraWindow } from './CameraWindow'
import { OvLabel } from './Overlay'

const BOX = 'var(--red-500)'

/** A taken preview: where, when (server ms), a cache-buster for the still,
 * and how many seconds before it the customer was last seen (0 = in view;
 * more = the box is his last position, shadow-tracked). */
export interface Shot extends PreviewTarget {
  kind: Alert['kind']; at: number; bust: number; ago: number
  /** the picture saved when the alert opened (at = then), shown when the live scene had no target (2026-10-18) */
  src?: string
}

export function AlertPreview({ shot, cameras, onRefresh, canRefresh, onClose }: {
  shot: Shot
  cameras: CameraConf[]
  onRefresh: () => void
  canRefresh: boolean
  onClose: () => void
}) {
  const { t } = useLang()
  const stream = cameras.find(c => c.id === shot.cam)?.stream
  const label = t(shot.kind === 'excessive_service' ? 'alert.excessive_service' : 'alert.excessive_wait')
  const caption = (
    <>
      <span className="feedName">{cameraLabel(shot.cam, t)}</span>
      <span className="feedClip">{t(shot.src ? 'alert.storedPicture' : 'alert.takenAt')} <Num>{fmtClockSeconds(shot.at)}</Num></span>
      {shot.ago > 0 && <span className="feedClip"><NumIn text={t('alert.lastSeen', { n: '{n}' })} n={shot.ago} /></span>}
      <button type="button" className="btnGhost btnSmall previewRefresh" onClick={onRefresh} disabled={!canRefresh}
              title={canRefresh ? undefined : t('alert.notTracked')}>
        {t('alert.refresh')}
      </button>
    </>
  )
  return (
    <CameraWindow src={shot.src ?? (stream ? stillUrl(stream, shot.bust) : null)} title={`${label} · ${cameraLabel(shot.cam, t)}`}
                  caption={caption} onClose={onClose}>
      {rect => {
        const [l, tp, w, h] = shot.box
        const x = rect.x + l * rect.w
        const y = rect.y + tp * rect.h
        const pts = shot.outline?.map(([px, py]) => `${rect.x + px * rect.w},${rect.y + py * rect.h}`).join(' ')
        return (
          <g>
            {pts
              ? <polygon points={pts} fill="none" style={{ stroke: BOX }} strokeWidth={3} strokeLinejoin="round" />
              : <rect x={x} y={y} width={w * rect.w} height={h * rect.h} rx={3} fill="none" style={{ stroke: BOX }} strokeWidth={3} />}
            <OvLabel x={x} y={y - 19} text={label} color={BOX} bounds={rect} />
          </g>
        )
      }}
    </CameraWindow>
  )
}
