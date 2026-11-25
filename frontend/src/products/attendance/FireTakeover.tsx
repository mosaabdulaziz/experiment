import { useEffect, useMemo, useRef, useState } from 'react'
import { alertClipUrl, type CameraInfo, type IrisEvent } from './api'
import { CAM_LABELS, LiveFrame } from './components'
import { camLabel, effectiveClips, fmtDuration, onsetOf } from './useFireAlerts'

const ARM_DELAY_MS = 1000 // ENTER cannot dismiss an alarm the operator has not had a moment to see

/** Takeover alarm (3b): a centered pop-up window for a new, unacknowledged
 * FIRE incident (smoke stays in the live wall's rail, per the wireframe's
 * default). The backdrop is transparent (2026-09-06: it used to be a solid
 * full-screen block -- the operator needs to keep seeing the dashboard
 * behind it) -- only the window itself is opaque. Mounted once at the App
 * level so it interrupts whichever page the operator is on, not just the
 * fire tab.
 *
 * Only ever shows ONE incident at a time -- a second fire arriving mid-takeover
 * is queued (badge in the alarm bar) rather than stacked, and is revealed once
 * the current one is acknowledged. */
export function FireTakeover({ alerts, cameras, now, onAcknowledge }: {
  alerts: IrisEvent[]
  cameras: CameraInfo[]
  now: number
  onAcknowledge: (id: number) => void
}) {
  const pending = useMemo(() => alerts
    .filter(a => a.incident_status === 'ongoing' && !a.acknowledged_at
               && a.message.startsWith('Fire '))
    .sort((a, b) => new Date(onsetOf(a)).getTime() - new Date(onsetOf(b)).getTime()),
    [alerts])

  const target = pending[0] ?? null
  const queuedCount = Math.max(pending.length - 1, 0)

  const [showingClip, setShowingClip] = useState(false)
  const armedAtRef = useRef<number>(0)

  useEffect(() => {
    armedAtRef.current = Date.now() + ARM_DELAY_MS
    setShowingClip(false)
  }, [target?.id])

  useEffect(() => {
    if (!target) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Enter' && Date.now() >= armedAtRef.current) onAcknowledge(target.id)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [target, onAcknowledge])

  if (!target) return null

  const cam = cameras.find(c => c.id === target.camera) ?? null
  const streamName = cam ? camLabel(cam, CAM_LABELS) : (target.camera ?? '—')
  const burning = fmtDuration((now - new Date(onsetOf(target)).getTime()) / 1000)
  const clips = effectiveClips(target)
  const others = cameras.filter(c => c.id !== target.camera)

  return (
    <div className="takeover" role="alertdialog" aria-modal="true" aria-live="assertive">
      <div className="takeoverWindow">
        <div className="takeoverBar">
          <span className="takeoverHeadline">🔴 تم رصد حريق</span>
          <span className="takeoverMeta" dir="ltr">{streamName} · {cam?.zone ?? ''}</span>
          {queuedCount > 0 && <span className="takeoverQueued">+{queuedCount} في الانتظار</span>}
          <span className="takeoverSpacer" />
          <span className="takeoverMeta" dir="ltr">
            بدأ {new Date(onsetOf(target)).toLocaleTimeString('en-GB')} · غير مُقرّ منذ {burning}
          </span>
        </div>

        <div className="takeoverBody">
          <div className="takeoverPlayer">
            {showingClip && clips[0] ? (
              <video className="camFrame" src={alertClipUrl(clips[0].clip_id, target.camera ?? '')}
                    controls autoPlay muted playsInline />
            ) : (
              <LiveFrame camId={target.camera ?? ''} className="camFrame" />
            )}
            <span className="takeoverLiveTag">{showingClip ? 'مقطع مسجل' : 'مباشر'}</span>
            <span className="takeoverBurningTag">مستمر منذ {burning}</span>
          </div>

          <div className="takeoverActions">
            <button type="button" className="btnPrimary takeoverAckBtn"
                   onClick={() => onAcknowledge(target.id)}>
              إقرار الإنذار
            </button>
            <button type="button" className="btnGhost btnOnDark"
                   disabled={clips.length === 0}
                   onClick={() => setShowingClip(s => !s)}>
              {showingClip ? 'العودة للبث المباشر' : 'عرض المقطع المسجل'}
            </button>
          </div>

          <p className="takeoverHint" dir="rtl">اضغط ENTER للإقرار</p>

          {others.length > 0 && (
            <div className="takeoverOthers">
              <span className="takeoverOthersLabel">الكاميرات الأخرى</span>
              <div className="takeoverOthersGrid">
                {others.map(c => (
                  <div key={c.id} className="takeoverOtherTile">
                    <LiveFrame camId={c.id} className="camFrame" />
                    <span className="camName">{camLabel(c, CAM_LABELS)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
