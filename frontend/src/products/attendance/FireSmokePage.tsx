import { useState } from 'react'
import type { CameraInfo, IrisEvent } from './api'
import { CAM_LABELS, LiveFrame, camIsLive } from './components'
import { IncidentDetailModal } from './IncidentDetailModal'
import {
  KIND_LABEL, camLabel, fireKind, fmtDateTime, fmtDuration, onsetOf,
} from './useFireAlerts'
import { productHref } from '../../shared/router'

const ENDED_WINDOW_MS = 3600_000

/** Live wall (3a) -- ambient monitoring, the default fire/smoke screen.
 * Stream grid on the fire-tagged cameras + a right rail of active/recently-
 * ended incidents. Data (alerts/cameras/now/acknowledge) is owned by App so
 * the same feed also drives the takeover overlay and the history page. */
export function FireSmokePage({ alerts, cameras, now, connected, loading, error, reload, onAcknowledge, onClear }: {
  alerts: IrisEvent[]
  cameras: CameraInfo[]
  now: number
  connected: boolean
  loading: boolean
  error: string | null
  reload: () => void
  onAcknowledge: (id: number) => void
  onClear: () => Promise<void>
}) {
  const [openId, setOpenId] = useState<number | null>(null)
  const [clearing, setClearing] = useState(false)
  const [clearError, setClearError] = useState<string | null>(null)
  const openEvent = alerts.find(a => a.id === openId) ?? null

  const ongoing = alerts.filter(a => a.incident_status === 'ongoing')
  const endedRecently = alerts
    .filter(a => {
      if (a.incident_status !== 'resolved' || a.incident_duration_seconds == null) return false
      const endedAt = new Date(onsetOf(a)).getTime() + a.incident_duration_seconds * 1000
      return now - endedAt <= ENDED_WINDOW_MS
    })
    .sort((a, b) => {
      const endA = new Date(onsetOf(a)).getTime() + (a.incident_duration_seconds ?? 0) * 1000
      const endB = new Date(onsetOf(b)).getTime() + (b.incident_duration_seconds ?? 0) * 1000
      return endB - endA
    })

  // NOT camera.running -- see camIsLive's doc: every configured camera
  // reports running=true the instant its pipeline is up, even one nobody is
  // publishing to, so it never reflects whether there's real footage.
  const onlineCount = cameras.filter(camIsLive).length

  const handleClear = async () => {
    if (!window.confirm(
      'سيتم حذف كل حوادث الحريق والدخان المعروضة هنا (النشطة والمنتهية) نهائيًا. هل أنت متأكد؟',
    )) return
    setClearing(true)
    setClearError(null)
    try {
      await onClear()
    } catch (e) {
      setClearError(e instanceof Error ? e.message : 'فشل مسح الحوادث')
    } finally {
      setClearing(false)
    }
  }

  return (
    <>
      <div className="pageHead">
        <div>
          <h1>كشف الحريق والدخان</h1>
          <nav className="crumbs" aria-label="مسار التنقل">
            <span>الرئيسية</span>
            <span className="crumbSep">‹</span>
            <span className="crumbHere">كشف الحريق والدخان</span>
          </nav>
        </div>
        <div className="pageHeadTools noPrint">
          <span className="filterTag">{cameras.length} كاميرا · {onlineCount} متصلة</span>
          {ongoing.length > 0 && (
            <span className="liveBadge liveOn fireOngoingChip">
              <span className="liveDot fireOngoingDot" aria-hidden="true" />
              {ongoing.length} حادثة مستمرة
            </span>
          )}
          <span className={connected ? 'liveBadge liveOn' : 'liveBadge'}>
            <span className="liveDot" aria-hidden="true" />
            {connected ? 'مباشر' : 'جارٍ الاتصال'}
          </span>
          <a className="btnGhost btnSmall" href={productHref('/fire/history')}>السجل</a>
          <button type="button" className="btnGhost btnSmall" onClick={reload}>تحديث</button>
        </div>
      </div>

      {error && <div className="errorBar">{error}</div>}

      <div className="liveWallBody">
        <div className="liveWallMain">
          {loading ? (
            <p className="camEmpty">جارٍ تحميل الكاميرات…</p>
          ) : cameras.length === 0 ? (
            <p className="camEmpty">لا توجد كاميرات مخصصة للحريق</p>
          ) : (
            <div className="streamGrid">
              {cameras.map(cam => (
                <StreamTile key={cam.id} camera={cam} alerts={ongoing} />
              ))}
            </div>
          )}
          <div className="healthStrip">
            <span>حالة النظام:</span>
            {cameras.map(cam => (
              <span key={cam.id} className={camIsLive(cam) ? 'healthOk' : 'healthDown'}>
                {camLabel(cam, CAM_LABELS)} {camIsLive(cam) ? '● متصلة' : '● غير متصلة'}
              </span>
            ))}
          </div>
        </div>

        <aside className="liveWallRail">
          <div className="liveWallRailHead">
            <span>الحوادث النشطة</span>
            <span className={ongoing.length > 0 ? 'railCountPill railCountDanger' : 'railCountPill'}>
              {ongoing.length}
            </span>
            <span className="incidentCardSpacer" />
            <button
              type="button"
              className="btnGhost btnSmall"
              disabled={clearing || (ongoing.length === 0 && endedRecently.length === 0)}
              onClick={handleClear}
            >
              {clearing ? '…' : 'مسح'}
            </button>
          </div>
          {clearError && <p className="errorBar">{clearError}</p>}
          <div className="liveWallRailBody">
            {ongoing.length === 0 && endedRecently.length === 0 && (
              <p className="fireUnknown">لا توجد حوادث حالياً — كل الكاميرات خالية</p>
            )}
            {ongoing.map(ev => (
              <ActiveIncidentCard key={ev.id} event={ev} cameras={cameras} now={now}
                                 onAcknowledge={onAcknowledge} onOpen={() => setOpenId(ev.id)} />
            ))}
            {endedRecently.length > 0 && (
              <>
                <div className="railDivider">انتهت خلال الساعة الماضية</div>
                {endedRecently.map(ev => (
                  <EndedIncidentCard key={ev.id} event={ev} cameras={cameras} onOpen={() => setOpenId(ev.id)} />
                ))}
              </>
            )}
          </div>
        </aside>
      </div>

      {openEvent && (
        <IncidentDetailModal
          event={openEvent}
          cameras={cameras}
          now={now}
          onClose={() => setOpenId(null)}
          onAcknowledge={onAcknowledge}
        />
      )}
    </>
  )
}

function StreamTile({ camera, alerts }: { camera: CameraInfo; alerts: IrisEvent[] }) {
  const incident = alerts.find(a => a.camera === camera.id) ?? null
  const kind = incident ? fireKind(incident) : 'unknown'
  const name = camLabel(camera, CAM_LABELS)

  // camera.running (NOT used here, see camIsLive's doc) is true for every
  // sidecar.yaml-configured camera the instant its pipeline is up, even one
  // with nobody publishing to it -- checking it left a camera with no real
  // footage rendering <LiveFrame>, which just polls forever and shows a
  // dimmed "reconnecting" placeholder instead of ever admitting it's down
  // (found live 2026-09-06: cam-8 sat exactly like this after a restart).
  if (!camIsLive(camera)) {
    return (
      <div className="streamTile streamTileOffline">
        <div className="camOffline">{name} — غير متصلة</div>
      </div>
    )
  }

  const tileClass = incident
    ? (kind === 'smoke' ? 'streamTile streamTileSmoke' : 'streamTile streamTileFire')
    : 'streamTile streamTileClear'

  return (
    <div className={tileClass}>
      <LiveFrame camId={camera.id} className="camFrame" />
      {incident && (
        <div className="streamTileTop">
          <span>● {KIND_LABEL[kind]} — مستمر</span>
          <span dir="ltr">{fmtDuration((Date.now() - new Date(onsetOf(incident)).getTime()) / 1000)}</span>
        </div>
      )}
      <div className="streamTileBottom">
        <strong dir="ltr">{name}</strong>
        <span className="streamTileMeta" dir="ltr">
          {camera.zone ?? '—'}{incident ? ` · منذ ${fmtDateTime(onsetOf(incident))}` : ' · خالية'}
        </span>
      </div>
    </div>
  )
}

function ActiveIncidentCard({ event, cameras, now, onAcknowledge, onOpen }: {
  event: IrisEvent; cameras: CameraInfo[]; now: number
  onAcknowledge: (id: number) => void; onOpen: () => void
}) {
  const kind = fireKind(event)
  const cam = cameras.find(c => c.id === event.camera) ?? null
  const name = cam ? camLabel(cam, CAM_LABELS) : (event.camera ?? '—')
  return (
    <div className={kind === 'smoke' ? 'incidentCard incidentCardSmoke' : 'incidentCard incidentCardFire'}>
      <div className="incidentCardRow1">
        <span>{KIND_LABEL[kind]}</span>
        {/* This card only ever renders for incident_status === 'ongoing' (see
            the `ongoing` filter above), so the "ongoing" pill stays regardless
            of acknowledgement -- acknowledging silences the takeover alarm,
            it does not mean the fire/smoke is gone. Acknowledged gets its own
            ADDITIONAL pill rather than replacing this one, so an operator
            glancing at the rail can't mistake an acknowledged-but-still-
            ongoing incident for a resolved one. */}
        <span className="statusPill statusPillDanger">مستمر</span>
        {event.acknowledged_at && (
          <span className="statusPill">تم الإقرار</span>
        )}
        <span className="incidentCardSpacer" />
        <span dir="ltr" className="fireOngoingChip">
          {fmtDuration((now - new Date(onsetOf(event)).getTime()) / 1000)}
        </span>
      </div>
      <div className="incidentCardRow2" dir="ltr">
        <strong>{name}</strong>
        {cam?.zone && <span>{cam.zone}</span>}
        <span>بدأ {fmtDateTime(onsetOf(event))}</span>
      </div>
      <div className="incidentCardActions">
        {!event.acknowledged_at && (
          <button type="button" className="btnPrimary btnSmall" onClick={() => onAcknowledge(event.id)}>
            إقرار
          </button>
        )}
        <button type="button" className="btnGhost btnSmall" onClick={onOpen}>فتح الحادثة</button>
      </div>
    </div>
  )
}

function EndedIncidentCard({ event, cameras, onOpen }: {
  event: IrisEvent; cameras: CameraInfo[]; onOpen: () => void
}) {
  const kind = fireKind(event)
  const cam = cameras.find(c => c.id === event.camera) ?? null
  const name = cam ? camLabel(cam, CAM_LABELS) : (event.camera ?? '—')
  const started = new Date(onsetOf(event)).getTime()
  const ended = event.incident_duration_seconds != null ? started + event.incident_duration_seconds * 1000 : null
  return (
    <div className="incidentCard incidentCardEnded rowClickable" onClick={onOpen}>
      <div className="incidentCardRow1">
        <span>{KIND_LABEL[kind]} · {name}</span>
        <span className="incidentCardSpacer" />
        <span dir="ltr">
          {event.incident_duration_seconds != null ? `استمر ${fmtDuration(event.incident_duration_seconds)}` : ''}
        </span>
      </div>
      <div className="incidentCardRow2" dir="ltr">
        {fmtDateTime(onsetOf(event))} → {ended != null ? fmtDateTime(new Date(ended).toISOString()) : '—'} · انتهى
      </div>
    </div>
  )
}
