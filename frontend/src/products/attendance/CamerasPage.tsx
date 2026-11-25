import { useEffect, useState, type Dispatch, type SetStateAction } from 'react'
import {
  api, attendancePolicyApi,
  type AttendancePolicy, type CameraInfo, type MonitoringSnapshot, type RecordingInfo,
} from './api'
import { CAM_LABELS, LiveFrame, camIsLive, useWallClock } from './components'
import { ZoneEditor } from './MonitoringPage'

/** prompt.md point 4: "see all the cameras, select one, adjust its zones,
 * mark them restricted or not and who's allowed in, and mark the place
 * office or break room" -- all in one place, per-camera, rather than the
 * previous split (zones in a modal, camera type in a Settings list).
 *
 * 2026-09-06: the grid used to be a hardcoded ATTENDANCE_CAM_IDS list, so it
 * always rendered the same fixed set of tiles no matter which cameras were
 * actually producing footage -- most sessions only run a handful of the
 * configured CHIRLA cameras, so the rest sat there forever as a dead "غير
 * متصلة الآن" placeholder (can't usefully draw zones on a camera with no
 * visible frame anyway). The grid now lists whatever /api/cameras reports
 * as an attendance-role camera that's ACTUALLY streaming right now (see
 * camIsLive's doc for why that's `offset`, not `running`) -- that endpoint
 * parses sidecar.yaml's roles live (see stream_manager.list_streams /
 * _load_sidecar_yaml), so this page tracks the sidecar's actual config
 * instead of a copy that can silently drift from it. */

function fmtClock(d: Date): string {
  return d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

function CameraGridTile({ camId, live, recording, selected, onSelect }: {
  camId: string
  live: boolean
  recording: RecordingInfo | undefined
  selected: boolean
  onSelect: () => void
}) {
  const wallClock = useWallClock(recording)
  return (
    <div
      className={selected ? 'camTile camTileBig camTileSelected' : 'camTile camTileBig'}
      role="button"
      tabIndex={0}
      title="اضغط للاختيار"
      onClick={onSelect}
      onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') onSelect() }}
    >
      {live ? <LiveFrame camId={camId} /> : <div className="camOffline">غير متصلة الآن</div>}
      {live && <span className="camLive">بث مباشر</span>}
      <span className="camName">{CAM_LABELS[camId] ?? camId}</span>
      {wallClock && <span className="camClock" title={recording?.clock_source === 'camera' ? 'ساعة الكاميرا (RTCP)' : 'ساعة الخادم'}>{fmtClock(wallClock)}</span>}
    </div>
  )
}

export function CamerasPage({ snap, setSnap, attendancePolicy, setAttendancePolicy }: {
  snap: MonitoringSnapshot | null
  setSnap: Dispatch<SetStateAction<MonitoringSnapshot | null>>
  attendancePolicy: AttendancePolicy | null
  setAttendancePolicy: (p: AttendancePolicy) => void
}) {
  const [cameras, setCameras] = useState<CameraInfo[]>([])
  const [recordings, setRecordings] = useState<RecordingInfo[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    const tick = () => {
      api.listCameras().then(c => { if (!cancelled) setCameras(c) }).catch(() => {})
      api.listRecordings().then(r => { if (!cancelled) setRecordings(r) }).catch(() => {})
    }
    tick()
    const id = setInterval(tick, 2000)
    return () => { cancelled = true; clearInterval(id) }
  }, [])

  const recordingByCam = Object.fromEntries(recordings.map(r => [r.cam_id, r]))
  const attendanceCams = cameras
    .filter(c => c.roles.includes('attendance') && camIsLive(c))
    .sort((a, b) => a.id.localeCompare(b.id, undefined, { numeric: true }))

  return (
    <>
      <div className="pageHead">
        <div>
          <h1>الكاميرات</h1>
          <nav className="crumbs" aria-label="مسار التنقل">
            <span>الرئيسية</span><span className="crumbSep">‹</span><span className="crumbHere">الكاميرات</span>
          </nav>
        </div>
      </div>

      <div className="camerasGrid">
        {attendanceCams.map(cam => (
          <CameraGridTile
            key={cam.id}
            camId={cam.id}
            live
            recording={recordingByCam[cam.id]}
            selected={cam.id === selectedId}
            onSelect={() => setSelectedId(cam.id)}
          />
        ))}
        {attendanceCams.length === 0 && (
          <p className="camEmpty">لا توجد كاميرات تبث حاليًا</p>
        )}
      </div>

      {selectedId ? (
        <ZoneEditor
          key={selectedId}
          camId={selectedId}
          initial={snap?.zones[selectedId] ?? []}
          inline
          cameraType={attendancePolicy?.camera_types[selectedId] ?? 'office'}
          onSaved={zones => setSnap(s => (s ? { ...s, zones: { ...s.zones, [selectedId]: zones } } : s))}
          onCameraTypeSaved={async type => {
            if (!attendancePolicy) return
            const saved = await attendancePolicyApi.save({
              ...attendancePolicy,
              camera_types: { ...attendancePolicy.camera_types, [selectedId]: type },
            })
            setAttendancePolicy(saved)
          }}
        />
      ) : (
        <p className="camEmpty cardHint">اختر كاميرا أعلاه لتعديل مناطقها ونوع المكان والأدوار المسموح لها بالدخول</p>
      )}
    </>
  )
}
