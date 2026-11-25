/** Is every camera sending video, and is every feature it should run running? The camera-health incidents still
 *  going on (products/safety/camera_health.py), then each camera's features and their state. */
import { useConsole } from '../App'
import { IncidentCard } from '../components/IncidentCard'
import { Pictogram } from '../components/Pictogram'
import { useLang } from '../i18n'
import { isOpen } from '../lib/incidents'
import { CAMERA_FEATURES, type CameraFeature, type SafetyCamera, type SafetyEvent } from '../lib/types'

export function HealthPage() {
  const { t } = useLang()
  const { events, cameras, camerasError, settings, cameraName, acknowledge, play } = useConsole()
  const open = events.filter(e => e.feature === 'health' && isOpen(e))
  const allRunning = (cameras ?? []).every(c => c.features.every(f => c.streams.some(s => s.feature === f && s.running)))

  if (!cameras) {
    return (
      <div className="sf-page">
        <p className={camerasError ? 'sf-error' : 'sf-empty'}>
          {camerasError ? t('common.loadError', { error: camerasError }) : t('common.loading')}
        </p>
      </div>
    )
  }
  return (
    <div className="sf-page">
      {!open.length && allRunning && <p className="sf-ok">{t('health.allGood')}</p>}
      {open.map(ev => (
        <IncidentCard key={ev.id} event={ev} settings={settings} cameraName={cameraName} onAcknowledge={acknowledge}
          onPlay={play} />
      ))}
      <div className="sf-panel sf-table-wrap">
        <table className="sf-table">
          <thead>
            <tr><th>{t('health.camera')}</th><th>{t('health.features')}</th><th>{t('health.status')}</th></tr>
          </thead>
          <tbody>
            {cameras.map(c => <Row key={c.id} camera={c} problems={open.filter(e => e.camera_id === c.id)} />)}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function Row({ camera, problems }: { camera: SafetyCamera; problems: SafetyEvent[] }) {
  const { t, time } = useLang()
  const featureName = (role: unknown) =>
    (CAMERA_FEATURES as unknown[]).includes(role) ? t(`feature.${role as CameraFeature}`) : String(role ?? '')
  const problemText = (e: SafetyEvent) => {
    const d = e.details ?? {}
    const text = d.problem === 'offline' ? t('health.offline')
      : d.problem === 'feature_down' ? t('health.featureDown', { feature: featureName(d.role) })
        : d.problem === 'slow' ? t('health.slow', { feature: featureName(d.role) }) : e.message
    return `${text} ${t('health.since', { time: time(e.incident_started_at ?? e.timestamp) })}`
  }
  // Not running yet and no incident (a problem opens one only after it lasts): said plainly, without a time.
  const down = camera.features.filter(f => !camera.streams.some(s => s.feature === f && s.running))
  const lines = problems.length ? problems.map(problemText)
    : down.map(f => t('health.featureDown', { feature: t(`feature.${f}`) }))

  return (
    <tr>
      <td>{camera.name}</td>
      <td>
        <span className="sf-feature-cell">
          {camera.features.map(f => {
            const stream = camera.streams.find(s => s.feature === f && s.running)
            const fps = stream?.fps ?? null
            const rate = fps === null ? '' : t('health.fps', { n: Math.round(fps * 10) / 10 })
            return (
              <span key={f} className="sf-health-feature" data-stopped={stream ? undefined : true}
                title={`${t(`feature.${f}`)}: ${t(stream ? 'cameras.running' : 'cameras.stopped')}${rate ? `, ${rate}` : ''}`}>
                <Pictogram feature={f} size={22} />
                {rate && <span className="sf-health-fps">{rate}</span>}
              </span>
            )
          })}
        </span>
      </td>
      <td>
        {camera.features.length === 0 ? <span className="sf-muted">—</span>
          : lines.length ? lines.map(l => <div key={l} className="sf-warn">{l}</div>)
            : <span className="sf-ok">{t('health.ok')}</span>}
      </td>
    </tr>
  )
}
