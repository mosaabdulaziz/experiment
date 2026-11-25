import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { configApi } from '@/api/config'
import { BrandLogo } from '@/components/ui/BrandLogo'
import { IconCamera, IconChat, IconEye } from '@/components/ui/icons'
import styles from './LandingPage.module.css'

type HealthState = 'checking' | 'ok' | 'bad'

const FEATURES = [
  {
    icon: <IconEye width={20} height={20} strokeWidth={1.5} />,
    title: 'Real-time AI monitoring',
    body: 'A vision-language model analyzes short batches of frames and posts comments and alerts as events unfold.',
  },
  {
    icon: <IconCamera width={20} height={20} strokeWidth={1.5} />,
    title: 'Multi-source capture',
    body: 'Run on a USB webcam, RTSP stream, or an uploaded video file. Switch sources from the dashboard without restarting.',
  },
  {
    icon: <IconChat width={20} height={20} strokeWidth={1.5} />,
    title: 'Conversational chat',
    body: 'Ask the camera anything. Each turn sees the live frame plus the recent event log, with full session memory.',
  },
]

export function LandingPage() {
  const [health, setHealth] = useState<HealthState>('checking')

  useEffect(() => {
    let cancelled = false
    configApi
      .getAppConfig()
      .then(() => !cancelled && setHealth('ok'))
      .catch(() => !cancelled && setHealth('bad'))
    return () => {
      cancelled = true
    }
  }, [])

  const statusClass =
    health === 'ok' ? styles.statusOk : health === 'bad' ? styles.statusBad : ''
  const statusText =
    health === 'ok' ? 'Backend online' : health === 'bad' ? 'Backend unreachable' : 'Checking…'

  return (
    <div className={styles.page}>
      <div className={styles.orbs} aria-hidden="true">
        <div className={`${styles.orb} ${styles.orb1}`} />
        <div className={`${styles.orb} ${styles.orb2}`} />
        <div className={`${styles.orb} ${styles.orb3}`} />
      </div>

      <section className={styles.hero}>
        <div className={styles.heroLogoWrap}>
          <div className={styles.heroLogo}>
            <BrandLogo size={104} />
          </div>
        </div>
        <h1 className={styles.heroTitle}>
          <span className={styles.accent}>IRIS</span>
        </h1>
        <p className={styles.heroTagline}>Smart Surveillance</p>
        <p className={styles.heroBody}>
          A FastAPI-backed surveillance dashboard that streams frames to a vision-language model
          in real time. Watches live feeds, files, and RTSP sources — emits comments, raises alerts,
          and answers questions on demand.
        </p>
        <div className={styles.heroCta}>
          <Link to="/dashboard" className={styles.btnPrimary}>
            Open Dashboard →
          </Link>
          <Link to="/analytics" className={styles.btnSecondary}>
            View Analytics
          </Link>
        </div>
        <div className={`${styles.status} ${statusClass}`} role="status">
          <span className={styles.statusDot} />
          {statusText}
        </div>
      </section>

      <section className={styles.features}>
        {FEATURES.map(f => (
          <div key={f.title} className={styles.feature}>
            <div className={styles.featureIcon}>{f.icon}</div>
            <div className={styles.featureTitle}>{f.title}</div>
            <div className={styles.featureBody}>{f.body}</div>
          </div>
        ))}
      </section>

      <footer className={styles.footer}>IRIS · Smart Surveillance</footer>
    </div>
  )
}
