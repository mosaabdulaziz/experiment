import { NavLink, Link } from 'react-router-dom'
import { useCameraStore } from '@/stores/cameraStore'
import { useConfigStore } from '@/stores/configStore'
import { useEventStore } from '@/stores/eventStore'
import { useModeStore } from '@/stores/modeStore'
import { BrandLogo } from '@/components/ui/BrandLogo'
import { ModeToggle } from './ModeToggle'
import { IconChart, IconDashboard, IconGear, IconIncidents } from '@/components/ui/icons'
import styles from './Header.module.css'

const STATUS_LABEL = {
  idle: 'IDLE',
  analyzing: '● ANALYZING',
  paused: '⏸ PAUSED',
  error: '✕ ERROR',
} as const

const STATUS_CLASS = {
  idle: styles.badgeIdle,
  analyzing: styles.badgeAnalyzing,
  paused: styles.badgeIdle,
  error: styles.badgeError,
}

export function Header() {
  const running = useCameraStore(s => s.running)
  const status = useEventStore(s => s.status)
  const eventCount = useEventStore(s => s.events.length)
  const mode = useModeStore(s => s.mode)
  const vlmEnabled = useConfigStore(s => s.vlmEnabled)
  const localCamera = useConfigStore(s => s.perceptionBackend === 'local')

  return (
    <header className={styles.header}>
      <Link to="/" className={styles.brand}>
        <BrandLogo size={34} />
        <div className={styles.brandText}>
          <span className={styles.brandName}>
            <span className={styles.brandAccent}>IRIS</span>
          </span>
          <span className={styles.brandSub}>Smart Surveillance</span>
        </div>
      </Link>

      <div className={styles.center}>
        {/* LIVE/OFF follows the in-process camera; the DeepStream streams show their own state */}
        {localCamera && (
          <span className={`${styles.badge} ${running ? styles.badgeLive : styles.badgeOff}`}>
            <span
              className={`${styles.liveDot} ${running ? '' : styles.dotMuted}`}
              aria-hidden="true"
            />
            {running ? 'LIVE' : 'OFF'}
          </span>
        )}
        {/* without a VLM there is no analysis to show, pause or chat to; ERROR still shows a lost connection */}
        {(vlmEnabled || status === 'error') && (
          <span className={`${styles.badge} ${STATUS_CLASS[status]}`}>{STATUS_LABEL[status]}</span>
        )}
        {vlmEnabled && <ModeToggle />}
        {vlmEnabled && mode === 'chat' && (
          <span className={`${styles.badge} ${styles.badgeIdle}`} title="Proactive analysis is paused">
            ⏸ ANALYSIS PAUSED
          </span>
        )}
      </div>

      <nav className={styles.nav}>
        <NavLink
          to="/dashboard"
          className={({ isActive }) =>
            `${styles.navLink} ${isActive ? styles.navLinkActive : ''}`
          }
        >
          <IconDashboard /> Dashboard
        </NavLink>
        <NavLink
          to="/incidents"
          className={({ isActive }) =>
            `${styles.navLink} ${isActive ? styles.navLinkActive : ''}`
          }
        >
          <IconIncidents /> Incidents
        </NavLink>
        <NavLink
          to="/analytics"
          className={({ isActive }) =>
            `${styles.navLink} ${isActive ? styles.navLinkActive : ''}`
          }
        >
          <IconChart /> Analytics
        </NavLink>
        <NavLink
          to="/settings"
          className={({ isActive }) =>
            `${styles.navLink} ${isActive ? styles.navLinkActive : ''}`
          }
          title="Configuration and management tools"
        >
          <IconGear /> Settings
        </NavLink>
      </nav>

      <div className={styles.right}>
        <span className={styles.counter}>
          {eventCount} event{eventCount !== 1 ? 's' : ''}
        </span>
      </div>
    </header>
  )
}
