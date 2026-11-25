import { useEffect, useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { facesApi } from '@/api/faces'
import { IconGear, IconLines, IconUser } from '@/components/ui/icons'
import styles from './SettingsPage.module.css'

const POLL_MS = 30_000

/**
 * Settings hub — a two-pane layout that groups configuration tools
 * (Face gallery, AI agent instructions, and future sections like
 * recognition or camera sources) under one navigation item, so the
 * primary header stays focused on monitoring views.
 */
export function SettingsPage() {
  const [pendingCaptures, setPendingCaptures] = useState(0)

  const location = useLocation()

  // Surface unassigned face captures on the rail so operators notice
  // pending work without opening the gallery. Refetches immediately on
  // section navigation and periodically while the settings page is open.
  useEffect(() => {
    let cancelled = false
    const load = async () => {
      try {
        const captures = await facesApi.listCaptures()
        if (!cancelled) {
          setPendingCaptures(captures.filter(c => !c.assigned_to).length)
        }
      } catch (e) {
        // Backend may be offline; keep the last known count.
        console.debug('Pending-capture badge refresh failed:', e)
      }
    }
    load()
    const timer = setInterval(load, POLL_MS)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [location.pathname])

  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `${styles.railLink} ${isActive ? styles.railLinkActive : ''}`

  return (
    <div className={styles.page}>
      <aside className={styles.rail}>
        <div className={styles.railTitle}>
          <IconGear width={13} height={13} />
          <span>Settings</span>
        </div>
        <nav className={styles.railNav} aria-label="Settings sections">
          <NavLink to="/settings/faces" className={linkClass} end>
            <IconUser width={14} height={14} />
            <span className={styles.railLinkLabel}>Face Gallery</span>
            {pendingCaptures > 0 && (
              <span className={styles.badge} title={`${pendingCaptures} pending capture${pendingCaptures !== 1 ? 's' : ''}`}>
                {pendingCaptures}
              </span>
            )}
          </NavLink>
          <NavLink to="/settings/monitoring" className={linkClass} end>
            <IconLines width={14} height={14} />
            <span className={styles.railLinkLabel}>AI Agent</span>
          </NavLink>
        </nav>
        <p className={styles.railHint}>Configuration and management tools.</p>
      </aside>

      <main className={styles.content}>
        <Outlet />
      </main>
    </div>
  )
}
