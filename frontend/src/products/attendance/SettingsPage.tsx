import type { Dispatch, SetStateAction } from 'react'
import type { Theme } from './App'
import { AttendanceSettingsPanel, SettingsPanel } from './MonitoringPage'
import type { AttendancePolicy, MonitoringSnapshot } from './api'

/** prompt.md point 5 (2026-09-03): Settings gets its own page, moved back
 * off the Attendance page it was put on earlier. prompt.md image-6
 * (2026-09-04) added the "نوع العرض" row-card below and restyled the rest
 * of this page to match its title+description | control card language --
 * see SettingsPanel/AttendanceSettingsPanel in MonitoringPage.tsx for the
 * two backend-store panels that now use the same row-card CSS. */
function ThemeCard({ theme, setTheme }: { theme: Theme; setTheme: (t: Theme) => void }) {
  const isDark = theme === 'dark'
  return (
    <section className="settingsRowCard">
      <div className="settingsRowInfo">
        <h3>نوع العرض</h3>
        <p>تحديث نوع العرض. سيؤثر هذا الخيار على كامل لوحة التحكم.</p>
      </div>
      <div className="settingsRowControl">
        <div className="themeSwitch">
          <span>فاتح</span>
          <button
            type="button"
            className="themeSwitchTrack"
            data-on={isDark}
            role="switch"
            aria-checked={isDark}
            aria-label="نوع العرض"
            onClick={() => setTheme(isDark ? 'light' : 'dark')}
          >
            <span className="themeSwitchKnob" />
          </button>
          <span>داكن</span>
        </div>
      </div>
    </section>
  )
}

export function SettingsPage({ snap, setSnap, attendancePolicy, setAttendancePolicy, theme, setTheme }: {
  snap: MonitoringSnapshot | null
  setSnap: Dispatch<SetStateAction<MonitoringSnapshot | null>>
  attendancePolicy: AttendancePolicy | null
  setAttendancePolicy: (p: AttendancePolicy) => void
  theme: Theme
  setTheme: (t: Theme) => void
}) {
  return (
    <>
      <div className="pageHead">
        <div>
          <h1>الإعدادات</h1>
          <nav className="crumbs" aria-label="مسار التنقل">
            <span>الرئيسية</span><span className="crumbSep">‹</span><span className="crumbHere">الإعدادات</span>
          </nav>
        </div>
      </div>

      <div className="settingsStack">
        <ThemeCard theme={theme} setTheme={setTheme} />
        {attendancePolicy && <AttendanceSettingsPanel policy={attendancePolicy} onSaved={setAttendancePolicy} />}
        {snap && <SettingsPanel policy={snap.policy} onSaved={p => setSnap(s => (s ? { ...s, policy: p } : s))} />}
      </div>
    </>
  )
}
