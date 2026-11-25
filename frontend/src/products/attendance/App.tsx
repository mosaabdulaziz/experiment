import { useCallback, useEffect, useRef, useState } from 'react'
import {
  api, attendancePolicyApi, monitoringApi,
  type AttendancePolicy, type AttendanceRecord, type MonitoringSnapshot, type RecordingInfo,
} from './api'
import { useAttendanceWebSocket } from './useAttendanceWebSocket'
import {
  AttendanceBubbles, AttendanceTable, CameraPanel, DepartmentStatus, ExportButtons, HourlyAttendanceChart, KpiCards,
} from './components'
import { BrandLogo } from '../../shared/ui/BrandLogo'
import { MonitoringPage } from './MonitoringPage'
import { AddEmployeePage, EmployeesPage } from './EmployeesPage'
import { HistoryPage } from './HistoryPage'
import { ViolationsPanel } from './ViolationsPanel'
import { AccessAlertToasts, AlertsPage, openAlertCount } from './AccessAlerts'
import { CamerasPage } from './CamerasPage'
import { SettingsPage } from './SettingsPage'
import { FireSmokePage } from './FireSmokePage'
import { FireHistoryPage } from './FireHistoryPage'
import { FireTakeover } from './FireTakeover'
import { useFireAlerts, useFireCameras, useNowTick } from './useFireAlerts'
import { productHref, productPath } from '../../shared/router'

/** Tiny hash router: '#/' dashboard, '#/monitoring' productivity page. */
function useHashRoute(): string {
  const [route, setRoute] = useState(() => productPath())
  useEffect(() => {
    const onChange = () => setRoute(productPath())
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  return route
}

// 2026-09-12 (issue 1, Phase 1): was '2023-06-08', the CHIRLA recording's
// fake simulated date -- the backend no longer dates anything that way (see
// consumer.py/server.py), so a hardcoded default here just meant this page
// silently asked for a day that could never have data again, both for its
// initial REST fetch (load(day) below) and for every live websocket push
// (onAttendance only accepts a message whose day matches this state, so even
// a real-time update from "today" got dropped while this was stuck on the
// old fake date). Note: useState(todayIso) only runs when the page loads, so a
// tab left open across midnight used to stay on yesterday (found live
// 2026-09-13) -- the effect next to `day` below now moves it forward.
function todayIso(): string {
  const d = new Date()
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`
}

export type Theme = 'light' | 'dark'
const THEME_KEY = 'attendance-theme'

/** prompt.md image-6, 2026-09-04: "نوع العرض" (display type). Client-side
 * preference only -- no backend concept of a theme -- applied instantly via
 * a data-theme attribute app.css's dark palette reads (see :root[data-theme
 * ="dark"]), not gated behind any "save" button since every other theme
 * toggle in real software applies live. */
function useTheme(): [Theme, (t: Theme) => void] {
  const [theme, setThemeState] = useState<Theme>(() => {
    try {
      return (localStorage.getItem(THEME_KEY) as Theme | null) ?? 'light'
    } catch {
      return 'light'
    }
  })
  useEffect(() => {
    document.documentElement.dataset.theme = theme
  }, [theme])
  const setTheme = (t: Theme) => {
    setThemeState(t)
    try { localStorage.setItem(THEME_KEY, t) } catch { /* private mode etc. -- just won't persist */ }
  }
  return [theme, setTheme]
}

export function App() {
  const route = useHashRoute()
  const [theme, setTheme] = useTheme()
  const [live, setLive] = useState(false)
  // one websocket for the whole app: attendance snapshots feed the
  // dashboard page; monitoring snapshots feed BOTH the monitoring page's
  // own live view (fetched there directly) and, up here, whatever the
  // attendance page needs from it -- violations, zones (for the zone
  // editor) and its policy (for Settings). Both moved onto the attendance
  // page 2026-09-03 at the user's request; only the video overlay stayed on
  // the monitoring page.
  const [records, setRecords] = useState<AttendanceRecord[]>([])
  const [snap, setSnap] = useState<MonitoringSnapshot | null>(null)
  const [attendancePolicy, setAttendancePolicy] = useState<AttendancePolicy | null>(null)
  const [day, setDay] = useState(todayIso)
  // Follow the real date across midnight (2026-09-13): every 30 s, if the page
  // is still on the day it last followed as "today", move it to the new today.
  // A date picked by hand in the date picker is left alone.
  const followedDay = useRef(day)
  useEffect(() => {
    const id = setInterval(() => {
      const t = todayIso()
      if (t !== followedDay.current) {
        const prev = followedDay.current
        followedDay.current = t
        setDay(d => (d === prev ? t : d))
      }
    }, 30_000)
    return () => clearInterval(id)
  }, [])
  const [search, setSearch] = useState('')

  useEffect(() => {
    monitoringApi.snapshot().then(setSnap).catch(() => {})
    attendancePolicyApi.get().then(setAttendancePolicy).catch(() => {})
  }, [])

  const onMonitoring = useCallback((msg: MonitoringSnapshot) => setSnap(msg), [])
  useAttendanceWebSocket(msg => {
    setLive(true)
    if (msg.day === day) setRecords(msg.records)
  }, onMonitoring)

  // Same "one socket feeds every consumer" reasoning as the attendance feed
  // above: the live wall, the takeover overlay and the history page all read
  // the same fire alerts/cameras feed, so it's fetched once here rather than
  // each page opening its own /ws connection.
  const fire = useFireAlerts()
  const fireCameras = useFireCameras()
  const fireAnyOngoing = fire.alerts.some(a => a.incident_status === 'ongoing')
  const fireNow = useNowTick(fireAnyOngoing ? 1000 : null)

  return (
    <>
    <AccessAlertToasts violations={snap?.violations ?? []} />
    <FireTakeover alerts={fire.alerts} cameras={fireCameras} now={fireNow} onAcknowledge={fire.acknowledge} />
    <div className="shell">
      <nav className="rail noPrint" aria-label="التنقل">
        <span className="railLogo" title="IRIS"><BrandLogo size={32} /></span>
        <a className={route === '/' ? 'railBtn railActive' : 'railBtn'} href={productHref('/')} title="الحضور" aria-label="الحضور">
          <RailIcon d="M3 10.5 12 3l9 7.5V21a1 1 0 0 1-1 1h-5v-7h-6v7H4a1 1 0 0 1-1-1z" />
        </a>
        <a className={route === '/cameras' ? 'railBtn railActive' : 'railBtn'} href={productHref('/cameras')} title="الكاميرات" aria-label="الكاميرات">
          <RailIcon d="M17 10.5V7a2 2 0 0 0-2-2H4a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2v-3.5l5 4v-11z" />
        </a>
        <a className={route.startsWith('/admin/employees') ? 'railBtn railActive' : 'railBtn'} href={productHref('/admin/employees')} title="الموظفين" aria-label="الموظفين">
          <RailIcon d="M12 12a4.5 4.5 0 1 0 0-9 4.5 4.5 0 0 0 0 9zm0 2c-4.4 0-8 2.2-8 5v1a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-1c0-2.8-3.6-5-8-5z" />
        </a>
        <a className={route === '/alerts' ? 'railBtn railActive' : 'railBtn'} href={productHref('/alerts')} title="تنبيهات الدخول" aria-label="تنبيهات الدخول">
          <RailIcon d="M12 2a6 6 0 0 0-6 6v3.6L4.3 15a1 1 0 0 0 .9 1.5h13.6a1 1 0 0 0 .9-1.5L18 11.6V8a6 6 0 0 0-6-6zm-2.5 16.5a2.5 2.5 0 0 0 5 0z" />
          {openAlertCount(snap?.violations ?? []) > 0 && <span className="railCount">{openAlertCount(snap?.violations ?? [])}</span>}
        </a>
        <a
          className={route === '/monitoring' ? 'railBtn railActive railInProgress' : 'railBtn railInProgress'}
          href={productHref('/monitoring')} title="مراقبة الإنتاجية (قيد التطوير)" aria-label="مراقبة الإنتاجية (قيد التطوير)"
        >
          <RailIcon d="M9 3v4M15 3v4M4 8h16M5 6h14a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1z" stroke />
          <span className="railBadge" aria-hidden="true" />
        </a>
        <a className={route.startsWith('/fire') ? 'railBtn railActive' : 'railBtn'} href={productHref('/fire')} title="كشف الحريق والدخان" aria-label="كشف الحريق والدخان">
          <RailIcon d="M12 2c-.3 2.7-2 4-3.5 5.5C7 9 6 10.7 6 13a6 6 0 0 0 12 0c0-2-1-3.3-2-4.5.3 1.3-.2 2.3-1 2.5.4-2-1-3-1.3-5C13.4 4.2 12.7 3 12 2zm0 8.5c.6.9 1 1.6 1 2.5a1 1 0 0 1-2 0c0-.4.2-.8.4-1.2.2-.4.4-.8.6-1.3z" />
        </a>
        <a className={route === '/settings' ? 'railBtn railActive railBottom' : 'railBtn railBottom'} href={productHref('/settings')} title="الإعدادات" aria-label="الإعدادات">
          <RailIcon d="M12 8a4 4 0 1 0 4 4 4 4 0 0 0-4-4zm9 4a7.7 7.7 0 0 0-.1-1.2l2-1.6-2-3.4-2.4 1a7.7 7.7 0 0 0-2-1.2L16 3h-4l-.4 2.6a7.7 7.7 0 0 0-2 1.2l-2.5-1-2 3.4 2 1.6A7.7 7.7 0 0 0 7 12a7.7 7.7 0 0 0 .1 1.2l-2 1.6 2 3.4 2.4-1a7.7 7.7 0 0 0 2 1.2L12 21h4l.4-2.6a7.7 7.7 0 0 0 2-1.2l2.5 1 2-3.4-2-1.6A7.7 7.7 0 0 0 21 12z" />
        </a>
      </nav>

      <main className="page">
        <header className="topbar noPrint">
          <div className="userBox">
            <span className="userAvatar" aria-hidden="true">م</span>
            <div>
              <strong>مشرف النظام</strong>
              <small>إدارة الحضور</small>
            </div>
            <span className={live ? 'liveBadge liveOn' : 'liveBadge'}>
              <span className="liveDot" aria-hidden="true" />
              {live ? 'مباشر' : 'جارٍ الاتصال'}
            </span>
          </div>
          <div className="searchBox">
            <input
              type="search"
              placeholder="بحث بالاسم أو الرقم الوظيفي أو القسم…"
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
          </div>
        </header>
        {route === '/monitoring' ? (
          <MonitoringPage live={live} />
        ) : route === '/admin/employees/new' ? (
          <AddEmployeePage attendancePolicy={attendancePolicy} setAttendancePolicy={setAttendancePolicy} />
        ) : route === '/admin/employees' ? (
          <EmployeesPage attendancePolicy={attendancePolicy} setAttendancePolicy={setAttendancePolicy} />
        ) : route === '/alerts' ? (
          <AlertsPage snap={snap} day={day} policy={attendancePolicy} />
        ) : route === '/cameras' ? (
          <CamerasPage snap={snap} setSnap={setSnap} attendancePolicy={attendancePolicy} setAttendancePolicy={setAttendancePolicy} />
        ) : route === '/fire' ? (
          <FireSmokePage
            alerts={fire.alerts} cameras={fireCameras} now={fireNow}
            connected={fire.connected} loading={fire.loading} error={fire.error}
            reload={fire.reload} onAcknowledge={fire.acknowledge} onClear={fire.clear}
          />
        ) : route === '/fire/history' ? (
          <FireHistoryPage
            alerts={fire.alerts} cameras={fireCameras} now={fireNow} onAcknowledge={fire.acknowledge}
          />
        ) : route === '/settings' ? (
          <SettingsPage
            snap={snap} setSnap={setSnap} attendancePolicy={attendancePolicy} setAttendancePolicy={setAttendancePolicy}
            theme={theme} setTheme={setTheme}
          />
        ) : route.startsWith('/employee/') ? (
          <HistoryPage employeeId={route.slice('/employee/'.length)} />
        ) : (
          <DashboardPage
            day={day} setDay={setDay} records={records} setRecords={setRecords} search={search} snap={snap}
            policy={attendancePolicy}
          />
        )}
      </main>
    </div>
    </>
  )
}

function DashboardPage({ day, setDay, records, setRecords, search, snap, policy }: {
  day: string
  setDay: (d: string) => void
  records: AttendanceRecord[]
  setRecords: (r: AttendanceRecord[]) => void
  search: string
  snap: MonitoringSnapshot | null
  policy: AttendancePolicy | null  // work hours + late grace, for the diagrams
}) {
  const [recordings, setRecordings] = useState<RecordingInfo[]>([])
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async (targetDay: string) => {
    setError(null)
    try {
      setRecords(await api.daily(targetDay))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'تعذر تحميل البيانات')
    }
  }, [])

  useEffect(() => {
    load(day)
  }, [day, load])

  // recordings drive the camera tiles' clocks + which tiles show; polled at
  // the same 2s cadence the camera list uses (streams come and go).
  useEffect(() => {
    let cancelled = false
    const tick = () => {
      api.listRecordings().then(r => {
        if (!cancelled) setRecordings(r)
      }).catch(() => {})
    }
    tick()
    const id = setInterval(tick, 2000)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [])

  return (
    <>
        <div className="pageHead">
          <div>
            <h1>نظام الحضور والانصراف الذكي</h1>
            <nav className="crumbs" aria-label="مسار التنقل">
              <span>الرئيسية</span>
              <span className="crumbSep">‹</span>
              <span className="crumbHere">نظام الحضور والانصراف الذكي</span>
            </nav>
          </div>
          <div className="pageHeadTools noPrint">
            <ExportButtons records={records} day={day} policy={policy} />
            <input
              className="dayInput"
              type="date"
              value={day}
              onChange={e => setDay(e.target.value)}
              title="تاريخ اليوم"
            />
          </div>
        </div>

        {error && <div className="errorBar">{error}</div>}

        <KpiCards records={records} />

        {/* Layout per prompt.md image-7 (2026-09-13). RTL: each row's first
            child sits on the right. */}
        <div className="dashRow dashRowWideFirst">
          <HourlyAttendanceChart records={records} policy={policy} isToday={day === todayIso()} />
          <DepartmentStatus records={records} />
        </div>
        {/* Camera row between the two diagram rows (prompt.md point 3,
            2026-09-14); the dashStack wrapper keeps its row styling. */}
        <div className="dashStack dashCamRow">
          <CameraPanel recordings={recordings} zones={snap?.zones ?? {}} />
        </div>
        <div className="dashRow dashRowNarrowFirst">
          <AttendanceBubbles records={records} policy={policy} />
          <AttendanceTable records={records} search={search} policy={policy} />
        </div>

        <div className="dashStack">
          <ViolationsPanel violations={snap?.violations ?? []} day={day} />
        </div>
    </>
  )
}

function RailIcon({ d, stroke = false }: { d: string; stroke?: boolean }) {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
      <path
        d={d}
        fill={stroke ? 'none' : 'currentColor'}
        stroke={stroke ? 'currentColor' : 'none'}
        strokeWidth={stroke ? 1.8 : 0}
        strokeLinecap="round"
      />
    </svg>
  )
}
