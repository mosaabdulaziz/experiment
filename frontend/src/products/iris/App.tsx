import { Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { AppShell } from '@/components/layout/AppShell'
import { DashboardPage } from '@/pages/DashboardPage'
import { AnalyticsPage } from '@/pages/AnalyticsPage'
import { IncidentsPage } from '@/pages/IncidentsPage'
import { LandingPage } from '@/pages/LandingPage'
import { FaceGalleryPage } from '@/pages/FaceGalleryPage'
import { SettingsPage } from '@/pages/SettingsPage'
import { SettingsMonitoringPage } from '@/pages/SettingsMonitoringPage'
import { useWebSocket } from '@/hooks/useWebSocket'
import { useAppConfig } from '@/hooks/useAppConfig'

function ShellRoute({ children }: { children: React.ReactNode }) {
  return <AppShell>{children}</AppShell>
}

export function App() {
  // Connect WebSocket once on app boot; keep it alive across route changes
  useWebSocket()
  useAppConfig()
  const location = useLocation()

  // Landing page is "bare" — no app shell wrapper
  if (location.pathname === '/') {
    return (
      <Routes>
        <Route path="/" element={<LandingPage />} />
      </Routes>
    )
  }

  return (
    <Routes>
      <Route
        path="/dashboard"
        element={
          <ShellRoute>
            <DashboardPage />
          </ShellRoute>
        }
      />
      <Route
        path="/incidents"
        element={
          <ShellRoute>
            <IncidentsPage />
          </ShellRoute>
        }
      />
      <Route
        path="/analytics"
        element={
          <ShellRoute>
            <AnalyticsPage />
          </ShellRoute>
        }
      />
      <Route
        path="/settings"
        element={
          <ShellRoute>
            <SettingsPage />
          </ShellRoute>
        }
      >
        <Route index element={<Navigate to="/settings/faces" replace />} />
        <Route path="faces" element={<FaceGalleryPage />} />
        <Route path="monitoring" element={<SettingsMonitoringPage />} />
      </Route>
      <Route path="/faces" element={<Navigate to="/settings/faces" replace />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
