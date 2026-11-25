import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// One dev server for every product, so a developer runs one command and the
// switcher moves between dashboards. Ports come from the checkout's .env (the
// justfile exports them); the fallbacks are the historical defaults, so a bare
// `npm run dev` outside `just` behaves as each app did before they were merged.
const attendancePort = Number(process.env.ATTENDANCE_BACKEND_PORT) || 8001
const barberPort = Number(process.env.BARBER_BACKEND_PORT) || 8003
const fallPort = Number(process.env.FALL_BACKEND_PORT) || 8004
const concealmentPort = Number(process.env.CONCEALMENT_BACKEND_PORT) || 8005
const irisPort = Number(process.env.IRIS_HOST_PORT) || 18000
const go2rtcPort = Number(process.env.GO2RTC_API_PORT) || 1984

const http = (port: number) => ({ target: `http://localhost:${port}`, changeOrigin: true })
const ws = (port: number) => ({ target: `ws://localhost:${port}`, ws: true })

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      // Belongs to ONE product: the surveillance UI is copied from its own repo
      // and imports itself as `@/...`. Mapping it here keeps that source
      // untouched, so the next copy across stays a copy instead of a merge.
      // Other products use relative imports and need nothing here.
      '@': fileURLToPath(new URL('./src/products/iris', import.meta.url)),
    },
  },
  server: {
    port: Number(process.env.ATTENDANCE_FRONTEND_PORT) || 5180,
    host: '0.0.0.0',   // the user browses from another machine (see justfile)
    proxy: {
      // Each product's own service, under its own prefix -- the prefix is what
      // keeps two products' APIs from colliding on one dev server.
      '/attendance/api': { ...http(attendancePort), rewrite: p => p.replace(/^\/attendance/, '') },
      '/attendance/ws': { ...ws(attendancePort), rewrite: p => p.replace(/^\/attendance/, '') },
      // xfwd: barber serves customer photos to LAN browsers only, so it needs the
      // browser's address rather than this dev server's.
      '/barber/api': { ...http(barberPort), xfwd: true, rewrite: p => p.replace(/^\/barber/, '') },
      '/barber/ws': { ...ws(barberPort), xfwd: true, rewrite: p => p.replace(/^\/barber/, '') },
      // The fall product's own service (`just fall-up`); it has no WebSocket -- the tab draws the
      // engine's already-published frame and polls the review queue. Its routes are bare
      // (`/status`, not `/api/status` like the other products'), so the prefix goes too.
      '/fall_detection/api': { ...http(fallPort), rewrite: p => p.replace(/^\/fall_detection\/api/, '') },
      '/concealment_detection/api': { ...http(concealmentPort), rewrite: p => p.replace(/^\/concealment_detection\/api/, '') },
      // Shared platform: the camera list, camera frames, and the alert feed
      // (fire/smoke, zones, overcrowding) that products read but do not own.
      '/api': http(irisPort),
      '/stream': http(irisPort),
      '/ws': ws(irisPort),
      '/go2rtc': { ...http(go2rtcPort), ws: true, rewrite: p => p.replace(/^\/go2rtc/, '') },
    },
  },
})
