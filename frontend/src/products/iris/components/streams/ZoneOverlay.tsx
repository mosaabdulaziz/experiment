import { useEffect, useMemo, useRef, useState } from 'react'
import { useStreamsStore } from '@/stores/streamsStore'
import { zonesApi } from '@/api/zones'
import type { Zone, ZonePoint, IntrusionMode } from '@/types/zone'
import styles from './ZoneOverlay.module.css'

interface Props {
  streamId: string
  /** Rendered size of the tile's video box (px). */
  width: number
  height: number
  /** Intrinsic size of the MJPEG frame, captured from the tile's <img> onLoad. */
  natural: { width: number; height: number } | null
}

// Same color language as the editor + live boxes: restricted → accent, valid → green,
// overcrowding-only (intrusion off) → yellow.
function zoneColor(mode: IntrusionMode): string {
  const colorVar = mode === 'valid' ? '--green' : mode === 'off' ? '--yellow' : '--accent'
  const fallback = mode === 'valid' ? '#6bf2a8' : mode === 'off' ? '#ffd166' : '#ff8a4c'
  return getComputedStyle(document.documentElement).getPropertyValue(colorVar).trim() || fallback
}

/** Read-only overlay for a dashboard tile: draws all of the stream's saved zones on a canvas
 *  over the live feed, mapped through the same object-fit: contain math the zone editor uses so
 *  they line up with the letterboxed image. Purely presentational — no hit-testing or editing. */
export function ZoneOverlay({ streamId, width, height, natural }: Props) {
  // Refetch when a zone edit lands (applied version bumps) so the outlines stay current.
  const appliedVersion = useStreamsStore(s => s.zonesStatus?.appliedVersion ?? 0)
  const [zones, setZones] = useState<Zone[]>([])
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    let cancelled = false
    zonesApi
      .list(streamId)
      .then(res => { if (!cancelled) setZones(res.zones ?? []) })
      .catch(() => { if (!cancelled) setZones([]) }) // no zones / fetch failed → draw nothing
    return () => { cancelled = true }
  }, [streamId, appliedVersion])

  // object-fit: contain — the true displayed sub-rect of the feed within the tile box.
  const displayRect = useMemo(() => {
    if (!width || !height || !natural?.width || !natural?.height) return null
    const boxAspect = width / height
    const imgAspect = natural.width / natural.height
    if (imgAspect > boxAspect) {
      const w = width
      const h = width / imgAspect
      return { x: 0, y: (height - h) / 2, w, h }
    }
    const h = height
    const w = height * imgAspect
    return { x: (width - w) / 2, y: 0, w, h }
  }, [width, height, natural])

  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx) return
    canvas.width = width
    canvas.height = height
    ctx.clearRect(0, 0, width, height)
    if (!displayRect) return

    const toPx = (p: ZonePoint): [number, number] => [
      p.x * displayRect.w + displayRect.x,
      p.y * displayRect.h + displayRect.y,
    ]

    for (const zone of zones) {
      if (zone.points.length < 2) continue
      const color = zoneColor(zone.intrusion.mode)
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.beginPath()
      zone.points.forEach((p, i) => {
        const [x, y] = toPx(p)
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      if (zone.points.length >= 3) {
        ctx.closePath()
        ctx.globalAlpha = 0.18
        ctx.fillStyle = color
        ctx.fill()
        ctx.globalAlpha = 1
      }
      ctx.stroke()
    }
  }, [zones, width, height, displayRect])

  return <canvas ref={canvasRef} className={styles.canvas} aria-hidden="true" />
}
