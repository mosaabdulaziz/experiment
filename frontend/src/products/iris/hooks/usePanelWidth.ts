/** A side panel's width, remembered in this browser (per `storageKey`) so it stays as the operator left it. */
import { useCallback, useState } from 'react'
import { clampWidth, type WidthLimits } from '@/components/ui/resize'

function stored(key: string, fallback: number, limits: WidthLimits): number {
  try {
    const v = Number(localStorage.getItem(key))
    return Number.isFinite(v) && v > 0 ? clampWidth(v, limits) : fallback
  } catch {
    return fallback
  }
}

export function usePanelWidth(storageKey: string, defaultWidth: number, limits: WidthLimits) {
  const [width, setWidth] = useState(() => stored(storageKey, defaultWidth, limits))
  const change = useCallback((w: number) => {
    setWidth(w)
    try { localStorage.setItem(storageKey, String(w)) } catch { /* private mode: not remembered */ }
  }, [storageKey])
  return [width, change] as const
}
