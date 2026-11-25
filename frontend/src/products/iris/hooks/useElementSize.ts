import { useEffect, useRef, useState } from 'react'

/** Tracks an element's content-box size via ResizeObserver. */
export function useElementSize<T extends HTMLElement = HTMLElement>() {
  const ref = useRef<T | null>(null)
  const [size, setSize] = useState({ width: 0, height: 0 })

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(entries => {
      const rect = entries[0].contentRect
      setSize({ width: rect.width, height: rect.height })
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  return { ref, width: size.width, height: size.height }
}
