import { useEffect, useRef, useState } from 'react'
import type { RefObject } from 'react'

/**
 * True once the element has scrolled near the viewport, and stays true afterwards.
 *
 * One-way by design: the incidents grid uses this to decide when to mount a <video>
 * and fetch its metadata, and un-mounting those again on scroll-away would re-issue
 * the same requests every time the user scrolls back.
 */
export function useInView<T extends HTMLElement>(
  rootMargin = '200px',
): [RefObject<T | null>, boolean] {
  const ref = useRef<T>(null)
  const [inView, setInView] = useState(false)

  useEffect(() => {
    const el = ref.current
    if (!el || inView) return
    if (typeof IntersectionObserver === 'undefined') {
      setInView(true)
      return
    }
    const observer = new IntersectionObserver(
      entries => {
        if (entries.some(e => e.isIntersecting)) setInView(true)
      },
      { rootMargin },
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [rootMargin, inView])

  return [ref, inView]
}
