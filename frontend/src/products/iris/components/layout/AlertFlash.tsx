import { useEffect, useRef, useState } from 'react'
import { useEventStore } from '@/stores/eventStore'
import styles from './AlertFlash.module.css'

export function AlertFlash() {
  // A counter, not an id: an incident raised to critical in place keeps its id and must still flash.
  const flashSeq = useEventStore(s => s.flashSeq)
  // Seeded with the current value: only flashes that happen after mount show (history loads never bump it).
  const seenRef = useRef(flashSeq)
  const [flashKey, setFlashKey] = useState(0)
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    if (flashSeq === seenRef.current) return
    seenRef.current = flashSeq
    setFlashKey(k => k + 1)
    setVisible(true)
    const timer = setTimeout(() => setVisible(false), 1500)
    return () => clearTimeout(timer)
  }, [flashSeq])

  if (!visible) return null
  return <div key={flashKey} className={styles.flash} aria-hidden="true" />
}
