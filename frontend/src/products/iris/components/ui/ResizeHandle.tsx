/** The grab bar on a side panel's inner edge: drag it (mouse, pen or touch) to make the panel wider or slimmer, use the
 *  arrow keys / Home / End when it has focus, double-click it to go back to the default width. Place it inside a
 *  `position: relative` wrapper around the panel; it sits in the gap beside the panel. */
import { useRef, useState, type KeyboardEvent, type PointerEvent } from 'react'
import { dragWidth, keyWidth, type WidthLimits } from './resize'
import styles from './ResizeHandle.module.css'

interface Props {
  width: number
  onChange: (width: number) => void
  limits: WidthLimits
  defaultWidth: number
  label: string
}

const isRtl = (el: Element) => getComputedStyle(el).direction === 'rtl'

export function ResizeHandle({ width, onChange, limits, defaultWidth, label }: Props) {
  const drag = useRef<{ start: number; startX: number; rtl: boolean } | null>(null)
  const [dragging, setDragging] = useState(false)

  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return
    e.preventDefault()
    e.currentTarget.setPointerCapture(e.pointerId)
    drag.current = { start: width, startX: e.clientX, rtl: isRtl(e.currentTarget) }
    setDragging(true)
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
  }
  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    const d = drag.current
    if (d) onChange(dragWidth(d.start, d.startX, e.clientX, d.rtl, limits))
  }
  const end = () => {
    if (!drag.current) return
    drag.current = null
    setDragging(false)
    document.body.style.cursor = ''
    document.body.style.userSelect = ''
  }
  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const next = keyWidth(width, e.key, isRtl(e.currentTarget), limits)
    if (next === null) return
    e.preventDefault()
    onChange(next)
  }

  return (
    <div
      className={styles.handle}
      role="separator"
      aria-orientation="vertical"
      aria-label={label}
      aria-valuemin={limits.min}
      aria-valuemax={limits.max}
      aria-valuenow={width}
      tabIndex={0}
      title={label}
      data-dragging={dragging || undefined}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={end}
      onPointerCancel={end}
      onLostPointerCapture={end}
      onKeyDown={onKeyDown}
      onDoubleClick={() => onChange(defaultWidth)}
    >
      <span className={styles.grip} aria-hidden="true" />
    </div>
  )
}
