/** Small shared pieces: the nav icons, the dialogs. The house's own (barber/ui/ui.tsx). */
import type { ReactNode } from 'react'

export function Icon({ d, stroke = false, size = 20 }: { d: string; stroke?: boolean; size?: number }) {
  return (
    <svg viewBox="0 0 24 24" width={size} height={size} aria-hidden="true">
      <path
        d={d}
        fill={stroke ? 'none' : 'currentColor'}
        stroke={stroke ? 'currentColor' : 'none'}
        strokeWidth={stroke ? 1.9 : 0}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

export const ICONS = {
  live: 'M4 5h16a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1zm4 4v6l5-3z',
  review: 'M8 4h8a2 2 0 0 1 2 2v14l-6-3-6 3V6a2 2 0 0 1 2-2z',
  fall: 'M12 3 2.5 20h19zM12 10v4.5m0 2.7v.1',
}

/** The house's modal, copied from barber/ui/ui.tsx:30-51.
 *
 * The house's copy asks the bilingual provider for the cancel label; this product is
 * English-only with no provider (Global Constraint 4), so the label is the literal word. */
export function Confirm({ title, children, confirmLabel, danger = false, onConfirm, onCancel }: {
  title: string
  children: ReactNode
  confirmLabel: string
  danger?: boolean
  onConfirm: () => void
  onCancel: () => void
}) {
  return (
    <div className="confirmBackdrop" onClick={onCancel}>
      <div className="confirmBox" role="dialog" aria-modal="true" aria-label={title} onClick={e => e.stopPropagation()}>
        <h3>{title}</h3>
        <div>{children}</div>
        <div className="panelActions" style={{ justifyContent: 'flex-end' }}>
          <button className="btnPlain" onClick={onCancel}>Cancel</button>
          <button className={danger ? 'btnDanger' : 'btnPrimary'} onClick={onConfirm}>{confirmLabel}</button>
        </div>
      </div>
    </div>
  )
}
