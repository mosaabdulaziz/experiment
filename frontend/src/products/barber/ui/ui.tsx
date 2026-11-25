/** Small shared pieces: nav icons, dialogs. */
import type { ReactNode } from 'react'
import { useLang } from '../i18n'

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
  insights: 'M4 20V10m6 10V4m6 16v-7m4 7H3',
  reports: 'M7 3h7l5 5v13H7zM14 3v5h5M10 13h6m-6 4h6',
  alerts: 'M12 3 2.5 20h19zM12 10v4.5m0 2.7v.1',
  zones: 'M5 7.5 17.5 4.5l2 12.5-13 2.5z',
  customers: 'M9 11a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7zm7.5-1a2.8 2.8 0 1 0 0-5.6M3 20v-.8C3 16.3 5.7 14 9 14s6 2.3 6 5.2v.8m2-6c2.3.3 4 2 4 4.2v.8',
  staff: 'M7 3v8m0 0a3 3 0 1 0 0 6 3 3 0 0 0 0-6zm10-8v8m0 0a3 3 0 1 0 0 6 3 3 0 0 0 0-6zM7 11l10-8M17 11 7 3',
}

export function Confirm({ title, children, confirmLabel, danger = false, onConfirm, onCancel }: {
  title: string
  children: ReactNode
  confirmLabel: string
  danger?: boolean
  onConfirm: () => void
  onCancel: () => void
}) {
  const { t } = useLang()
  return (
    <div className="confirmBackdrop" onClick={onCancel}>
      <div className="confirmBox" role="dialog" aria-modal="true" aria-label={title} onClick={e => e.stopPropagation()}>
        <h3>{title}</h3>
        <div>{children}</div>
        <div className="panelActions" style={{ justifyContent: 'flex-end' }}>
          <button className="btnPlain" onClick={onCancel}>{t('common.cancel')}</button>
          <button className={danger ? 'btnDanger' : 'btnPrimary'} onClick={onConfirm}>{confirmLabel}</button>
        </div>
      </div>
    </div>
  )
}

export function Switch({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <button type="button" role="switch" aria-checked={checked} aria-label={label} className="switch" onClick={() => onChange(!checked)}>
      <span className="switchKnob" />
    </button>
  )
}
