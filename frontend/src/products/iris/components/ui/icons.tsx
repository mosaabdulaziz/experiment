import type { SVGProps } from 'react'

type IconProps = SVGProps<SVGSVGElement>

const base = (props: IconProps) => ({
  width: 12,
  height: 12,
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 2,
  ...props,
})

export const IconUpload = (props: IconProps) => (
  <svg viewBox="0 0 16 16" {...base(props)}>
    <path d="M8 2v8M4 6l4-4 4 4M2 13h12" />
  </svg>
)

export const IconWebcam = (props: IconProps) => (
  <svg viewBox="0 0 24 24" {...base(props)}>
    <rect x="2" y="6" width="15" height="12" rx="2" />
    <path d="M17 9l5-3v12l-5-3" />
  </svg>
)

export const IconPause = (props: IconProps) => (
  <svg viewBox="0 0 24 24" width={12} height={12} fill="currentColor" {...props}>
    <rect x="6" y="4" width="4" height="16" />
    <rect x="14" y="4" width="4" height="16" />
  </svg>
)

export const IconPlay = (props: IconProps) => (
  <svg viewBox="0 0 24 24" width={12} height={12} fill="currentColor" {...props}>
    <polygon points="5,3 19,12 5,21" />
  </svg>
)

export const IconClose = (props: IconProps) => (
  <svg viewBox="0 0 16 16" {...base({ width: 11, height: 11, ...props })}>
    <path d="M3 3l10 10M13 3L3 13" />
  </svg>
)

export const IconChevronDown = (props: IconProps) => (
  <svg viewBox="0 0 16 16" {...base({ strokeLinecap: 'round', strokeLinejoin: 'round', ...props })}>
    <path d="M2 5l6 6 6-6" />
  </svg>
)

export const IconSend = (props: IconProps) => (
  <svg
    viewBox="0 0 24 24"
    width={16}
    height={16}
    fill="none"
    stroke="currentColor"
    strokeWidth={2.5}
    strokeLinecap="round"
    {...props}
  >
    <line x1="22" y1="2" x2="11" y2="13" />
    <polygon points="22 2 15 22 11 13 2 9 22 2" />
  </svg>
)

export const IconEye = (props: IconProps) => (
  <svg viewBox="0 0 24 24" {...base({ strokeWidth: 1, ...props })}>
    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
    <circle cx="12" cy="12" r="3" />
  </svg>
)

export const IconCamera = (props: IconProps) => (
  <svg viewBox="0 0 24 24" {...base({ strokeWidth: 1.5, ...props })}>
    <rect x="2" y="6" width="15" height="12" rx="2" />
    <path d="M17 9l5-3v12l-5-3" />
  </svg>
)

export const IconLines = (props: IconProps) => (
  <svg viewBox="0 0 16 16" {...base({ strokeWidth: 1.8, ...props })}>
    <path d="M2 4h12M2 8h8M2 12h5" />
  </svg>
)

export const IconChat = (props: IconProps) => (
  <svg viewBox="0 0 16 16" {...base({ strokeWidth: 1.5, ...props })}>
    <path d="M14 10a2 2 0 01-2 2H5l-3 3V4a2 2 0 012-2h8a2 2 0 012 2v6z" />
  </svg>
)

export const IconCircle = (props: IconProps) => (
  <svg viewBox="0 0 16 16" width={11} height={11} fill="currentColor" {...props}>
    <circle cx="8" cy="8" r="4" />
    <circle cx="8" cy="8" r="7" fill="none" stroke="currentColor" strokeWidth={1.5} />
  </svg>
)

export const IconExpand = (props: IconProps) => (
  <svg viewBox="0 0 16 16" {...base({ width: 13, height: 13, strokeLinecap: 'round', strokeLinejoin: 'round', ...props })}>
    <path d="M6 2H2v4M10 2h4v4M10 14h4v-4M6 14H2v-4" />
  </svg>
)

export const IconSearch = (props: IconProps) => (
  <svg viewBox="0 0 16 16" {...base({ width: 12, height: 12, ...props })}>
    <circle cx="7" cy="7" r="5" />
    <path d="M11 11l3 3" />
  </svg>
)

export const IconChart = (props: IconProps) => (
  <svg viewBox="0 0 24 24" {...base({ width: 16, height: 16, strokeLinecap: 'round', strokeLinejoin: 'round', ...props })}>
    <path d="M3 3v18h18" />
    <path d="M7 14l4-4 4 4 5-6" />
  </svg>
)

export const IconDashboard = (props: IconProps) => (
  <svg viewBox="0 0 24 24" {...base({ width: 16, height: 16, strokeLinecap: 'round', strokeLinejoin: 'round', ...props })}>
    <rect x="3" y="3" width="7" height="9" rx="1" />
    <rect x="14" y="3" width="7" height="5" rx="1" />
    <rect x="14" y="12" width="7" height="9" rx="1" />
    <rect x="3" y="16" width="7" height="5" rx="1" />
  </svg>
)

export const IconPencil = (props: IconProps) => (
  <svg viewBox="0 0 16 16" {...base({ strokeLinecap: 'round', strokeLinejoin: 'round', ...props })}>
    <path d="M11.3 1.7l3 3L5 14H2v-3z" />
    <path d="M9.5 3.5l3 3" />
  </svg>
)

export const IconZone = (props: IconProps) => (
  <svg viewBox="0 0 16 16" {...base({ strokeLinecap: 'round', strokeLinejoin: 'round', ...props })}>
    <path d="M8 2l5.5 4.2-2.1 6.6H4.6L2.5 6.2z" />
    <circle cx="8" cy="2" r="1.15" fill="currentColor" stroke="none" />
    <circle cx="13.5" cy="6.2" r="1.15" fill="currentColor" stroke="none" />
    <circle cx="2.5" cy="6.2" r="1.15" fill="currentColor" stroke="none" />
  </svg>
)

export const IconPlus = (props: IconProps) => (
  <svg viewBox="0 0 16 16" {...base({ strokeLinecap: 'round', ...props })}>
    <path d="M8 3v10M3 8h10" />
  </svg>
)

export const IconTrash = (props: IconProps) => (
  <svg viewBox="0 0 16 16" {...base({ width: 13, height: 13, strokeLinecap: 'round', strokeLinejoin: 'round', ...props })}>
    <path d="M2.5 4h11M6 4V2.5h4V4M4 4l.6 9a1 1 0 001 .9h4.8a1 1 0 001-.9L12 4M6.5 7v4M9.5 7v4" />
  </svg>
)

export const IconAlert = (props: IconProps) => (
  <svg viewBox="0 0 24 24" {...base({ strokeLinecap: 'round', strokeLinejoin: 'round', ...props })}>
    <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
    <line x1="12" y1="9" x2="12" y2="13" />
    <line x1="12" y1="17" x2="12.01" y2="17" />
  </svg>
)

export const IconGear = (props: IconProps) => (
  <svg
    viewBox="0 0 24 24"
    {...base({ width: 16, height: 16, strokeLinecap: 'round', strokeLinejoin: 'round', ...props })}
  >
    <circle cx="12" cy="12" r="3.2" />
    <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 11-2.83 2.83l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 11-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 11-2.83-2.83l.06-.06a1.65 1.65 0 00.33-1.82 1.65 1.65 0 00-1.51-1H3a2 2 0 110-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 112.83-2.83l.06.06a1.65 1.65 0 001.82.33H9a1.65 1.65 0 001-1.51V3a2 2 0 114 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 112.83 2.83l-.06.06a1.65 1.65 0 00-.33 1.82V9a1.65 1.65 0 001.51 1H21a2 2 0 110 4h-.09a1.65 1.65 0 00-1.51 1z" />
  </svg>
)

export const IconUser = (props: IconProps) => (
  <svg viewBox="0 0 24 24" {...base({ width: 16, height: 16, strokeLinecap: 'round', strokeLinejoin: 'round', ...props })}>
    <path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2" />
    <circle cx="12" cy="7" r="4" />
  </svg>
)

export const IconFilm = (props: IconProps) => (
  <svg viewBox="0 0 24 24" {...base({ width: 22, height: 22, strokeWidth: 1.4, ...props })}>
    <rect x="2" y="4" width="20" height="16" rx="2" />
    <path d="M7 4v16M17 4v16M2 12h20M2 8h5M2 16h5M17 8h5M17 16h5" />
  </svg>
)

export const IconIncidents = (props: IconProps) => (
  <svg viewBox="0 0 24 24" {...base({ width: 16, height: 16, strokeLinecap: 'round', strokeLinejoin: 'round', ...props })}>
    <rect x="2" y="5" width="14" height="14" rx="2" />
    <path d="M16 10l6-3v10l-6-3" />
    <circle cx="9" cy="12" r="2.5" fill="currentColor" stroke="none" />
  </svg>
)
