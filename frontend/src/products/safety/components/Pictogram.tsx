/** Each feature as the outline of the safety sign it belongs to, drawn like IRIS's line icons (currentColor strokes on
 *  a 24 grid): a warning triangle for vehicles and falls, a prohibition ring for restricted zones, a mandatory disc
 *  for PPE, a square for fire and for the camera itself (health). Its colour is IRIS's, per feature (safety.css,
 *  .sf-sign-<feature>). */
import type { ReactNode } from 'react'
import type { Feature } from '../lib/types'

const TRIANGLE = <path d="M12 2.8 L22 20.4 H2 Z" strokeLinejoin="round" />
const RING = <><circle cx="12" cy="12" r="9.6" /><path d="M5.2 5.2 L18.8 18.8" /></>
const DISC = <circle cx="12" cy="12" r="9.6" />
const SQUARE = <rect x="2.6" y="2.6" width="18.8" height="18.8" rx="3" />

const SIGNS: Record<Feature, ReactNode> = {
  proximity: (
    <>
      {TRIANGLE}
      {/* a forklift: body, mast, forks, wheels */}
      <path d="M7.2 17.2 V13.8 H10.6 L11.6 15.6 V17.2 Z M13.8 11 V17.2 M13.8 16.6 H16.4" strokeWidth="1.3"
        strokeLinejoin="round" />
      <circle cx="8.4" cy="18" r="0.6" /><circle cx="10.8" cy="18" r="0.6" />
    </>
  ),
  fall: (
    <>
      {TRIANGLE}
      {/* a person tipping over */}
      <circle cx="14.4" cy="10.8" r="1.2" />
      <path d="M13.2 12.6 L10 15.4 M11.6 14 L8.6 13.4 M10 15.4 L7.8 18 M10 15.4 L12.6 17.8" strokeWidth="1.3" />
    </>
  ),
  zones: (
    <>
      {RING}
      {/* a person, crossed out */}
      <circle cx="12" cy="7.6" r="1.2" />
      <path d="M12 9.6 V14 M12 14 L10.2 17.2 M12 14 L13.8 17.2 M9.8 11.4 H14.2" strokeWidth="1.3" />
    </>
  ),
  ppe: (
    <>
      {DISC}
      {/* a hard hat */}
      <path d="M7.4 14.2 A4.6 4.6 0 0 1 16.6 14.2 M6 14.2 H18 M12 9.6 V12" strokeWidth="1.4" />
    </>
  ),
  fire: (
    <>
      {SQUARE}
      <path d="M12 5.6 C13.4 8.2 16.2 9.6 16.2 13.2 A4.2 4.2 0 0 1 7.8 13.2 C7.8 11.2 9 10 9.8 9.2 C10 10.8 10.6 11.4 11.3 11.8 C11 9.6 11.2 7.6 12 5.6 Z"
        strokeWidth="1.3" strokeLinejoin="round" />
    </>
  ),
  health: (
    <>
      {SQUARE}
      {/* a camera */}
      <path d="M6.4 9.4 H13.8 V14.8 H6.4 Z M13.8 11.2 L17.6 9.4 V14.8 L13.8 13 Z" strokeWidth="1.3"
        strokeLinejoin="round" />
    </>
  ),
}

export function Pictogram({ feature, size = 24, title }: { feature: Feature; size?: number; title?: string }) {
  return (
    <svg className={`sf-sign sf-sign-${feature}`} width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" role={title ? 'img' : undefined}
      aria-label={title} aria-hidden={title ? undefined : true}>
      {SIGNS[feature]}
    </svg>
  )
}
