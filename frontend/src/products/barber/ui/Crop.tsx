/** A customer crop served by barber-service (LAN clients only). Off the LAN
 * (403), after an erase or the retention sweep (404), or with no photo at
 * all, it shows a neutral silhouette instead -- never a broken image. */
import { useEffect, useState } from 'react'
import { cropUrl } from '../api/client'

export function Crop({ path, alt, className = '' }: { path: string | null; alt: string; className?: string }) {
  const [failed, setFailed] = useState(false)
  useEffect(() => { setFailed(false) }, [path])
  if (!path || failed) return <Silhouette label={alt} className={className} />
  return <img className={`crop ${className}`.trim()} src={cropUrl(path)} alt={alt} loading="lazy" decoding="async"
              onError={() => setFailed(true)} />
}

function Silhouette({ label, className }: { label: string; className: string }) {
  return (
    <svg className={`crop cropSilhouette ${className}`.trim()} viewBox="0 0 100 100" preserveAspectRatio="xMidYMax meet"
         {...(label ? { role: 'img', 'aria-label': label } : { 'aria-hidden': true })}>
      <circle cx="50" cy="38" r="17" />
      <path d="M16 100c0-21 15-34 34-34s34 13 34 34z" />
    </svg>
  )
}
