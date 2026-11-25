/** An incident's message in the console's language. The app writes its messages in English; in Arabic the console
 *  words them itself from what the event keeps as data (its details and tags: PPE's missing items, the fall service's
 *  evaluation and confidence, fire's class, an intrusion's object and zone, proximity's gap and vehicle, a health
 *  problem) -- and shows the app's English for an event that keeps too little to say it (an older one). Either way a
 *  camera is named, never its stream key. Pure. */
import type { Key, Lang } from '../i18n'
import { en } from '../i18n/en'
import { CLASS_AR, PPE_AR } from '../i18n/editor'
import type { SafetyEvent } from './types'

type T = (key: Key, vars?: Record<string, string | number>) => string

const text = (v: unknown): string | null => (typeof v === 'string' && v.trim() ? v.trim() : null)
const number = (v: unknown): number | null => {
  const n = typeof v === 'number' ? v : typeof v === 'string' && v.trim() ? Number(v) : NaN
  return Number.isFinite(n) ? n : null
}
const has = (key: string): key is Key => key in en
/** "a, b and c" in Arabic: "أ وب وج". */
const and = (items: string[]) => items.join(' و')

export function messageOf(ev: SafetyEvent, lang: Lang, t: T, cameraName: (id: string | null) => string): string {
  return (lang === 'en' ? null : arabic(ev, t, cameraName)) ?? named(ev, cameraName)
}

/** The app's words, with the camera's name where they say its stream key ("Fire detected on store@fire"). */
function named(ev: SafetyEvent, cameraName: (id: string | null) => string): string {
  if (!ev.camera?.includes('@') || !ev.camera_id) return ev.message
  return ev.message.split(ev.camera).join(cameraName(ev.camera_id))
}

function arabic(ev: SafetyEvent, t: T, cameraName: (id: string | null) => string): string | null {
  const d = (ev.details ?? {}) as Record<string, unknown>
  const camera = cameraName(ev.camera_id)
  const tags = ev.tags ?? []
  switch (ev.feature) {
    case 'ppe': {
      const items = (text(d.missing) ?? '').split(',').map(s => s.trim()).filter(Boolean)
      const id = number(d.track_id)
      if (!items.length || id === null) return null
      let out = t('msg.ppe', { id, items: and(items.map(i => PPE_AR[i] ?? i)) })
      const again = number(d.possible_duplicate_of)
      if (again !== null) {
        const seconds = /alerted (\d+) s earlier/.exec(ev.message)
        out += seconds ? t('msg.ppeAgain', { id: again, s: Number(seconds[1]) }) : t('msg.ppeAgainNoTime', { id: again })
      }
      return out
    }
    case 'fall': {
      const evaluation = text(d.evaluation)
      const confidence = number(d.confidence)
      if (!evaluation || confidence === null) return null
      const review = d.review === '1'
      const key = review ? `msg.fallMay.${evaluation}` : `review.${evaluation}`
      if (!has(key)) return null
      return t(review ? 'msg.fallReview' : 'msg.fall', { what: t(key), p: Math.round(confidence * 100) })
    }
    case 'fire': {
      const cls = text(d.class) ?? (/^smoke/i.test(ev.message) ? 'smoke' : /^fire/i.test(ev.message) ? 'fire' : null)
      if (cls !== 'fire' && cls !== 'smoke') return null
      return t(cls === 'smoke' ? 'msg.smoke' : 'msg.fire', { camera })
    }
    case 'zones': {
      if (tags.includes('zone_overcrowding')) {
        const said = /— (.+?)(?: \((\d+) people\))?$/.exec(ev.message)
        const zone = text(d.zone_name) ?? said?.[1] ?? null
        const count = number(d.count) ?? (said?.[2] ? Number(said[2]) : null)
        if (!zone) return t('msg.overcrowdingCamera', { camera })
        return count !== null ? t('msg.overcrowding', { zone, n: count }) : t('msg.overcrowdingZone', { zone })
      }
      if (!tags.includes('zone_intrusion') && !tags.includes('zone_exit')) return null
      const label = text(d.object_label)
      const what = label ? (CLASS_AR[label] ?? label) : t('msg.someone')
      const id = number(d.object_id)
      const subject = id !== null ? `${what} #${new Intl.NumberFormat('ar-EG').format(id)}` : what
      const zone = text(d.zone_name)
      const left = tags.includes('zone_exit')
      if (!zone) return t(left ? 'msg.leftValid' : 'msg.enteredRestricted', { subject })
      return t(left ? 'msg.left' : 'msg.entered', { subject, zone })
    }
    case 'proximity': {
      const gap = number(d.gap)
      const cls = text(d.vehicle_class)
      if (gap === null || !cls) return null
      const zone = text(d.zone_name)
      const said = t('msg.proximity', { gap, vehicle: CLASS_AR[cls] ?? cls, where: zone ? t('msg.inZone', { zone }) : '' })
      return d.suppressed === 'depth' || tags.includes('depth-suppressed') ? t('msg.proximityHeld', { text: said }) : said
    }
    case 'health': {
      const role = text(d.role)
      const feature = role && has(`feature.${role}`) ? t(`feature.${role}` as Key) : role ?? ''
      const problem = d.problem === 'offline' ? t('health.offline')
        : d.problem === 'feature_down' ? t('health.featureDown', { feature })
          : d.problem === 'slow' ? t('health.slow', { feature }) : null
      return problem ? t('msg.health', { camera, problem }) : null
    }
    default:
      return null
  }
}
