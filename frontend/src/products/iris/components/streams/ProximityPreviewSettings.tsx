import { useEffect, useId, useRef, useState } from 'react'
import {
  streamsApi,
  type ProximityClipStyle,
  type ProximityPreview,
  type ProximityPreviewChange,
  type ProximityPreviewLayer,
} from '@/api/streams'
import { IconChevronDown, IconGear } from '@/components/ui/icons'
import styles from './StreamGrid.module.css'

const REFRESH_MS = 10_000 // picks up a change another viewer made
const RANGE_SAVE_MS = 300 // the range is saved once the slider rests this long

const LAYERS: { key: ProximityPreviewLayer; label: string; what: string }[] = [
  { key: 'boxes', label: 'Boxes', what: 'boxes and their labels' },
  { key: 'masks', label: 'Masks', what: 'person and vehicle masks' },
  { key: 'lines', label: 'Distance lines', what: 'person–vehicle distance lines' },
  { key: 'depth', label: 'Depth map', what: 'the depth model’s map (cameras with the depth check)' },
]

const CLIP_STYLES: { key: ProximityClipStyle; label: string }[] = [
  { key: 'masks', label: 'Masks' },
  { key: 'boxes', label: 'Boxes' },
]

/** The proximity pipeline's live-view layers, line range and alert-clip style, behind a "Preview settings" button
 *  that slides them open to its left. Set for every camera and every viewer (the pipeline picks a change up within
 *  about a second; a clip takes the style it is built with). Hidden when the pipeline does not run proximity. */
export function ProximityPreviewSettings() {
  const [preview, setPreview] = useState<ProximityPreview | null>(null)
  const [open, setOpen] = useState(false)
  const [busy, setBusy] = useState(false)
  const [range, setRange] = useState<number | null>(null) // the slider while it moves, until saved
  const rangeTimer = useRef<number | null>(null)
  const panelId = useId()

  useEffect(() => {
    let alive = true
    const load = () =>
      streamsApi
        .proximityPreview()
        .then(p => {
          if (alive) setPreview(p)
        })
        .catch(() => {
          /* keep what is shown; the next refresh tries again */
        })
    load()
    const timer = window.setInterval(load, REFRESH_MS)
    return () => {
      alive = false
      window.clearInterval(timer)
      if (rangeTimer.current != null) window.clearTimeout(rangeTimer.current)
    }
  }, [])

  if (!preview?.available) return null

  const save = async (change: ProximityPreviewChange) => {
    const before = preview
    setPreview({ ...before, ...change })
    setBusy(true)
    try {
      setPreview(await streamsApi.setProximityPreview(change))
    } catch {
      setPreview(before)
    } finally {
      setBusy(false)
    }
  }

  const slide = (value: number) => {
    setRange(value)
    if (rangeTimer.current != null) window.clearTimeout(rangeTimer.current)
    rangeTimer.current = window.setTimeout(() => {
      rangeTimer.current = null
      void save({ line_range: value }).finally(() => setRange(null))
    }, RANGE_SAVE_MS)
  }

  const shownRange = range ?? preview.line_range

  return (
    <div className={styles.previewMenu}>
      <div
        id={panelId}
        className={`${styles.previewPanel} ${open ? styles.previewPanelOpen : ''}`}
        aria-hidden={!open}
      >
        {LAYERS.map(({ key, label, what }) => (
          <button
            key={key}
            type="button"
            role="switch"
            aria-checked={preview[key]}
            className={styles.zonesSwitch}
            onClick={() => save({ [key]: !preview[key] })}
            disabled={busy}
            title={`${preview[key] ? 'Hide' : 'Show'} the ${what} on every camera's live view (for all viewers)`}
          >
            <span className={styles.zonesSwitchLabel}>{label}</span>
            <span className={`${styles.zonesSwitchTrack} ${preview[key] ? styles.zonesSwitchTrackOn : ''}`}>
              <span className={styles.zonesSwitchKnob} />
            </span>
          </button>
        ))}
        {preview.lines && (
          <label
            className={styles.lineRange}
            title="How far apart a person and a moving vehicle can be and still get a distance line, in person heights (for all viewers). Pairs within a zone's max gap always get one."
          >
            <span className={styles.zonesSwitchLabel}>Line range</span>
            <input
              type="range"
              min={0.5}
              max={5}
              step={0.25}
              value={shownRange}
              onChange={e => slide(Number(e.target.value))}
              className={styles.lineRangeInput}
              aria-valuetext={`${shownRange} person heights`}
            />
            <span className={styles.lineRangeValue}>{shownRange.toFixed(2)} ph</span>
          </label>
        )}
        <div
          className={styles.clipStyle}
          role="group"
          aria-label="Alert clips draw"
          title="How alert clips and snapshots draw the incident's person and vehicle (clips made from now on)"
        >
          <span className={styles.zonesSwitchLabel}>Clips</span>
          <span className={styles.segmented}>
            {CLIP_STYLES.map(({ key, label }) => (
              <button
                key={key}
                type="button"
                aria-pressed={preview.clip_style === key}
                className={`${styles.segment} ${preview.clip_style === key ? styles.segmentOn : ''}`}
                onClick={() => preview.clip_style !== key && save({ clip_style: key })}
                disabled={busy}
              >
                {label}
              </button>
            ))}
          </span>
        </div>
      </div>
      <button
        type="button"
        className={styles.previewToggle}
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen(!open)}
        title={open ? 'Hide the preview settings' : 'Show the preview settings'}
      >
        <IconGear width={14} height={14} />
        <span>Preview settings</span>
        <IconChevronDown className={`${styles.previewChevron} ${open ? styles.previewChevronOpen : ''}`} />
      </button>
    </div>
  )
}
