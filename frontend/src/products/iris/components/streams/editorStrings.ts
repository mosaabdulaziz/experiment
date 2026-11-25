/** What the zone editor says (ZoneEditorOverlay, ZoneRail, ZoneConfigPanel, ProximitySection, zoneDraft's summaries
 *  and reasons), in English by default. A product shows it in another language by handing its own through
 *  EditorStringsContext (the safety console's Arabic); IRIS itself never does, so its editor reads as it always has. */
import { createContext, useContext } from 'react'
import { ppeItemName } from '@/types/zone'

const titleCase = (s: string) => s.replace(/\b\w/g, c => c.toUpperCase())
const sentence = (s: string) => s.charAt(0).toUpperCase() + s.slice(1)

export const EDITOR_EN = {
  // header, canvas and footer
  live: 'Live',
  off: 'Off',
  on: 'On',
  close: 'Close',
  closeEditor: 'Close zone editor',
  editorLabel: (id: string) => `${id} zone editor`,
  feedAlt: (id: string) => `${id} feed`,
  loadFailed: 'Failed to load zones',
  saveFailed: 'Failed to save zones',
  loading: 'Loading zones…',
  addFirst: 'Add a zone, then draw it on the feed',
  ringClosed: 'Ring closed — press Finish Polygon',
  editing: (name: string) => `Editing ${name} — drag points to adjust`,
  points: (n: number) => `${n} point${n === 1 ? '' : 's'}`,
  closeHint: ' — click the first point to close',
  clear: 'Clear',
  finish: 'Finish Polygon',
  save: 'Save Zones',
  saving: 'Saving…',
  saved: 'Saved',
  savedNoAnalytics: 'Saved — analytics disabled',
  applying: 'Applying to pipeline…',
  active: 'Zones active',
  applyTimeout: "Saved — pipeline hasn't confirmed yet. It will apply when the sidecar reconnects.",
  // the zone list
  zones: 'Zones',
  noZones: 'No zones yet. Add one, then draw it on the feed.',
  addZone: 'Add zone',
  zoneName: 'Zone name',
  deleteZone: (name: string) => `Delete ${name}`,
  newZone: (n: number) => `Zone ${n}`,
  zoneFallback: 'Zone',
  // a zone's settings
  intrusion: 'Intrusion',
  intrusionMode: 'Intrusion mode',
  restricted: 'Restricted',
  valid: 'Valid',
  restrictedHint: 'Alerts when someone enters this zone.',
  validHint: 'Alerts when someone leaves this zone.',
  alertOn: 'Alert on',
  overcrowding: 'Overcrowding',
  threshold: 'People threshold',
  thresholdHint: 'Alerts when this many are inside at once.',
  count: 'Count',
  ppe: 'PPE',
  ppeHint: 'Workers standing in this zone must wear what is picked above.',
  ppeHintNoChoice: 'Workers standing in this zone must wear their PPE.',
  checkFor: 'Check for',
  falls: 'Falls',
  fallsHint: 'Falls are watched in this zone.',
  fire: 'Fire and smoke',
  fireHint: 'Fire and smoke are watched in this zone.',
  // proximity
  proximity: 'Proximity',
  proximityHint: 'Alerts when a person stays this close to a moving vehicle.',
  vehicles: 'Vehicles',
  maxGap: 'Max gap',
  maxGapHint: (gap: number, metres: number, critical: number) =>
    `${gap.toFixed(2)} person heights ≈ ${metres.toFixed(1)} m · critical below ${critical.toFixed(2)}`,
  depthCheck: 'Depth check',
  noDepthModel: ' — no depth model on this pipeline',
  depthCutoff: 'Depth cut-off',
  depthCutoffHint: (gap: number) =>
    `${gap.toFixed(2)} relative depth gap · pairs further apart are held back (Incidents → Held back by depth)`,
  advanced: 'Advanced',
  persistence: 'Persistence',
  persistenceUnit: 's close before it alerts',
  grace: 'Grace',
  graceUnit: 's a pair may drop out',
  cooldown: 'Cooldown',
  cooldownUnit: 's before a new incident',
  stationaryWindow: 'Stationary window',
  seconds: 's',
  stationaryMovement: 'Stationary movement',
  personHeights: 'person heights',
  hiddenFeet: 'Hidden feet',
  hiddenFeetFlag: 'Hidden feet: flag as unconfirmed',
  hiddenFeetIgnore: 'Hidden feet: ignore',
  // the zone list's one-line summary, and why a zone cannot be saved yet (zoneDraft.ts)
  sumRestricted: 'Restricted',
  sumValid: 'Valid',
  sumCrowd: (n: number) => `Crowd ≥${n}`,
  sumProximity: 'Proximity',
  sumPpe: (items: string[] | undefined) => (items ? `PPE: ${items.map(ppeItemName).join(' + ')}` : 'PPE'),
  sumFalls: 'Falls',
  sumFire: 'Fire',
  sumNone: 'No detection',
  separator: ' · ',
  finishDrawing: (name: string) => `Finish drawing ${name}`,
  turnOn: (name: string) => `Turn on a feature for ${name}`,
  pickIntrusion: (name: string) => `Pick an intrusion class for ${name}`,
  pickOvercrowding: (name: string) => `Pick an overcrowding class for ${name}`,
  pickVehicle: (name: string) => `Pick a vehicle class for ${name}`,
  pickPpe: (name: string) => `Pick the equipment PPE checks in ${name}`,
  // what the pills say: a detector's class, a PPE item
  className: (cls: string) => titleCase(cls),
  ppeItem: (item: string) => sentence(ppeItemName(item)),
}

export type EditorStrings = typeof EDITOR_EN

export const EditorStringsContext = createContext<EditorStrings>(EDITOR_EN)

export function useEditorStrings(): EditorStrings {
  return useContext(EditorStringsContext)
}
