/** What "Discard changes" on one camera must undo elsewhere. An area edit on
 * one camera can change another (setOwner clears the owner on the pair's
 * other cameras), so reverting only the current camera could leave the
 * other one half-edited. Rule: the pair ids (old and new) of every zone that
 * changed on `cam` are the linked areas; a configured camera with a changed
 * zone in one of those areas is a linked camera. Discard reverts `cam`
 * fully and, on linked cameras, only the zones of the linked areas -- their
 * unrelated unsaved edits stay. */
import type { Zone, ZonesByCam } from '../../api/types'

const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b)

/** Pair ids (before and after) of the zones that differ between draft and saved. */
function changedPairs(draft: Zone[] = [], saved: Zone[] = []): Set<string> {
  const before = new Map(saved.map(z => [z.id, z]))
  const after = new Map(draft.map(z => [z.id, z]))
  const pairs = new Set<string>()
  for (const id of new Set([...before.keys(), ...after.keys()])) {
    const a = before.get(id)
    const b = after.get(id)
    if (same(a, b)) continue
    if (a?.pair) pairs.add(a.pair)
    if (b?.pair) pairs.add(b.pair)
  }
  return pairs
}

/** The linked areas and the linked cameras (other than `cam`, in `cameras` order). */
export function linkedScope(draft: ZonesByCam, saved: ZonesByCam, cam: string, cameras: string[]): { pairs: string[]; cameras: string[] } {
  const pairs = changedPairs(draft[cam], saved[cam])
  const linked = cameras.filter(c => c !== cam && [...changedPairs(draft[c], saved[c])].some(p => pairs.has(p)))
  return { pairs: [...pairs].sort(), cameras: linked }
}

/** `cam` first, then the linked cameras. */
export function camerasToRevert(draft: ZonesByCam, saved: ZonesByCam, cam: string, cameras: string[]): string[] {
  return [cam, ...linkedScope(draft, saved, cam, cameras).cameras]
}

/** One camera's draft with the zones of `pairs` -- and the seats of those
 * chairs -- put back as saved (an added one dropped, a deleted one restored
 * in its saved place); every other zone as in the draft. */
export function revertPairs(draft: Zone[] = [], saved: Zone[] = [], pairs: Set<string>): Zone[] {
  const paired = (z?: Zone) => !!z?.pair && pairs.has(z.pair)
  // seats carry no pair: one goes with its chair (seat_of, before or after) if that chair is in scope
  const scopedIds = new Set([...draft, ...saved].filter(z => paired(z)).map(z => z.id))
  const inScope = (z?: Zone) => paired(z) || (z?.type === 'seat' && !!z.seat_of && scopedIds.has(z.seat_of))
  const savedById = new Map(saved.map(z => [z.id, z]))
  const out: Zone[] = []
  for (const z of draft) {
    const s = savedById.get(z.id)
    if (inScope(z) || inScope(s)) { if (s) out.push(s) }      // added into a linked area: dropped
    else out.push(z)
  }
  // restore deleted linked zones right after the saved zone they followed
  saved.forEach((s, i) => {
    if (!inScope(s) || out.some(z => z.id === s.id)) return
    let at = 0
    for (let j = i - 1; j >= 0; j--) {
      const k = out.findIndex(z => z.id === saved[j].id)
      if (k >= 0) { at = k + 1; break }
    }
    out.splice(at, 0, s)
  })
  return out
}

/** The draft after "Discard changes" on `cam`. */
export function revertLinked(draft: ZonesByCam, saved: ZonesByCam, cam: string, cameras: string[]): ZonesByCam {
  const { pairs, cameras: linked } = linkedScope(draft, saved, cam, cameras)
  const set = new Set(pairs)
  const next: ZonesByCam = { ...draft, [cam]: saved[cam] ?? [] }
  for (const c of linked) next[c] = revertPairs(draft[c], saved[c], set)
  return next
}
