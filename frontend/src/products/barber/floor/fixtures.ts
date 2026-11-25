import type { Zone, ZonesByCam } from '../api/types'

const sq = (x: number, y: number): number[][] => [[x, y], [x + 0.05, y], [x + 0.05, y + 0.05], [x, y + 0.05]]
const z = (p: Partial<Zone> & Pick<Zone, 'id' | 'type'>): Zone => ({ name: p.id, points: sq(0.1, 0.1), ...p })

/** A three-camera shop used by both areas.test.ts and selectors.test.ts:
 * a centre camera (the door line) plus two side cameras, each owning one
 * side of the paired chairs, and one unpaired chair (r-L5). */
export const SHOP: ZonesByCam = {
  'barber-c': [
    z({ id: 'c-door', type: 'entrance_line', points: [[0.4, 0.05], [0.6, 0.05]], inside: [0.5, 0.2] }),
    z({ id: 'c-L2', type: 'chair', pair: 'L2', points: sq(0.2, 0.6) }),
    z({ id: 'c-L1', type: 'chair', pair: 'L1', points: sq(0.2, 0.3) }),
    z({ id: 'c-R1', type: 'chair', pair: 'R1', points: sq(0.7, 0.3) }),
    z({ id: 'c-wash', type: 'washing', points: sq(0.45, 0.9) }),
    z({ id: 'c-sofa', type: 'waiting', points: sq(0.45, 0.5) }),
  ],
  'barber-r': [
    z({ id: 'r-L1', type: 'chair', pair: 'L1', owner: true, name: 'L1' }),
    z({ id: 'r-L2', type: 'chair', pair: 'L2', owner: true, name: 'L2' }),
    z({ id: 'r-L5', type: 'chair', name: 'L5' }),
    z({ id: 'r-s1', type: 'seat', seat_of: 'r-L1' }), z({ id: 'r-s2', type: 'seat', seat_of: 'r-L2' }),
    z({ id: 'r-s5', type: 'seat', seat_of: 'r-L5' }),
  ],
  'barber-l': [
    z({ id: 'l-R1', type: 'chair', pair: 'R1', owner: true, name: 'R1' }),
    z({ id: 'l-s1', type: 'seat', seat_of: 'l-R1' }),
  ],
}
