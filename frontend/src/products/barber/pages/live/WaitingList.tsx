import type { Staff } from '../../api/types'
import type { AreaTable } from '../../floor/areas'
import { useLang } from '../../i18n'
import { buildBoard, ticketLabel } from '../../live/selectors'
import type { LooseSnapshot } from '../../live/sidePanel'
import { Num } from '../../ui/Num'
import { Timer } from '../../ui/Timer'

/** Who is waiting, longest first: ticket and timer (red past the wait target). */
export function WaitingList({ snap, now, areas, staff }: { snap: LooseSnapshot; now: number; areas: AreaTable; staff: Staff[] }) {
  const { t } = useLang()
  const waiting = buildBoard(snap, areas, staff).waiting
  return (
    <section className="card sideBlock sideScroll" aria-labelledby="waitingTitle">
      <h2 id="waitingTitle" className="sectionLabel">{t('live.waiting')} · <Num>{waiting.length}</Num></h2>
      {waiting.length === 0
        ? <p className="waitNone">{t('live.noneWaiting')}</p>
        : (
          <ol className="waitList">
            {waiting.map(w => (
              <li key={w.visitId}>
                <Num className="ticket">{ticketLabel(w.ticket)}</Num>
                <Timer since={w.since} now={now} target={w.target} />
              </li>
            ))}
          </ol>
        )}
    </section>
  )
}
