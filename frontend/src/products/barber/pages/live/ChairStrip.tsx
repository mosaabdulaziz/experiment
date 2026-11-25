import type { ChairOcc, ResumeDecision, Staff } from '../../api/types'
import type { AreaTable } from '../../floor/areas'
import { useLang } from '../../i18n'
import { chairStrip, type ChairChip, type ChipGroup } from '../../live/chairStrip'
import { buildBoard, ticketLabel } from '../../live/selectors'
import type { LooseSnapshot } from '../../live/sidePanel'
import { useStore, type Store } from '../../live/store'
import { elapsedSince, fmtClock, fmtTimer } from '../../ui/format'
import { Num, NumIn } from '../../ui/Num'
import { Timer } from '../../ui/Timer'

/** How the re-ID check decided this sitting (2026-10-18, user: say it, with the score, instead of leaving it to the
 * timer going on or restarting): resumed (the previous customer back) / same customer (after the barber's cover) /
 * new customer, the similarity, and in the tooltip what it had to reach. */
function Decision({ d }: { d: ResumeDecision }) {
  const { t } = useLang()
  const sim = d.sim == null ? null : d.sim.toFixed(2)
  const gap = fmtTimer(d.gap_s)
  const hint = sim == null ? t('strip.decHintTime', { gap })
    : t(d.kind === 'reseat' ? 'strip.decHintReseat' : 'strip.decHint', { gap, sim, need: d.threshold?.toFixed(2) ?? '–' })
  const what = !d.resumed ? t('strip.decNew')
    : d.kind === 'reseat' ? t('strip.decSame')
    : d.ticket != null ? <NumIn text={t('strip.decResumedTicket', { n: '{n}' })} n={ticketLabel(d.ticket)} />
    : t('strip.decResumed')
  return (
    <div className={`stripDecision ${d.resumed ? 'decSame' : 'decNew'}`} title={hint}>
      <span className="decIcon" aria-hidden="true">{d.resumed ? '↺' : '+'}</span>
      <span className="decWhat">{what}</span>
      <span className="decSim">{sim != null ? <NumIn text={t('strip.decSim', { n: '{n}' })} n={sim} /> : t('strip.decByTime')}</span>
    </div>
  )
}

function Chip({ c, now, target }: { c: ChairChip; now: number; target: number | null }) {
  const { t } = useLang()
  return (
    <li className={`stripChip chip-${c.kind}`}>
      <div className="stripChipHead">
        <Num className="stripChair">{c.name}</Num>
        {c.pedicure ? null : c.barber
          ? <span className="stripBarber">{c.barber.name}{!c.barber.active && ` ${t('live.inactive')}`}</span>
          : <span className="stripBarber noBarber">{t('live.noBarber')}</span>}
      </div>
      {c.startedAt && (c.kind === 'busy' || c.kind === 'washing' || c.kind === 'away') && (
        <div className="stripStarted"><NumIn text={t('strip.startedAt', { t: '{n}' })} n={fmtClock(c.startedAt)} /></div>
      )}
      {c.decision && <Decision d={c.decision} />}
      {c.kind === 'free'
        ? <div className="stripFree">{t('live.free')}</div>
        : c.kind === 'no_view'
        ? <div className="stripNoView">{t('strip.noView')}</div>
        : c.kind === 'starting'
        ? <div className="stripStarting">{t('strip.starting')}</div>
        : c.kind === 'checking'
        ? (
          <div className="stripChecking" title={t('strip.checkingHint')}>
            <span className="checkDot" aria-hidden="true" />
            {c.ticket != null
              ? <NumIn text={t('strip.checkingTicket', { n: '{n}' })} n={ticketLabel(c.ticket)} />
              : t('strip.checking')}
          </div>
        )
        : c.kind === 'away'
        ? (
          <div className="stripBusy">
            <span className="washTag">{t('strip.away')}</span>
            {c.ticket != null && <Num className="ticket">{ticketLabel(c.ticket)}</Num>}
            {c.since && (
              <span className="timer stripLeft">
                <NumIn text={t('strip.awayFor', { n: '{n}' })} n={fmtTimer(elapsedSince(c.since, now))} />
              </span>
            )}
          </div>
        )
        : (
          <div className="stripBusy">
            {c.kind === 'washing' && <span className="washTag">{t('live.washing')}</span>}
            {c.ticket != null && <Num className="ticket">{ticketLabel(c.ticket)}</Num>}
            {c.since && <Timer since={c.since} now={now} target={c.kind === 'busy' && !c.pedicure ? target : null} />}
          </div>
        )}
    </li>
  )
}

const GROUP_TITLE = { left: 'live.leftRow', right: 'live.rightRow', vip: 'strip.vip', kids: 'strip.kids', other: 'strip.other', pedicure: 'strip.pedicure' } as const

function Group({ g, now, target }: { g: ChipGroup; now: number; target: number | null }) {
  const { t } = useLang()
  return (
    <div className={`stripGroup stripGroup-${g.side}`}>
      <h3 className="stripGroupTitle">{t(GROUP_TITLE[g.side])}</h3>
      <ul className="stripChips">
        {g.chips.map(c => <Chip key={c.area} c={c} now={now} target={target} />)}
      </ul>
    </div>
  )
}

const EMPTY: Record<string, ChairOcc> = {}
const NO_CHAIRS: Store<Record<string, ChairOcc>> = { get: () => EMPTY, subscribe: () => () => {} }

/** Every chair above the cameras, the Shop board's rows in the same physical
 * order (left row on the left in Arabic too): barber, and the chair's state
 * -- sitting down, checking (new customer or the previous one back), the haircut (ticket and a running timer), away
 * (time left to come back), no camera view, or free. Without chair states
 * (an older backend) it is busy / washing / free from the board. */
export function ChairStrip({ snap, now, areas, staff, chairs = NO_CHAIRS }: {
  snap: LooseSnapshot; now: number; areas: AreaTable; staff: Staff[]; chairs?: Store<Record<string, ChairOcc>>
}) {
  const { t } = useLang()
  const strip = chairStrip(buildBoard(snap, areas, staff), useStore(chairs), now)
  if (strip.total === 0) return null
  const target = snap.targets?.service_target_s ?? null
  return (
    <section className="card chairStrip" aria-labelledby="chairStripTitle">
      <header className="stripHead">
        <h2 id="chairStripTitle" className="sectionLabel">{t('strip.title')}</h2>
        <span className="stripCount stripCountBusy">{t('strip.busy')} <Num>{strip.busy}</Num></span>
        <span className="stripCount">{t('strip.free')} <Num>{strip.free}</Num></span>
        {strip.noView > 0 && <span className="stripCount stripCountNoView">{t('strip.noViewCount')} <Num>{strip.noView}</Num></span>}
      </header>
      <div className="stripGroups">
        {strip.groups.map(g => <Group key={g.side} g={g} now={now} target={target} />)}
      </div>
    </section>
  )
}
