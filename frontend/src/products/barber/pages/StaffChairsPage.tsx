/** Staff & chairs: the daily roster (who works which chair, confirmed once
 * per shop day) and the barber list. A change on a chair saves immediately;
 * moving a barber off another chair says so, briefly. */
import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent, type KeyboardEvent } from 'react'
import { api, ApiError } from '../api/client'
import type { RosterConfirmation, RosterDetail, Staff, ZonesByCam } from '../api/types'
import type { PageProps } from '../App'
import { buildAreas } from '../floor/areas'
import { useLang } from '../i18n'
import { rosterMove } from '../live/selectors'
import { fmtClock, isoDate } from '../ui/format'
import { Num } from '../ui/Num'
import { chairOptions, createLatestGuard } from './staffChairs'
import { Switch } from '../ui/ui'
import { productHref } from '../../../shared/router'

const MOVED_MS = 4000

export function StaffChairsPage({ live, staff, cameras, reloadStaff }: Pick<PageProps, 'live' | 'staff' | 'cameras' | 'reloadStaff'>) {
  const { t, fmt } = useLang()
  const shopDay = live.snap?.shop_day ?? isoDate(new Date())
  const [zones, setZones] = useState<ZonesByCam | null>(null)
  const [roster, setRoster] = useState<RosterDetail | null>(null)
  const [confirmation, setConfirmation] = useState<RosterConfirmation | null>(null)
  const [loadFailed, setLoadFailed] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [moved, setMoved] = useState<string | null>(null)
  const movedTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editValue, setEditValue] = useState('')
  const [newName, setNewName] = useState('')
  const [pendingChairs, setPendingChairs] = useState<Set<string>>(new Set())
  // Several chairs can be reassigned in quick succession; each fires its own
  // assign-then-reload GET, and those GETs can settle out of order. This
  // guard keeps only the most recently *started* reload's result, so a
  // slower, superseded one never overwrites the roster with stale data.
  const rosterGuard = useRef(createLatestGuard()).current

  const camIds = useMemo(() => cameras.map(c => c.id), [cameras])
  const areas = useMemo(() => buildAreas(zones ?? live.snap?.zones ?? {}, camIds), [zones, live.snap?.zones, camIds])
  const chairs = useMemo(() => areas.groups.flatMap(g => g.ids).map(id => areas.byId[id]), [areas])

  const loadRoster = useCallback(() => {
    const token = rosterGuard.start()
    return api.roster().then(r => { if (rosterGuard.isCurrent(token)) setRoster(r) })
  }, [rosterGuard])

  useEffect(() => {
    let off = false
    Promise.all([api.zones().then(setZones), loadRoster(), api.rosterConfirmation().then(setConfirmation)])
      .catch(() => { if (!off) setLoadFailed(true) })
    return () => { off = true }
  }, [loadRoster])

  // The shop day rolls over (or another screen confirms): refetch, so the
  // banner never shows yesterday's confirmation.
  const snapDay = live.snap?.shop_day
  const snapConfirmed = live.snap?.roster_confirmed
  useEffect(() => {
    if (snapDay === undefined) return
    let off = false
    api.rosterConfirmation().then(c => { if (!off) setConfirmation(c) }).catch(() => {})
    return () => { off = true }
  }, [snapDay, snapConfirmed])

  useEffect(() => () => { if (movedTimer.current) clearTimeout(movedTimer.current) }, [])

  const showMoved = useCallback((msg: string) => {
    setMoved(msg)
    if (movedTimer.current) clearTimeout(movedTimer.current)
    movedTimer.current = setTimeout(() => setMoved(null), MOVED_MS)
  }, [])

  const fail = (e: unknown) => setError(e instanceof ApiError ? e.detail : String(e))

  const onAssign = useCallback((chairId: string, staffId: string | null) => {
    const before: Record<string, string> = {}
    for (const [c, d] of Object.entries(roster ?? {})) before[c] = d.staff_id
    const movedFrom = rosterMove(before, chairId, staffId)
    setError(null)
    setPendingChairs(prev => new Set(prev).add(chairId))
    api.assignChair(chairId, staffId)
      .then(loadRoster)
      .then(() => {
        if (!movedFrom) return
        const name = staffId ? staff.find(s => s.id === staffId)?.name ?? staffId : ''
        const chairName = areas.byId[movedFrom]?.name ?? movedFrom
        showMoved(t('staff.moved', { name, chair: chairName }))
      })
      .catch(fail)
      .finally(() => setPendingChairs(prev => {
        if (!prev.has(chairId)) return prev
        const next = new Set(prev)
        next.delete(chairId)
        return next
      }))
  }, [roster, staff, areas, t, loadRoster, showMoved])

  const onConfirm = useCallback(() => {
    setError(null)
    api.confirmRoster().then(setConfirmation).catch(fail)
  }, [])

  const onToggleActive = useCallback((s: Staff, active: boolean) => {
    setError(null)
    api.patchStaff(s.id, { active }).then(() => reloadStaff()).catch(fail)
  }, [reloadStaff])

  const startEdit = (s: Staff) => { setEditingId(s.id); setEditValue(s.name) }
  const cancelEdit = () => setEditingId(null)
  const saveEdit = (id: string) => {
    const name = editValue.trim()
    setEditingId(null)
    if (!name) return
    setError(null)
    api.patchStaff(id, { name }).then(() => reloadStaff()).catch(fail)
  }
  const onEditKey = (id: string) => (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') { e.preventDefault(); saveEdit(id) }
    else if (e.key === 'Escape') { e.preventDefault(); cancelEdit() }
  }

  const onAdd = (e: FormEvent) => {
    e.preventDefault()
    const name = newName.trim()
    if (!name) return
    setError(null)
    api.addStaff(name).then(() => { setNewName(''); reloadStaff() }).catch(fail)
  }

  return (
    <div className="staffPage">
      <header className="pageHead">
        <div className="pageHeadMain"><h1>{fmt.longDate(shopDay)}</h1></div>
      </header>

      {loadFailed && <p className="errorBar" role="alert">{t('common.loadFailed')}</p>}
      {error && <p className="errorBar" role="alert">{t('st.saveFailed', { detail: error })}</p>}

      {confirmation && (
        confirmation.confirmed ? (
          <div className="banner rosterBannerDone">
            <span>{t('st.confirmed', { time: fmtClock(confirmation.confirmed_at) })}</span>
          </div>
        ) : (
          <div className="banner rosterBanner">
            <span>{t('st.confirmBanner')}</span>
            <button type="button" className="btnPrimary btnSmall" onClick={onConfirm}>{t('st.confirm')}</button>
          </div>
        )
      )}

      <div className="twoCol">
        <section className="card">
          <h2 className="sectionLabel">{t('st.chairs')}</h2>
          {chairs.length === 0
            ? <p className="empty"><a href={productHref('/setup')}>{t('st.noChairs')}</a></p>
            : (
              <ul className="chairRoster">
                {chairs.map(area => {
                  const entry = roster?.[area.id]
                  const options = chairOptions(staff, entry?.staff_id ?? null)
                  return (
                    <li key={area.id} className="chairRosterRow">
                      <Num className="stationName">{area.name}</Num>
                      <select
                        className="select"
                        aria-label={area.name}
                        value={entry?.staff_id ?? ''}
                        disabled={pendingChairs.has(area.id)}
                        onChange={e => onAssign(area.id, e.target.value || null)}
                      >
                        <option value="">{t('live.noBarber')}</option>
                        {options.map(o => (
                          <option key={o.id} value={o.id} disabled={o.disabled}>
                            {o.disabled ? `${o.name} ${t('live.inactive')}` : o.name}
                          </option>
                        ))}
                      </select>
                      {entry && <span className="muted small">{t('st.since', { date: fmt.date(entry.since) })}</span>}
                    </li>
                  )
                })}
              </ul>
            )}
        </section>

        <section className="card">
          <h2 className="sectionLabel">{t('st.staff')}</h2>
          <ul className="barberList">
            {staff.map(s => (
              <li key={s.id} className={s.active ? 'barberItem' : 'barberItem barberItemInactive'}>
                <Switch checked={s.active} onChange={v => onToggleActive(s, v)} label={`${s.name} ${t('st.active')}`} />
                <div className="barberInfo">
                  {editingId === s.id ? (
                    <input
                      className="input"
                      value={editValue}
                      autoFocus
                      aria-label={t('st.name')}
                      onChange={e => setEditValue(e.target.value)}
                      onKeyDown={onEditKey(s.id)}
                    />
                  ) : (
                    <strong>{s.name}</strong>
                  )}
                </div>
                {editingId !== s.id && (
                  <button type="button" className="linkBtn" onClick={() => startEdit(s)}>{t('st.rename')}</button>
                )}
              </li>
            ))}
          </ul>
          <p className="muted small">{t('st.inactiveNote')}</p>
          <form className="staffAddForm" onSubmit={onAdd}>
            <input
              className="input"
              value={newName}
              onChange={e => setNewName(e.target.value)}
              placeholder={t('st.name')}
              aria-label={t('st.name')}
            />
            <button type="submit" className="btnPrimary btnSmall">{t('st.add')}</button>
          </form>
        </section>
      </div>

      <div className="toast" role="status" aria-live="polite">
        {moved && <span className="toastMsg">{moved}</span>}
      </div>
    </div>
  )
}
