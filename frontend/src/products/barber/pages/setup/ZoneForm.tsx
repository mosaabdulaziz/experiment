/** The selected zone's form: name, and what the zone is for -- the shop area
 * a chair / washing / waiting zone stands for and whether this camera owns it, the
 * chair a seat belongs to, the side of a door line that is inside. */
import { useEffect, useMemo, useState, type FormEvent } from 'react'
import type { Zone, ZonesByCam } from '../../api/types'
import { buildAreas } from '../../floor/areas'
import { useLang } from '../../i18n'
import { NumIn } from '../../ui/Num'
import { flipInside, pairChoices, rename, setKind, setOwner, setPair, setFaceChair, setSeatOf } from './zoneModel'

const NEW = '\u0000new'

export function ZoneForm({ cam, draft, zone, onChange, onDelete }: {
  cam: string
  draft: ZonesByCam
  zone: Zone
  onChange: (next: ZonesByCam) => void
  onDelete: () => void
}) {
  const { t } = useLang()
  const zones = draft[cam] ?? []
  const [adding, setAdding] = useState(false)
  const [newId, setNewId] = useState('')
  const [invalid, setInvalid] = useState(false)
  useEffect(() => { setAdding(false); setNewId(''); setInvalid(false) }, [zone.id])

  const put = (list: Zone[]) => onChange({ ...draft, [cam]: list })
  const pairable = zone.type === 'chair' || zone.type === 'washing' || zone.type === 'waiting' ? zone.type : null
  const choices = pairable ? pairChoices(draft, pairable) : []
  if (zone.pair && !choices.includes(zone.pair)) choices.push(zone.pair)
  const chairs = zones.filter(z => z.type === 'chair')
  const allChairs = useMemo(() => buildAreas(draft).areas.filter(a => a.type === 'chair'), [draft])

  const onArea = (value: string) => {
    if (value === NEW) { setAdding(true); return }
    setAdding(false)
    onChange(setPair(draft, cam, zone.id, value || null))
  }
  const addArea = (e: FormEvent) => {
    e.preventDefault()
    const id = newId.trim()
    if (!id || /\s/.test(id)) { setInvalid(true); return }
    setAdding(false)
    setNewId('')
    setInvalid(false)
    onChange(setPair(draft, cam, zone.id, id))
  }

  return (
    <div className="zoneForm">
      <h3 className="zoneFormTitle">
        <NumIn text={t('ze.selected', { type: t(`zt.${zone.type}`), n: '{n}' })} n={zone.name || zone.id} />
      </h3>
      <label className="field">
        {t('ze.name')}
        <input value={zone.name} placeholder={zone.id} dir="auto" onChange={e => put(rename(zones, zone.id, e.target.value))} />
      </label>

      {pairable && (
        <>
          <label className="field">
            {t('ze.area')}
            <select value={adding ? NEW : zone.pair ?? ''} onChange={e => onArea(e.target.value)}>
              <option value="">{t('ze.noArea')}</option>
              {choices.map(c => <option key={c} value={c}>{c}</option>)}
              <option value={NEW}>{t('ze.newArea')}</option>
            </select>
          </label>
          {adding && (
            <form className="newAreaRow" onSubmit={addArea}>
              <input className="input" value={newId} autoFocus dir="ltr" aria-label={t('ze.newAreaId')} placeholder={t('ze.newAreaId')}
                     aria-invalid={invalid} onChange={e => { setNewId(e.target.value); setInvalid(false) }} />
              <button type="submit" className="btnPlain btnSmall">{t('ze.add')}</button>
            </form>
          )}
          {invalid && <p className="fieldError" role="alert">{t('ze.newAreaInvalid')}</p>}
          <label className="checkRow">
            <input type="checkbox" checked={!!zone.owner} disabled={!zone.pair}
                   onChange={e => onChange(setOwner(draft, cam, zone.id, e.target.checked))} />
            {t('ze.owner')}
          </label>
        </>
      )}

      {zone.type === 'chair' && (
        <label className="checkRow">
          <input type="checkbox" checked={zone.kind === 'pedicure'}
                 onChange={e => put(setKind(zones, zone.id, e.target.checked ? 'pedicure' : null))} />
          {t('ze.pedicure')}
        </label>
      )}

      {zone.type === 'seat' && (
        <label className="field">
          {t('ze.seatOf')}
          <select value={zone.seat_of ?? ''} onChange={e => put(setSeatOf(zones, zone.id, e.target.value))}>
            {!zone.seat_of && <option value="">{t('ze.noChair')}</option>}
            {chairs.map(c => <option key={c.id} value={c.id}>{c.name || c.id}</option>)}
          </select>
        </label>
      )}

      {zone.type === 'face_area' && (
        <label className="field">
          {t('ze.faceChair')}
          <select value={zone.chair ?? ''} onChange={e => put(setFaceChair(zones, zone.id, e.target.value))}>
            {!zone.chair && <option value="">{t('ze.noChair')}</option>}
            {allChairs.map(c => <option key={c.id} value={c.id}>{c.name || c.id}</option>)}
          </select>
        </label>
      )}

      {zone.type === 'entrance_line' && (
        <div>
          <button type="button" className="btnPlain btnSmall" disabled={!zone.inside}
                  onClick={() => put(zones.map(z => (z.id === zone.id ? flipInside(z) : z)))}>{t('ze.flip')}</button>
        </div>
      )}

      <div>
        <button type="button" className="btnDanger btnSmall" onClick={onDelete}>{t('ze.delete')}</button>
      </div>
    </div>
  )
}
