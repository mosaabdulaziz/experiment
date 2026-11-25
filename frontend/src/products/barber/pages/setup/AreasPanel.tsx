/** Where each shop area is drawn, which camera owns it and whether the
 * owner has a seat -- from every camera's draft, unsaved edits included --
 * and what doesn't add up yet. */
import { useMemo } from 'react'
import type { ZonesByCam } from '../../api/types'
import { buildAreas, zoneProblems } from '../../floor/areas'
import { useLang } from '../../i18n'
import { cameraLabel } from '../../ui/cameraLabel'
import { Num } from '../../ui/Num'
import { problemText } from './problemText'

export function AreasPanel({ draft, cameras }: { draft: ZonesByCam; cameras: string[] }) {
  const { t } = useLang()
  const areas = useMemo(() => buildAreas(draft, cameras), [draft, cameras])
  const problems = useMemo(() => zoneProblems(draft, cameras), [draft, cameras])
  const paired = useMemo(() => new Set(cameras.flatMap(c => (draft[c] ?? []).map(z => z.pair).filter(Boolean))), [draft, cameras])
  return (
    <section className="card areasPanel">
      <h2 className="sectionLabel">{t('ap.title')}</h2>
      <table className="areasTable">
        <thead>
          <tr>
            <th scope="col"><span className="srOnly">{t('ze.area')}</span></th>
            {cameras.map(c => <th key={c} scope="col">{cameraLabel(c, t)}</th>)}
            <th scope="col">{t('ap.seat')}</th>
          </tr>
        </thead>
        <tbody>
          {areas.areas.map(a => (
            <tr key={a.id}>
              <th scope="row"><span className={`toolSwatch tool-${a.type}`} aria-hidden="true" /><Num>{a.id}</Num>{a.name !== a.id && <span className="areaName" dir="auto">{a.name}</span>}</th>
              {cameras.map(c => {
                const drawn = a.drawnIn.includes(c)
                return (
                  <td key={c} className={drawn ? 'areaYes' : 'areaNo'}>
                    {drawn ? '✓' : '✗'}
                    {drawn && a.owner === c && paired.has(a.id) && <span className="ownerTag">{t('ap.owner')}</span>}
                  </td>
                )
              })}
              <td className={a.type !== 'chair' ? '' : a.hasSeat ? 'areaYes' : 'areaNo'}>
                {a.type === 'chair' ? (a.hasSeat ? '✓' : '✗') : ''}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {problems.length === 0
        ? <p className="areasOk">{t('ap.ok')}</p>
        : (
          <ul className="problemList">
            {problems.map((p, i) => <li key={i}>{problemText(p, t, areas)}</li>)}
          </ul>
        )}
    </section>
  )
}
