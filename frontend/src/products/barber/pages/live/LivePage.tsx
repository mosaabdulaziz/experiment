import { useMemo } from 'react'
import type { PageProps } from '../../App'
import { buildAreas } from '../../floor/areas'
import { useLang } from '../../i18n'
import { CamerasView } from './CamerasView'
import { ChairStrip } from './ChairStrip'
import { SidePanel } from './SidePanel'

/** The front desk's screen: the chair chips above the cameras, and the side panel. */
export function LivePage({ live, staff, cameras, camerasState }: Pick<PageProps, 'live' | 'staff' | 'cameras' | 'camerasState'>) {
  const { t } = useLang()
  const { snap, conn, now } = live
  const zones = snap?.zones
  // Only the configured cameras count: zones.json can still hold a stale one.
  const camIds = useMemo(() => cameras.map(c => c.id), [cameras])
  const areas = useMemo(() => buildAreas(zones ?? {}, camIds), [zones, camIds])

  return (
    <div className="livePage">
      {/* 2026-10-18 (user): the "Live" strip only took space -- the rail marks the page, each tile says it is live */}
      <h1 className="srOnly">{t('nav.live')}</h1>
      {camerasState === 'failed'
        ? <p className="errorBar" role="alert">{t('common.loadFailed')}</p>
        : !snap || camerasState === 'loading'
        ? <p className="card muted">{t('common.loading')}</p>
        : (
          <>
            {conn !== 'connected' && <p className="pausedStrip" role="status">{t('conn.paused')}</p>}
            <div className="liveGrid">
              <ChairStrip snap={snap} now={now} areas={areas} staff={staff} chairs={live.chairs} />
              <div className="liveTabBody"><CamerasView live={live} cameras={cameras} areas={areas} /></div>
              <SidePanel snap={snap} now={now} staff={staff} areas={areas} scene={live.scene} cameras={cameras} />
            </div>
          </>
        )}
    </div>
  )
}
