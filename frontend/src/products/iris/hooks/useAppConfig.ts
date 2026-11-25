import { useEffect } from 'react'
import { configApi } from '@/api/config'
import { useConfigStore } from '@/stores/configStore'

/**
 * One-shot read of the backend config for values the UI needs globally. Failure is
 * silent by design — the store already holds the backend's own defaults, so a
 * missing config response degrades to correct-for-a-default-deployment rather than
 * to a broken page.
 */
export function useAppConfig() {
  const setClipTtlSeconds = useConfigStore(s => s.setClipTtlSeconds)
  const setVlmEnabled = useConfigStore(s => s.setVlmEnabled)
  const setPerceptionBackend = useConfigStore(s => s.setPerceptionBackend)

  useEffect(() => {
    let cancelled = false
    configApi
      .getAppConfig()
      .then(cfg => {
        if (cancelled) return
        const ttl = cfg.alerts?.clip?.ttl_seconds
        if (typeof ttl === 'number' && ttl > 0) setClipTtlSeconds(ttl)
        setVlmEnabled(cfg.monitoring?.vlm_enabled === true)
        setPerceptionBackend(cfg.perception?.backend ?? 'local') // the backend's own default
      })
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [setClipTtlSeconds, setVlmEnabled, setPerceptionBackend])
}
