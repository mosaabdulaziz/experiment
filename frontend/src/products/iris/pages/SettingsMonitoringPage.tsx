import { SystemPromptPanel } from '@/components/prompt/SystemPromptPanel'
import styles from './SettingsMonitoringPage.module.css'

export function SettingsMonitoringPage() {
  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <h1 className={styles.title}>AI Agent</h1>
        <p className={styles.subtitle}>
          The persona, language, and instructions the AI monitoring agent follows while watching the feeds.
        </p>
      </header>
      <div className={styles.panelWrap}>
        <SystemPromptPanel />
      </div>
    </div>
  )
}
