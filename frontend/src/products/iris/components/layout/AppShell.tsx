import type { ReactNode } from 'react'
import { Header } from './Header'
import styles from './AppShell.module.css'

interface AppShellProps {
  children: ReactNode
}

export function AppShell({ children }: AppShellProps) {
  return (
    <div className={styles.shell}>
      <Header />
      {children}
    </div>
  )
}
