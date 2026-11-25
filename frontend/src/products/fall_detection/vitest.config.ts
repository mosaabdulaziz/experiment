// This product's unit tests (pure view-model logic, no DOM): npm test -w @iris/fall-detection
import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: { include: ['**/*.test.ts'], environment: 'node' },
})
