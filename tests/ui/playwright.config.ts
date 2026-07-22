import { defineConfig } from '@playwright/test'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const PORT = process.env.DEEPDATA_PORT || '18080'
const BASE_URL = process.env.BASE_URL || `http://localhost:${PORT}`
const CONFIG_DIR = path.dirname(fileURLToPath(import.meta.url))
const DEEPDATA_BIN = path.resolve(CONFIG_DIR, process.env.DEEPDATA_BIN || '../../deepdata-server')
const DATA_ROOT = path.resolve(
  CONFIG_DIR,
  process.env.DEEPDATA_DATA_DIR || '.deepdata-test-data',
)

if (!/^\d+$/.test(PORT) || Number(PORT) < 1 || Number(PORT) > 65535) {
  throw new Error(`invalid DEEPDATA_PORT: ${PORT}`)
}

export default defineConfig({
  testDir: './specs',
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: process.env.CI ? 'github' : 'html',
  use: {
    baseURL: BASE_URL,
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { browserName: 'chromium' },
    },
  ],
  webServer: {
    command: `${JSON.stringify(DEEPDATA_BIN)} serve --port ${PORT}`,
    url: `${BASE_URL}/readyz`,
    reuseExistingServer: !process.env.CI,
    timeout: 30000,
    env: {
      VECTORDB_MODE: 'local',
      EMBEDDER_TYPE: 'hash',
      VECTORDB_BASE_DIR: DATA_ROOT,
      VECTORDB_DATA_DIR: path.join(DATA_ROOT, 'local'),
      HYDRATION_COUNT: '0',
      DISABLE_WARMUP: '1',
      GRPC_PORT: '0',
    },
  },
})
