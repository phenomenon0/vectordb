import { defineConfig } from '@playwright/test'

const PORT = process.env.DEEPDATA_PORT || '18080'
const BASE_URL = process.env.BASE_URL || `http://localhost:${PORT}`

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
    command: `../../deepdata-server serve --port ${PORT}`,
    url: `${BASE_URL}/health`,
    reuseExistingServer: !process.env.CI,
    timeout: 30000,
    env: {
      VECTORDB_MODE: 'hash',
      EMBEDDER_TYPE: 'hash',
      DATA_DIR: '.deepdata-test-data',
    },
  },
})
