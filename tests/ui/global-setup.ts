import { execSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const CONFIG_DIR = path.dirname(fileURLToPath(import.meta.url))
const ROOT = path.resolve(CONFIG_DIR, '../..')
const BINARY = path.resolve(CONFIG_DIR, process.env.DEEPDATA_BIN || '../../deepdata-server')

export default async function globalSetup() {
  // Build Go binary if not present or stale
  if (!existsSync(BINARY)) {
    console.log('Building DeepData server binary...')
    execSync(`go build -o ${JSON.stringify(BINARY)} ./cmd/deepdata/`, {
      cwd: ROOT,
      stdio: 'inherit',
      timeout: 120000,
    })
  }

  // Seed test data after server starts (handled in fixtures)
}
