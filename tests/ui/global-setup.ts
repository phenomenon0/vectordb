import { execSync } from 'child_process'
import { existsSync } from 'fs'
import path from 'path'

const ROOT = path.resolve(__dirname, '../..')
const BINARY = path.join(ROOT, 'deepdata-server')

export default async function globalSetup() {
  // Build Go binary if not present or stale
  if (!existsSync(BINARY)) {
    console.log('Building DeepData server binary...')
    execSync('go build -o deepdata-server ./cmd/deepdata/', {
      cwd: ROOT,
      stdio: 'inherit',
      timeout: 120000,
    })
  }

  // Seed test data after server starts (handled in fixtures)
}
