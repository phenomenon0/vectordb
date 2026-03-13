import { test as base, expect } from '@playwright/test'

const PORT = process.env.DEEPDATA_PORT || '18080'
const API = `http://localhost:${PORT}`

// Helper to call the DeepData API directly (bypasses UI)
async function api(method: string, path: string, body?: any): Promise<any> {
  const opts: RequestInit = {
    method,
    headers: { 'Content-Type': 'application/json' },
  }
  if (body) opts.body = JSON.stringify(body)
  const r = await fetch(`${API}${path}`, opts)
  const text = await r.text()
  try {
    return { status: r.status, ok: r.ok, data: JSON.parse(text) }
  } catch {
    return { status: r.status, ok: r.ok, data: text }
  }
}

// Fixture that seeds test data into a running DeepData instance
export const test = base.extend<{ seededServer: void; apiHelper: typeof api }>({
  apiHelper: async ({}, use) => {
    await use(api)
  },

  seededServer: [async ({}, use) => {
    // Wait for server to be healthy
    let healthy = false
    for (let i = 0; i < 30; i++) {
      try {
        const r = await fetch(`${API}/health`)
        if (r.ok) { healthy = true; break }
      } catch {}
      await new Promise(r => setTimeout(r, 500))
    }
    if (!healthy) throw new Error('Server not healthy after 15s')

    // Create test collection (ignore error if already exists)
    await api('POST', '/v2/collections', {
      name: 'test-collection',
      fields: [
        { name: 'text', type: 'text' },
        { name: 'vec', type: 'vector', dimension: 128 },
      ],
    })

    // Insert sample documents with known content for assertion
    const docs = [
      { id: 'test-doc-0', text: 'Machine learning is a subset of artificial intelligence', metadata: { topic: 'ml', source: 'test', priority: 1 } },
      { id: 'test-doc-1', text: 'Neural networks process data through layers of nodes', metadata: { topic: 'ml', source: 'test', priority: 2 } },
      { id: 'test-doc-2', text: 'Vector databases store high-dimensional embeddings', metadata: { topic: 'db', source: 'test', priority: 3 } },
      { id: 'test-doc-3', text: 'Semantic search finds documents by meaning', metadata: { topic: 'search', source: 'test', priority: 4 } },
      { id: 'test-doc-4', text: 'HNSW is an approximate nearest neighbor algorithm', metadata: { topic: 'algo', source: 'test', priority: 5 } },
    ]

    for (const doc of docs) {
      await api('POST', '/insert', {
        id: doc.id,
        text: doc.text,
        metadata: doc.metadata,
        collection: 'test-collection',
      })
    }

    await use()
  }, { auto: true }],
})

export { expect, api, API, PORT }
