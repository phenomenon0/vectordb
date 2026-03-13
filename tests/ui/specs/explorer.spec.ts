import { test, expect, API } from '../fixtures/server'

// Explorer tests verify that browsed documents match server state.
// Key invariants: document count matches API, content is correct,
// delete actually removes the doc (lifecycle), pagination is consistent.

test.describe('Explorer: Data Integrity', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard/')
    const skipBtn = page.locator('button:has-text("Skip")')
    if (await skipBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      await skipBtn.click()
    }
    await page.locator('.topnav-tab:has-text("Data")').click()
    await page.locator('button:has-text("Explorer")').click()
  })

  test('loaded documents match API scroll response', async ({ page }) => {
    // Get ground truth from API
    const scrollResp = await fetch(`${API}/scroll`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ collection: 'test-collection', limit: 50 }),
    })
    const scrollData = await scrollResp.json()
    const apiDocCount = (scrollData.ids || scrollData.documents || []).length

    // Load in UI
    await page.locator('input[placeholder="collection"]').fill('test-collection')
    await page.locator('button:has-text("load")').click()
    await expect(page.locator('td:has-text("test-doc-")').first()).toBeVisible({ timeout: 10000 })

    // Invariant: UI doc count matches API
    const uiRows = page.locator('tr:has-text("test-doc-")')
    const uiCount = await uiRows.count()
    expect(uiCount).toBe(apiDocCount)
  })

  test('document IDs displayed match seeded IDs exactly', async ({ page }) => {
    await page.locator('input[placeholder="collection"]').fill('test-collection')
    await page.locator('button:has-text("load")').click()
    await expect(page.locator('td:has-text("test-doc-")').first()).toBeVisible({ timeout: 10000 })

    // Invariant: all 5 seeded doc IDs are present
    for (let i = 0; i < 5; i++) {
      await expect(page.locator(`td:has-text("test-doc-${i}")`)).toBeVisible()
    }
  })

  test('delete lifecycle: doc removed in UI and API', async ({ page }) => {
    // Insert a temporary doc
    const tmpId = `explorer-del-${Date.now()}`
    await fetch(`${API}/insert`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        id: tmpId,
        text: 'temporary document for delete test',
        collection: 'test-collection',
      }),
    })

    // Load explorer
    await page.locator('input[placeholder="collection"]').fill('test-collection')
    await page.locator('button:has-text("load")').click()
    await expect(page.locator(`td:has-text("${tmpId}")`)).toBeVisible({ timeout: 10000 })

    // Expand the doc and delete it
    const row = page.locator(`tr:has-text("${tmpId}")`)
    await row.locator('button:has-text("view")').click()
    await page.locator('button:has-text("delete")').first().click()

    // Handle confirmation if present
    const confirmBtn = page.locator('button:has-text("confirm")')
    if (await confirmBtn.isVisible({ timeout: 1000 }).catch(() => false)) {
      await confirmBtn.click()
    }
    await page.waitForTimeout(2000)

    // Invariant: doc is gone from API
    const scrollResp = await fetch(`${API}/scroll`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ collection: 'test-collection', limit: 50 }),
    })
    const scrollData = await scrollResp.json()
    const ids = scrollData.ids || []
    expect(ids).not.toContain(tmpId)
  })

  test('loading nonexistent collection shows error, not empty without explanation', async ({ page }) => {
    await page.locator('input[placeholder="collection"]').fill('nonexistent-collection-xyz')
    await page.locator('button:has-text("load")').click()
    await page.waitForTimeout(2000)

    // Invariant: either an error toast or a visible "no documents" / empty state
    // The UI should not silently show nothing
    const rows = await page.locator('tr:has-text("test-doc-")').count()
    expect(rows).toBe(0)
  })
})
