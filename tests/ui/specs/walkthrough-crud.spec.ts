import { test, expect, API } from '../fixtures/server'

// LIFECYCLE TEST SHAPE (Class 2 prevention):
// create → populate → verify → mutate → verify → delete → verify gone
//
// Every step cross-references the API (ground truth) against the UI.
// This catches update-path bugs where the UI shows stale state.

test.describe('Walkthrough: Full CRUD Lifecycle', () => {
  const collName = `crud-lifecycle-${Date.now()}`

  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard/')
    const skipBtn = page.locator('button:has-text("Skip")')
    if (await skipBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      await skipBtn.click()
    }
  })

  test('create → insert → search → delete → verify purged', async ({ page }) => {
    // ---- 1. CREATE COLLECTION via UI ----
    await page.locator('.topnav-tab:has-text("Data")').click()
    await page.locator('button:has-text("Collections")').click()
    await page.locator('input[placeholder="my-collection"]').fill(collName)
    await page.locator('button:has-text("Create")').click()
    await expect(page.locator('.toast.success')).toBeVisible({ timeout: 5000 })

    // Verify via API
    const colResp = await fetch(`${API}/v2/collections`)
    const colData = await colResp.json()
    expect((colData.collections || []).some((c: any) => (c.name || c) === collName)).toBe(true)

    // ---- 2. INSERT DOCUMENTS via API ----
    const docIds = ['crud-1', 'crud-2', 'crud-3']
    for (const id of docIds) {
      const r = await fetch(`${API}/insert`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          id,
          text: `Document ${id} for CRUD lifecycle test`,
          metadata: { test: true, idx: docIds.indexOf(id) },
          collection: collName,
        }),
      })
      expect(r.ok).toBe(true)
    }

    // ---- 3. VERIFY via EXPLORER ----
    await page.locator('button:has-text("Explorer")').click()
    await page.locator('input[placeholder="collection"]').fill(collName)
    await page.locator('button:has-text("load")').click()
    await expect(page.locator('td:has-text("crud-1")').first()).toBeVisible({ timeout: 10000 })

    // Invariant: all 3 docs visible
    for (const id of docIds) {
      await expect(page.locator(`td:has-text("${id}")`)).toBeVisible()
    }

    // ---- 4. SEARCH in the collection ----
    await page.locator('button:has-text("Search")').first().click()
    const collSelect = page.locator('select').nth(1)
    await collSelect.selectOption(collName)
    await page.locator('input[placeholder="Enter search query..."]').fill('CRUD lifecycle')
    await page.locator('button:has-text("Search")').last().click()
    await page.waitForTimeout(3000)
    // Hash embedder returns all docs — just verify no error
    const errorToast = await page.locator('.toast.error, .toast:has-text("failed")').count()
    expect(errorToast).toBe(0)

    // ---- 5. DELETE ONE DOC via API ----
    await fetch(`${API}/delete`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: 'crud-2', collection: collName }),
    })

    // Verify via API: only 2 docs remain
    const scrollResp = await fetch(`${API}/scroll`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ collection: collName, limit: 50 }),
    })
    const scrollData = await scrollResp.json()
    const remainingIds = scrollData.ids || []
    expect(remainingIds).not.toContain('crud-2')
    expect(remainingIds).toContain('crud-1')
    expect(remainingIds).toContain('crud-3')

    // ---- 6. DELETE COLLECTION ----
    await page.locator('button:has-text("Collections")').click()
    await page.locator('button:has-text("refresh")').click()
    await expect(page.locator(`td:has-text("${collName}")`)).toBeVisible({ timeout: 5000 })
    const row = page.locator(`tr:has-text("${collName}")`)
    await row.locator('button:has-text("delete")').click()
    await row.locator('button:has-text("confirm")').click()
    await expect(page.locator('.toast.success')).toBeVisible({ timeout: 5000 })

    // ---- 7. VERIFY PURGED via API ----
    const afterResp = await fetch(`${API}/v2/collections`)
    const afterData = await afterResp.json()
    expect((afterData.collections || []).some((c: any) => (c.name || c) === collName)).toBe(false)

    // Scrolling the deleted collection should fail
    const deadScroll = await fetch(`${API}/scroll`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ collection: collName, limit: 50 }),
    })
    // Either 404 or empty results — but the docs should be gone
    if (deadScroll.ok) {
      const data = await deadScroll.json()
      const ids = data.ids || []
      expect(ids.length).toBe(0)
    }
  })
})
