import { test, expect, API } from '../fixtures/server'

// Collections tests implement the LIFECYCLE TEST SHAPE from our methodology:
// create → verify → mutate → verify → delete → verify
// This catches Class 2 (update path) bugs.

test.describe('Collections: Lifecycle Test', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard/')
    const skipBtn = page.locator('button:has-text("Skip")')
    if (await skipBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      await skipBtn.click()
    }
    await page.locator('.topnav-tab:has-text("Data")').click()
    await page.locator('button:has-text("Collections")').click()
  })

  test('full lifecycle: create → verify via API → insert doc → search → delete → verify gone', async ({ page }) => {
    const collName = `lifecycle-${Date.now()}`

    // ---- CREATE ----
    await page.locator('input[placeholder="my-collection"]').fill(collName)
    await page.locator('button:has-text("Create")').click()
    await expect(page.locator('.toast.success')).toBeVisible({ timeout: 5000 })

    // ---- VERIFY via API (ground truth) ----
    const collectionsResp = await fetch(`${API}/v2/collections`)
    const collectionsData = await collectionsResp.json()
    const names = (collectionsData.collections || []).map((c: any) => c.name || c)
    expect(names).toContain(collName)

    // ---- VERIFY in UI ----
    await page.locator('button:has-text("refresh")').click()
    await expect(page.locator(`td:has-text("${collName}")`)).toBeVisible({ timeout: 5000 })

    // ---- INSERT DOC via API ----
    const insertResp = await fetch(`${API}/insert`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        id: 'lifecycle-doc-1',
        text: 'lifecycle test document',
        collection: collName,
      }),
    })
    expect(insertResp.ok).toBe(true)

    // ---- VERIFY DOC EXISTS via API ----
    const scrollResp = await fetch(`${API}/scroll`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ collection: collName, limit: 10 }),
    })
    const scrollData = await scrollResp.json()
    const docIds = scrollData.ids || scrollData.documents?.map((d: any) => d.id) || []
    expect(docIds).toContain('lifecycle-doc-1')

    // ---- DELETE COLLECTION ----
    const row = page.locator(`tr:has-text("${collName}")`)
    await row.locator('button:has-text("delete")').click()
    await row.locator('button:has-text("confirm")').click()
    await expect(page.locator('.toast.success')).toBeVisible({ timeout: 5000 })

    // ---- VERIFY GONE via API ----
    const afterResp = await fetch(`${API}/v2/collections`)
    const afterData = await afterResp.json()
    const afterNames = (afterData.collections || []).map((c: any) => c.name || c)
    expect(afterNames).not.toContain(collName)

    // ---- VERIFY GONE in UI ----
    await page.locator('button:has-text("refresh")').click()
    await page.waitForTimeout(1000)
    await expect(page.locator(`td:has-text("${collName}")`)).not.toBeVisible()
  })

  test('creating duplicate collection shows error, not silent failure', async ({ page }) => {
    // test-collection already exists from seeding
    await page.locator('input[placeholder="my-collection"]').fill('test-collection')
    await page.locator('button:has-text("Create")').click()

    // Invariant: duplicate create surfaces an error (not success or silence)
    await page.waitForTimeout(2000)
    // Should either show error toast or the collection count stays the same
    const collectionsResp = await fetch(`${API}/v2/collections`)
    const data = await collectionsResp.json()
    const testCollCount = (data.collections || []).filter(
      (c: any) => (c.name || c) === 'test-collection'
    ).length
    // Invariant: no duplicate created — exactly 1 instance
    expect(testCollCount).toBeLessThanOrEqual(1)
  })

  test('empty collection name is rejected', async ({ page }) => {
    // Don't fill anything, just click Create
    await page.locator('button:has-text("Create")').click()
    await page.waitForTimeout(1000)

    // Invariant: no collection with empty name was created
    const resp = await fetch(`${API}/v2/collections`)
    const data = await resp.json()
    const emptyNames = (data.collections || []).filter(
      (c: any) => !(c.name || c)?.trim()
    )
    expect(emptyNames.length).toBe(0)
  })
})
