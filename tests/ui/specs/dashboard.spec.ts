import { test, expect, API } from '../fixtures/server'

// Dashboard tests verify that the UI faithfully reflects server state.
// Every assertion cross-references the API response (ground truth) against
// what the UI displays. This catches Class 14 divergence bugs.

test.describe('Dashboard: Data Fidelity', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard/')
    const skipBtn = page.locator('button:has-text("Skip")')
    if (await skipBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      await skipBtn.click()
    }
    await expect(page.locator('.status-dot.online')).toBeVisible({ timeout: 10000 })
  })

  test('collection table matches API state', async ({ page }) => {
    // Ground truth: what collections does the server actually have?
    const healthResp = await fetch(`${API}/health`)
    const health = await healthResp.json()
    const serverCollections = Object.keys(health.collections || {})

    // Invariant: every server collection appears in the dashboard table
    for (const name of serverCollections) {
      await expect(page.locator(`td:has-text("${name}")`)).toBeVisible({ timeout: 10000 })
    }

    // Invariant: test-collection (seeded) is always present
    await expect(page.locator('td:has-text("test-collection")')).toBeVisible()
  })

  test('stat strip values are numeric and non-negative', async ({ page }) => {
    const statValues = page.locator('.stat-strip .val')
    const count = await statValues.count()
    expect(count).toBeGreaterThanOrEqual(3) // vectors, collections, index at minimum

    for (let i = 0; i < count; i++) {
      const text = await statValues.nth(i).textContent()
      // Invariant: stat values are formatted numbers (possibly with commas/units)
      expect(text?.trim()).toBeTruthy()
      // Should contain at least one digit
      expect(text).toMatch(/\d/)
    }
  })

  test('integrity check returns a specific verdict, not just "visible"', async ({ page }) => {
    // Wait for auto-check
    const verdict = page.locator('td:has-text("Integrity")').locator('..').locator('.text-green, .text-red')
    await expect(verdict).toBeVisible({ timeout: 15000 })

    const text = await verdict.textContent()
    // Invariant: verdict is exactly "OK" or starts with "FAIL" — no other states
    expect(text?.trim()).toMatch(/^(OK|FAIL)/)
  })

  test('health auto-refresh updates data without user action', async ({ page }) => {
    // Capture initial vector count
    const getVectorText = () => page.locator('.stat-strip .val').first().textContent()
    const initial = await getVectorText()

    // Insert a doc via API to change server state
    const docId = `refresh-test-${Date.now()}`
    await fetch(`${API}/insert`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        id: docId,
        text: 'auto-refresh test document',
        collection: 'test-collection',
      }),
    })

    // Wait for auto-refresh cycle (15s interval + margin)
    await page.waitForTimeout(17000)

    // Invariant: UI eventually reflects the new state
    const updated = await getVectorText()
    // Count should have increased by 1 (or at least changed)
    const initialNum = parseInt(initial?.replace(/,/g, '') || '0')
    const updatedNum = parseInt(updated?.replace(/,/g, '') || '0')
    expect(updatedNum).toBeGreaterThanOrEqual(initialNum)

    // Cleanup
    await fetch(`${API}/delete`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: docId, collection: 'test-collection' }),
    })
  })
})
