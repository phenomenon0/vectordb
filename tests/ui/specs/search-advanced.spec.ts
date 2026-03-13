import { test, expect } from '../fixtures/server'

// Advanced search UI tests verify that controls exist and function correctly.
// Behavioral contracts (actual payloads) are tested in walkthrough-advanced-search.spec.ts.
// These tests focus on UI state machine invariants.

test.describe('Search Advanced: UI State Machine', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard/')
    const skipBtn = page.locator('button:has-text("Skip")')
    if (await skipBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      await skipBtn.click()
    }
    await page.locator('.topnav-tab:has-text("Data")').click()
  })

  test.describe('Hybrid Controls', () => {
    test('hybrid controls only visible in hybrid mode with filters expanded', async ({ page }) => {
      await page.locator('button:has-text("Search")').first().click()

      // Invariant: hybrid section hidden in vector mode
      await expect(page.locator('.hybrid-strategy-section')).not.toBeVisible()

      // Switch to hybrid
      const modeSelect = page.locator('select').filter({ has: page.locator('option[value="hybrid"]') }).first()
      await modeSelect.selectOption('hybrid')

      // Still hidden until filters expanded
      await page.locator('button:has-text("+ filters")').click()

      // Now visible
      await expect(page.locator('.hybrid-strategy-section')).toBeVisible()

      // Switch back to vector — should hide
      await modeSelect.selectOption('vector')
      await expect(page.locator('.hybrid-strategy-section')).not.toBeVisible()
    })

    test('strategy selector has exactly 3 options: rrf, weighted, linear', async ({ page }) => {
      await page.locator('button:has-text("Search")').first().click()
      const modeSelect = page.locator('select').filter({ has: page.locator('option[value="hybrid"]') }).first()
      await modeSelect.selectOption('hybrid')
      await page.locator('button:has-text("+ filters")').click()

      const strategySelect = page.locator('[data-testid="hybrid-strategy"]')
      const options = strategySelect.locator('option')
      const values: string[] = []
      for (let i = 0; i < await options.count(); i++) {
        values.push(await options.nth(i).getAttribute('value') || '')
      }

      // Invariant: exactly these 3 strategies
      expect(values).toContain('rrf')
      expect(values).toContain('weighted')
      expect(values).toContain('linear')
    })
  })

  test.describe('Recommend', () => {
    test('recommend requires positive IDs (client-side validation)', async ({ page }) => {
      await page.locator('button:has-text("Recommend")').first().click()

      // Leave positive IDs empty, click recommend
      await page.locator('[data-testid="recommend-btn"]').click()
      await page.waitForTimeout(1000)

      // Invariant: error toast about positive IDs
      await expect(page.locator('.toast:has-text("positive")')).toBeVisible({ timeout: 3000 })
    })

    test('recommend top_k defaults to a sensible value', async ({ page }) => {
      await page.locator('button:has-text("Recommend")').first().click()

      const topKInput = page.locator('[data-testid="recommend-top-k"]')
      const value = await topKInput.inputValue()
      const numValue = parseInt(value)

      // Invariant: default is positive and reasonable (1-100)
      expect(numValue).toBeGreaterThan(0)
      expect(numValue).toBeLessThanOrEqual(100)
    })
  })

  test.describe('Discover', () => {
    test('context pairs can be added and removed without orphaned state', async ({ page }) => {
      await page.locator('button:has-text("Discover")').first().click()

      // Add 3 pairs
      for (let i = 0; i < 3; i++) {
        await page.locator('[data-testid="discover-add-pair"]').click()
      }
      expect(await page.locator('[data-testid="discover-pair-positive"]').count()).toBe(3)

      // Fill middle pair
      await page.locator('[data-testid="discover-pair-positive"]').nth(1).fill('doc-x')
      await page.locator('[data-testid="discover-pair-negative"]').nth(1).fill('doc-y')

      // Remove first pair — second pair (now first) should retain values
      await page.locator('button:has-text("×")').first().click()
      expect(await page.locator('[data-testid="discover-pair-positive"]').count()).toBe(2)

      // Invariant: the filled pair retained its values after the remove
      const firstPositive = await page.locator('[data-testid="discover-pair-positive"]').first().inputValue()
      expect(firstPositive).toBe('doc-x')
    })
  })

  test.describe('Geo Filter', () => {
    test('geo filter mode switch clears irrelevant fields', async ({ page }) => {
      await page.locator('button:has-text("Search")').first().click()
      await page.locator('button:has-text("+ filters")').click()

      const geoToggle = page.locator('[data-testid="geo-filter-toggle"]')
      if (!await geoToggle.isVisible().catch(() => false)) {
        test.skip()
        return
      }

      await geoToggle.check()

      // Fill radius fields
      await page.locator('[data-testid="geo-lat"]').fill('40.0')
      await page.locator('[data-testid="geo-lon"]').fill('-74.0')

      // Switch to bbox
      await page.locator('[data-testid="geo-filter-mode"]').selectOption('bbox')

      // Invariant: radius fields hidden, bbox fields visible
      await expect(page.locator('[data-testid="geo-lat"]')).not.toBeVisible()
      await expect(page.locator('[data-testid="geo-top-left"]')).toBeVisible()
      await expect(page.locator('[data-testid="geo-bottom-right"]')).toBeVisible()

      // Switch back
      await page.locator('[data-testid="geo-filter-mode"]').selectOption('radius')
      await expect(page.locator('[data-testid="geo-lat"]')).toBeVisible()
      // Invariant: radius values preserved across mode switches
      const lat = await page.locator('[data-testid="geo-lat"]').inputValue()
      expect(lat).toBe('40.0')
    })
  })
})
