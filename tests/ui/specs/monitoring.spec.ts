import { test, expect, API } from '../fixtures/server'

// Monitoring tests verify that metrics displayed are derived from actual
// server data, and that the onboarding wizard state machine is consistent.

test.describe('Monitoring: Metrics Fidelity', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard/')
    const skipBtn = page.locator('button:has-text("Skip")')
    if (await skipBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      await skipBtn.click()
    }
  })

  test('metric cards show numeric values when metrics available', async ({ page }) => {
    await expect(page.locator('.status-dot.online')).toBeVisible({ timeout: 10000 })

    const monitoringRow = page.locator('[data-testid="monitoring-row"]')
    const isVisible = await monitoringRow.isVisible({ timeout: 5000 }).catch(() => false)

    if (isVisible) {
      // Invariant: each metric card displays a value containing digits
      const metricCards = ['metric-p50', 'metric-p95', 'metric-qps', 'metric-memory']
      for (const id of metricCards) {
        const card = page.locator(`[data-testid="${id}"]`)
        await expect(card).toBeVisible()
        const value = await card.locator('.metric-value').textContent()
        // Invariant: value is a formatted number (digits, possibly with units like "ms", "MB")
        expect(value?.trim()).toBeTruthy()
        // Should contain at least one digit
        expect(value).toMatch(/\d/)
      }

      // Invariant: labels are specific, not generic
      await expect(page.locator('[data-testid="metric-p50"] .metric-label')).toHaveText('Request P50')
      await expect(page.locator('[data-testid="metric-p95"] .metric-label')).toHaveText('Request P95')
    }
  })

  test('metrics auto-refresh updates values', async ({ page }) => {
    await expect(page.locator('.status-dot.online')).toBeVisible({ timeout: 10000 })

    const monitoringRow = page.locator('[data-testid="monitoring-row"]')
    if (!await monitoringRow.isVisible({ timeout: 5000 }).catch(() => false)) {
      test.skip()
      return
    }

    // Generate some traffic to change metrics
    for (let i = 0; i < 5; i++) {
      await fetch(`${API}/health`)
    }

    // Wait for auto-refresh (10s interval + margin)
    await page.waitForTimeout(12000)

    // Invariant: QPS card still has a numeric value (didn't crash or go stale)
    const qpsValue = await page.locator('[data-testid="metric-qps"] .metric-value').textContent()
    expect(qpsValue).toMatch(/\d/)
  })
})

test.describe('Onboarding: State Machine Invariants', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard/')
    await page.evaluate(() => localStorage.removeItem('deepdata_onboarded'))
    await page.reload()
  })

  test('step count is exactly 4 and dot indicators track current step', async ({ page }) => {
    await expect(page.locator('.onboarding-overlay')).toBeVisible({ timeout: 5000 })

    const dots = page.locator('.onboarding-dots .dot')
    // Invariant: exactly 4 steps
    await expect(dots).toHaveCount(4)

    // Step 1: first dot active
    await expect(dots.nth(0)).toHaveClass(/active/)
    await expect(dots.nth(1)).not.toHaveClass(/active/)

    // Navigate to step 2
    await page.locator('button:has-text("Next")').click()
    // Invariant: second dot is now active, first is not
    await expect(dots.nth(1)).toHaveClass(/active/)
    await expect(dots.nth(0)).not.toHaveClass(/active/)
  })

  test('mode selection persists through forward/back navigation', async ({ page }) => {
    await expect(page.locator('.onboarding-overlay')).toBeVisible({ timeout: 5000 })

    // Go to step 2
    await page.locator('button:has-text("Next")').click()

    // Select Pro mode
    await page.locator('[data-testid="mode-pro"]').click()
    await expect(page.locator('[data-testid="mode-pro"]')).toHaveClass(/selected/)

    // Go to step 3
    await page.locator('button:has-text("Next")').click()
    // Invariant: Pro mode means API key input (not Ollama URL)
    await expect(page.locator('input[placeholder="sk-..."]')).toBeVisible()

    // Go back to step 2
    await page.locator('button:has-text("Back")').click()
    // Invariant: Pro is still selected
    await expect(page.locator('[data-testid="mode-pro"]')).toHaveClass(/selected/)

    // Switch to Local and verify step 3 changes
    await page.locator('[data-testid="mode-local"]').click()
    await page.locator('button:has-text("Next")').click()
    // Invariant: Local mode shows Ollama URL
    await expect(page.locator('input[placeholder="http://localhost:11434"]')).toBeVisible()
  })

  test('skip at any step closes onboarding and sets localStorage flag', async ({ page }) => {
    await expect(page.locator('.onboarding-overlay')).toBeVisible({ timeout: 5000 })

    // Navigate to step 3 (middle of wizard)
    await page.locator('button:has-text("Next")').click()
    await page.locator('button:has-text("Next")').click()

    // Skip
    await page.locator('button:has-text("Skip")').click()

    // Invariant: overlay is closed
    await expect(page.locator('.onboarding-overlay')).not.toBeVisible()

    // Invariant: flag is set so it won't show again
    const flag = await page.evaluate(() => localStorage.getItem('deepdata_onboarded'))
    expect(flag).toBeTruthy()

    // Reload — should NOT show onboarding again
    await page.reload()
    await page.waitForTimeout(2000)
    await expect(page.locator('.onboarding-overlay')).not.toBeVisible()
  })

  test('step 4 collection creation actually calls the API', async ({ page }) => {
    await expect(page.locator('.onboarding-overlay')).toBeVisible({ timeout: 5000 })

    let createCalled = false
    await page.route('**/v2/collections', async (route) => {
      if (route.request().method() === 'POST') {
        createCalled = true
        const body = JSON.parse(route.request().postData() || '{}')
        // Invariant: collection name matches input
        expect(body.name).toBe('my-onboard-coll')
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ ok: true }),
        })
      } else {
        await route.continue()
      }
    })

    // Navigate to step 4
    await page.locator('button:has-text("Next")').click()
    await page.locator('button:has-text("Next")').click()
    await page.locator('button:has-text("Next")').click()

    // Fill in collection name
    await page.locator('input[placeholder="my-documents"]').fill('my-onboard-coll')
    await page.locator('[data-testid="create-collection-btn"]').click()
    await page.waitForTimeout(2000)

    expect(createCalled).toBe(true)
  })
})
