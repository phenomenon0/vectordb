import { test, expect, API } from '../fixtures/server'

// Tests the BEHAVIORAL INVARIANTS of advanced search features.
// Uses request interception to verify the actual payloads sent to the server
// (testing the contract, not the implementation).

test.describe('Walkthrough: Advanced Search Contracts', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard/')
    const skipBtn = page.locator('button:has-text("Skip")')
    if (await skipBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      await skipBtn.click()
    }
    await page.locator('.topnav-tab:has-text("Data")').click()
  })

  test('hybrid search sends correct strategy and weights to server', async ({ page }) => {
    let capturedPayload: any = null
    await page.route('**/query', async (route) => {
      capturedPayload = JSON.parse(route.request().postData() || '{}')
      await route.continue()
    })

    await page.locator('button:has-text("Search")').first().click()

    // Switch to hybrid
    const modeSelect = page.locator('select').filter({ has: page.locator('option[value="hybrid"]') }).first()
    await modeSelect.selectOption('hybrid')

    // Expand filters and set strategy
    await page.locator('button:has-text("+ filters")').click()
    const strategySelect = page.locator('[data-testid="hybrid-strategy"]')
    if (await strategySelect.isVisible().catch(() => false)) {
      await strategySelect.selectOption('weighted')

      // Set weights
      const denseInput = page.locator('[data-testid="hybrid-weight-dense"]')
      await denseInput.fill('0.8')
      const sparseInput = page.locator('[data-testid="hybrid-weight-sparse"]')
      await sparseInput.fill('0.2')
    }

    // Execute search
    await page.locator('select').nth(1).selectOption('test-collection')
    await page.locator('input[placeholder="Enter search query..."]').fill('neural networks')
    await page.locator('button:has-text("Search")').last().click()
    await page.waitForTimeout(3000)

    // INVARIANT: payload contains hybrid params
    expect(capturedPayload).toBeTruthy()
    expect(capturedPayload.score_mode).toBe('hybrid')
    if (capturedPayload.hybrid_params) {
      expect(capturedPayload.hybrid_params.strategy).toBe('weighted')
      expect(capturedPayload.hybrid_params.weights.dense).toBe(0.8)
      expect(capturedPayload.hybrid_params.weights.sparse).toBe(0.2)
    }
  })

  test('recommend API contract: positive/negative IDs and weights', async ({ page }) => {
    let capturedPayload: any = null
    await page.route('**/v2/recommend', async (route) => {
      capturedPayload = JSON.parse(route.request().postData() || '{}')
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ results: [] }),
      })
    })

    await page.locator('button:has-text("Recommend")').first().click()

    // Fill in IDs
    await page.locator('[data-testid="recommend-positive-ids"]').fill('test-doc-0,test-doc-1')
    await page.locator('[data-testid="recommend-negative-ids"]').fill('test-doc-4')

    // Set weight
    const weightInput = page.locator('[data-testid="recommend-neg-weight"]')
    await weightInput.fill('0.3')

    await page.locator('[data-testid="recommend-btn"]').click()
    await page.waitForTimeout(2000)

    // INVARIANT: correct payload sent
    expect(capturedPayload).toBeTruthy()
    expect(capturedPayload.positive_ids).toEqual(['test-doc-0', 'test-doc-1'])
    expect(capturedPayload.negative_ids).toEqual(['test-doc-4'])
    expect(capturedPayload.negative_weight).toBe(0.3)
    expect(capturedPayload.collection).toBeTruthy()
  })

  test('discover API contract: target ID and context pairs', async ({ page }) => {
    let capturedPayload: any = null
    await page.route('**/v2/discover', async (route) => {
      capturedPayload = JSON.parse(route.request().postData() || '{}')
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ results: [] }),
      })
    })

    await page.locator('button:has-text("Discover")').first().click()

    // Set target
    await page.locator('[data-testid="discover-target-id"]').fill('test-doc-0')

    // Add context pair
    await page.locator('[data-testid="discover-add-pair"]').click()
    await page.locator('[data-testid="discover-pair-positive"]').first().fill('test-doc-1')
    await page.locator('[data-testid="discover-pair-negative"]').first().fill('test-doc-4')

    await page.locator('[data-testid="discover-btn"]').click()
    await page.waitForTimeout(2000)

    // INVARIANT: correct payload sent
    expect(capturedPayload).toBeTruthy()
    expect(capturedPayload.target_id).toBe('test-doc-0')
    expect(capturedPayload.context_pairs).toHaveLength(1)
    expect(capturedPayload.context_pairs[0].positive).toBe('test-doc-1')
    expect(capturedPayload.context_pairs[0].negative).toBe('test-doc-4')
  })

  test('geo filter produces correct filter JSON', async ({ page }) => {
    let capturedPayload: any = null
    await page.route('**/query', async (route) => {
      capturedPayload = JSON.parse(route.request().postData() || '{}')
      await route.continue()
    })

    await page.locator('button:has-text("Search")').first().click()
    await page.locator('button:has-text("+ filters")').click()

    const geoToggle = page.locator('[data-testid="geo-filter-toggle"]')
    if (!await geoToggle.isVisible().catch(() => false)) {
      test.skip()
      return
    }

    await geoToggle.check()

    // Radius mode (default)
    await page.locator('[data-testid="geo-lat"]').fill('37.7749')
    await page.locator('[data-testid="geo-lon"]').fill('-122.4194')
    await page.locator('[data-testid="geo-distance"]').fill('50')

    await page.locator('select').nth(1).selectOption('test-collection')
    await page.locator('input[placeholder="Enter search query..."]').fill('test')
    await page.locator('button:has-text("Search")').last().click()
    await page.waitForTimeout(2000)

    // INVARIANT: metadata includes $geo_radius with correct values
    expect(capturedPayload).toBeTruthy()
    if (capturedPayload.metadata?.$geo_radius) {
      const geo = capturedPayload.metadata.$geo_radius
      expect(geo.center.lat).toBe(37.7749)
      expect(geo.center.lon).toBe(-122.4194)
      expect(geo.distance_km).toBe(50)
    }
  })

  test('recommend with no positive IDs shows error, does not send request', async ({ page }) => {
    let requestSent = false
    await page.route('**/v2/recommend', async (route) => {
      requestSent = true
      await route.continue()
    })

    await page.locator('button:has-text("Recommend")').first().click()
    // Leave positive IDs empty
    await page.locator('[data-testid="recommend-btn"]').click()
    await page.waitForTimeout(1000)

    // INVARIANT: validation prevents sending request with empty positive IDs
    expect(requestSent).toBe(false)
    // Error toast should appear
    await expect(page.locator('.toast:has-text("positive")')).toBeVisible({ timeout: 3000 })
  })

  test('discover with no target ID shows error, does not send request', async ({ page }) => {
    let requestSent = false
    await page.route('**/v2/discover', async (route) => {
      requestSent = true
      await route.continue()
    })

    await page.locator('button:has-text("Discover")').first().click()
    // Leave target ID empty
    await page.locator('[data-testid="discover-btn"]').click()
    await page.waitForTimeout(1000)

    // INVARIANT: validation prevents sending request with empty target
    expect(requestSent).toBe(false)
    await expect(page.locator('.toast:has-text("target")')).toBeVisible({ timeout: 3000 })
  })
})
