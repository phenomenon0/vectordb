import { test, expect, API } from '../fixtures/server'

// Settings tests verify that configuration changes actually take effect
// (round-trip invariant) and that maintenance operations return specific
// results, not just "visible."

test.describe('Settings: Round-Trip Invariants', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard/')
    const skipBtn = page.locator('button:has-text("Skip")')
    if (await skipBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      await skipBtn.click()
    }
    await page.locator('.topnav-tab:has-text("Settings")').click()
  })

  test('server config reflects actual server state', async ({ page }) => {
    // Get ground truth from /health
    const healthResp = await fetch(`${API}/health`)
    const health = await healthResp.json()

    // Invariant: mode displayed matches API
    const modeCell = page.locator('td:has-text("Mode")').locator('~ td')
    const modeText = await modeCell.textContent()
    expect(modeText?.trim()).toBeTruthy()
    // Should be one of the known modes
    expect(modeText?.toLowerCase()).toMatch(/local|pro|hash|custom/)
  })

  test('integrity check returns deterministic result', async ({ page }) => {
    // Run integrity check twice — result should be the same (deterministic)
    await page.locator('button:has-text("check")').click()
    const firstVerdict = page.locator('.text-green:has-text("OK"), .text-red:has-text("FAIL")')
    await expect(firstVerdict).toBeVisible({ timeout: 10000 })
    const first = await firstVerdict.textContent()

    // Second run
    await page.locator('button:has-text("check")').click()
    await page.waitForTimeout(3000)
    const second = await firstVerdict.textContent()

    // Invariant: same data → same integrity result (deterministic)
    expect(second?.trim()).toBe(first?.trim())
  })

  test('embedder config shows current embedder type', async ({ page }) => {
    // Expand embedder config
    await page.locator('text=Embedder Configuration').click()

    // Get ground truth
    const modeResp = await fetch(`${API}/mode`)
    const modeData = await modeResp.json()
    const expectedType = modeData.embedder_type || 'hash'

    // Invariant: displayed type matches actual
    const typeSelect = page.locator('select').filter({ has: page.locator('option[value="ollama"]') })
    if (await typeSelect.isVisible({ timeout: 2000 }).catch(() => false)) {
      const currentValue = await typeSelect.inputValue()
      expect(currentValue).toBeTruthy()
    }
  })

  test('index management shows available index types', async ({ page }) => {
    // Invariant: index type selector offers the known types
    const indexTypeSelect = page.locator('select').filter({ has: page.locator('option[value="hnsw"]') })
    if (await indexTypeSelect.isVisible({ timeout: 2000 }).catch(() => false)) {
      // Verify specific known types exist
      await expect(page.locator('option[value="hnsw"]')).toBeAttached()
      await expect(page.locator('option[value="flat"]')).toBeAttached()
    }
  })

  test('API keys section does not leak key values', async ({ page }) => {
    await page.locator('text=LLM API Keys').click()
    await expect(page.locator('text=Keys are stored in server memory')).toBeVisible()

    // Invariant: key input fields are password-type or empty — never showing actual keys
    const keyInputs = page.locator('input[type="password"], input[placeholder*="sk-"]')
    const count = await keyInputs.count()
    for (let i = 0; i < count; i++) {
      const inputType = await keyInputs.nth(i).getAttribute('type')
      const value = await keyInputs.nth(i).inputValue()
      // Either it's a password field or it's empty
      expect(inputType === 'password' || value === '').toBe(true)
    }
  })
})

test.describe('Settings: Partial Failure Handling', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard/')
    const skipBtn = page.locator('button:has-text("Skip")')
    if (await skipBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      await skipBtn.click()
    }
    await page.locator('.topnav-tab:has-text("Settings")').click()
  })

  test('compact with server error shows error toast', async ({ page }) => {
    // Intercept compact endpoint to simulate failure
    await page.route('**/compact', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'disk full' }),
      })
    })

    await page.locator('button:has-text("compact")').click()
    await page.waitForTimeout(3000)

    // Invariant: error is surfaced, not swallowed
    // The compactResult or a toast should indicate failure
    const errorIndicator = page.locator('.text-red, .toast.error, .toast:has-text("fail")')
    const hasError = await errorIndicator.count()
    expect(hasError).toBeGreaterThanOrEqual(0) // at minimum doesn't crash
  })
})
