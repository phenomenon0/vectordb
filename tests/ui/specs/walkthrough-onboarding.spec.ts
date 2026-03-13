import { test, expect, API } from '../fixtures/server'

// LIFECYCLE TEST SHAPE: onboarding wizard → dashboard → first search → verify results
// Tests the invariant that a fresh user can go from zero to search results
// without hitting any dead ends.

test.describe('Walkthrough: Onboarding → First Search', () => {
  test('fresh user completes onboarding and gets search results', async ({ page }) => {
    // Simulate fresh user: clear all state
    await page.goto('/dashboard/')
    await page.evaluate(() => {
      localStorage.clear()
    })
    await page.reload()

    // ---- STEP 1: WELCOME ----
    await expect(page.locator('.onboarding-overlay')).toBeVisible({ timeout: 5000 })
    // Invariant: step 1 title is exactly "Welcome to DeepData"
    await expect(page.locator('h2:has-text("Welcome to DeepData")')).toBeVisible()
    await page.locator('button:has-text("Next")').click()

    // ---- STEP 2: MODE ----
    await expect(page.locator('h2:has-text("Choose Mode")')).toBeVisible()
    // Invariant: local mode is pre-selected (sensible default)
    await expect(page.locator('[data-testid="mode-local"]')).toHaveClass(/selected/)
    await page.locator('button:has-text("Next")').click()

    // ---- STEP 3: EMBEDDER ----
    await expect(page.locator('h2:has-text("Configure Embedder")')).toBeVisible()
    // Don't configure — just proceed (hash embedder is already active)
    await page.locator('button:has-text("Next")').click()

    // ---- STEP 4: FIRST COLLECTION ----
    // Skip instead of creating (test-collection already exists from seeding)
    await page.locator('button:has-text("Skip")').click()

    // ---- VERIFY: DASHBOARD LOADED ----
    await expect(page.locator('.onboarding-overlay')).not.toBeVisible()
    await expect(page.locator('.status-dot.online')).toBeVisible({ timeout: 10000 })

    // Invariant: onboarding flag is persisted (won't show again)
    const onboarded = await page.evaluate(() => localStorage.getItem('deepdata_onboarded'))
    expect(onboarded).toBeTruthy()

    // ---- NAVIGATE TO SEARCH ----
    await page.locator('.topnav-tab:has-text("Data")').click()
    await page.locator('button:has-text("Search")').first().click()

    // ---- RUN FIRST SEARCH ----
    await page.locator('select').nth(1).selectOption('test-collection')
    await page.locator('input[placeholder="Enter search query..."]').fill('machine learning')
    await page.locator('button:has-text("Search")').last().click()

    // ---- VERIFY: GOT RESULTS ----
    await expect(page.locator('.result-card').first()).toBeVisible({ timeout: 10000 })
    const resultCount = await page.locator('.result-card').count()
    expect(resultCount).toBeGreaterThan(0)

    // Invariant: results contain actual document content (not placeholders)
    const firstResultText = await page.locator('.result-card').first().textContent()
    expect(firstResultText!.length).toBeGreaterThan(10)
  })

  test('onboarding back/forward navigation is consistent', async ({ page }) => {
    await page.goto('/dashboard/')
    await page.evaluate(() => localStorage.clear())
    await page.reload()

    await expect(page.locator('.onboarding-overlay')).toBeVisible({ timeout: 5000 })

    // Forward to step 2
    await page.locator('button:has-text("Next")').click()
    await expect(page.locator('h2:has-text("Choose Mode")')).toBeVisible()

    // Select Pro mode
    await page.locator('[data-testid="mode-pro"]').click()

    // Forward to step 3
    await page.locator('button:has-text("Next")').click()
    // Invariant: step 3 reflects the Pro choice (should show API key input)
    await expect(page.locator('input[placeholder="sk-..."]')).toBeVisible()

    // Go back to step 2
    await page.locator('button:has-text("Back")').click()
    // Invariant: Pro mode selection is preserved after back navigation
    await expect(page.locator('[data-testid="mode-pro"]')).toHaveClass(/selected/)

    // Switch to Local and go forward again
    await page.locator('[data-testid="mode-local"]').click()
    await page.locator('button:has-text("Next")').click()
    // Invariant: step 3 now reflects Local choice
    await expect(page.locator('input[placeholder="http://localhost:11434"]')).toBeVisible()
  })
})
