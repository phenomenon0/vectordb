import { test, expect } from '../fixtures/server'

// Knowledge graph UI state tests. Behavioral contract tests are in
// walkthrough-knowledge-graph.spec.ts. These verify the UI controls
// and state machine invariants.

test.describe('Knowledge Graph: UI Controls', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard/')
    const skipBtn = page.locator('button:has-text("Skip")')
    if (await skipBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      await skipBtn.click()
    }
    await page.locator('.topnav-tab:has-text("Data")').click()
    await page.locator('button:has-text("Knowledge Graph")').click()
  })

  test('all 7 entity types are available as checkboxes', async ({ page }) => {
    const expectedTypes = ['person', 'organization', 'location', 'concept', 'event', 'product', 'technology']
    for (const t of expectedTypes) {
      const checkbox = page.locator(`.kg-type-checkbox:has-text("${t}")`)
      await expect(checkbox).toBeVisible()
    }
    // Invariant: exactly 7 types, no more no less
    const allCheckboxes = page.locator('.kg-type-checkbox')
    expect(await allCheckboxes.count()).toBe(7)
  })

  test('extract button enable/disable tracks textarea content', async ({ page }) => {
    const extractBtn = page.locator('button.kg-extract-btn')
    const textarea = page.locator('textarea.kg-input-text')

    // Invariant: disabled when empty
    await expect(extractBtn).toBeDisabled()

    // Invariant: enabled when has content
    await textarea.fill('test')
    await expect(extractBtn).toBeEnabled()

    // Invariant: disabled again when cleared
    await textarea.fill('')
    await expect(extractBtn).toBeDisabled()

    // Invariant: whitespace-only still enables (server-side validation handles this)
    await textarea.fill('   ')
    // The button should be enabled for non-empty string (trim is server's job)
    await expect(extractBtn).toBeEnabled()
  })

  test('batch/temporal mutual exclusivity is enforced', async ({ page }) => {
    const batchCb = page.locator('label:has-text("Batch Mode") input[type="checkbox"]')
    const temporalCb = page.locator('label:has-text("Temporal Mode") input[type="checkbox"]')

    // Neither checked initially
    expect(await batchCb.isChecked()).toBe(false)
    expect(await temporalCb.isChecked()).toBe(false)

    // Check batch
    await batchCb.check()
    expect(await batchCb.isChecked()).toBe(true)
    expect(await temporalCb.isChecked()).toBe(false)

    // Check temporal — must uncheck batch
    await temporalCb.check()
    expect(await temporalCb.isChecked()).toBe(true)
    expect(await batchCb.isChecked()).toBe(false)

    // Check batch again — must uncheck temporal
    await batchCb.check()
    expect(await batchCb.isChecked()).toBe(true)
    expect(await temporalCb.isChecked()).toBe(false)

    // Uncheck batch — both off
    await batchCb.uncheck()
    expect(await batchCb.isChecked()).toBe(false)
    expect(await temporalCb.isChecked()).toBe(false)
  })
})
