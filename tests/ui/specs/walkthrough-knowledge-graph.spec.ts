import { test, expect } from '../fixtures/server'

// Knowledge graph tests verify API contract and UI state transitions.
// Since extraction requires an LLM, we use route interception to provide
// controlled responses and verify the UI handles them correctly.

test.describe('Walkthrough: Knowledge Graph Contracts', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard/')
    const skipBtn = page.locator('button:has-text("Skip")')
    if (await skipBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      await skipBtn.click()
    }
    await page.locator('.topnav-tab:has-text("Data")').click()
    await page.locator('button:has-text("Knowledge Graph")').click()
  })

  test('extract sends correct payload with entity types', async ({ page }) => {
    let capturedPayload: any = null
    await page.route('**/v2/extract', async (route) => {
      capturedPayload = JSON.parse(route.request().postData() || '{}')
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          entities: [
            { name: 'Apple Inc', type: 'organization', confidence: 0.95 },
            { name: 'Steve Jobs', type: 'person', confidence: 0.98 },
          ],
          relationships: [
            { source: 'Steve Jobs', relation: 'founded', target: 'Apple Inc' },
          ],
        }),
      })
    })

    // Enter text
    await page.locator('textarea.kg-input-text').fill(
      'Apple Inc was founded by Steve Jobs in Cupertino.'
    )

    // Select entity types
    const personCb = page.locator('.kg-type-checkbox:has-text("person") input[type="checkbox"]')
    const orgCb = page.locator('.kg-type-checkbox:has-text("organization") input[type="checkbox"]')
    if (await personCb.isVisible().catch(() => false)) {
      await personCb.check()
      await orgCb.check()
    }

    await page.locator('button.kg-extract-btn').click()
    await page.waitForTimeout(2000)

    // INVARIANT: correct payload sent
    expect(capturedPayload).toBeTruthy()
    expect(capturedPayload.content || capturedPayload.text).toContain('Apple Inc')
    if (capturedPayload.entity_types) {
      expect(capturedPayload.entity_types).toContain('person')
      expect(capturedPayload.entity_types).toContain('organization')
    }

    // INVARIANT: entities rendered in UI
    await expect(page.locator('text=Apple Inc')).toBeVisible({ timeout: 5000 })
    await expect(page.locator('text=Steve Jobs')).toBeVisible()
    // Relationship rendered
    await expect(page.locator('text=founded')).toBeVisible()
  })

  test('batch mode sends to batch endpoint', async ({ page }) => {
    let requestUrl = ''
    await page.route('**/v2/extract**', async (route) => {
      requestUrl = route.request().url()
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ job_id: 'test-job-123', status: 'completed', entities: [], relationships: [] }),
      })
    })

    // Enable batch mode
    const batchCb = page.locator('label:has-text("Batch Mode") input[type="checkbox"]')
    await batchCb.check()

    await page.locator('textarea.kg-input-text').fill('Text chunk 1\n---\nText chunk 2')
    await page.locator('button.kg-extract-btn').click()
    await page.waitForTimeout(2000)

    // INVARIANT: batch mode uses the batch endpoint
    expect(requestUrl).toContain('batch')
  })

  test('temporal mode sends to temporal endpoint', async ({ page }) => {
    let requestUrl = ''
    await page.route('**/v2/extract**', async (route) => {
      requestUrl = route.request().url()
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ temporal_events: [], entities: [], relationships: [] }),
      })
    })

    // Enable temporal mode
    const temporalCb = page.locator('label:has-text("Temporal Mode") input[type="checkbox"]')
    await temporalCb.check()

    await page.locator('textarea.kg-input-text').fill('WWII ended in 1945.')
    await page.locator('button.kg-extract-btn').click()
    await page.waitForTimeout(2000)

    // INVARIANT: temporal mode uses the temporal endpoint
    expect(requestUrl).toContain('temporal')
  })

  test('batch and temporal are mutually exclusive', async ({ page }) => {
    const batchCb = page.locator('label:has-text("Batch Mode") input[type="checkbox"]')
    const temporalCb = page.locator('label:has-text("Temporal Mode") input[type="checkbox"]')

    // Enable batch
    await batchCb.check()
    expect(await batchCb.isChecked()).toBe(true)
    expect(await temporalCb.isChecked()).toBe(false)

    // Enable temporal — should disable batch
    await temporalCb.check()
    expect(await temporalCb.isChecked()).toBe(true)
    expect(await batchCb.isChecked()).toBe(false)

    // Re-enable batch — should disable temporal
    await batchCb.check()
    expect(await batchCb.isChecked()).toBe(true)
    expect(await temporalCb.isChecked()).toBe(false)
  })

  test('extract button disabled when input is empty', async ({ page }) => {
    const extractBtn = page.locator('button.kg-extract-btn')

    // INVARIANT: empty input → disabled button
    await expect(extractBtn).toBeDisabled()

    // Type something
    await page.locator('textarea.kg-input-text').fill('Some text')
    await expect(extractBtn).toBeEnabled()

    // Clear it
    await page.locator('textarea.kg-input-text').fill('')
    await expect(extractBtn).toBeDisabled()
  })

  test('extraction error surfaces as toast, not silent failure', async ({ page }) => {
    await page.route('**/v2/extract', async (route) => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'LLM provider unavailable' }),
      })
    })

    await page.locator('textarea.kg-input-text').fill('Some text to extract')
    await page.locator('button.kg-extract-btn').click()
    await page.waitForTimeout(3000)

    // INVARIANT: error is visible to user
    const errorToast = page.locator('.toast:has-text("fail"), .toast:has-text("error"), .toast.error')
    await expect(errorToast.first()).toBeVisible({ timeout: 5000 })

    // INVARIANT: not stuck in extracting state
    await expect(page.locator('button.kg-extract-btn')).toBeEnabled({ timeout: 5000 })
  })
})
