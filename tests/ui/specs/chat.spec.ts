import { test, expect } from '../fixtures/server'

// Chat tests verify the LLM connection state machine and provider switching.
// Invariants: provider change updates URL, state transitions are clean,
// disconnect is not confused with "never connected."

test.describe('Chat: State Machine Invariants', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard/')
    const skipBtn = page.locator('button:has-text("Skip")')
    if (await skipBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      await skipBtn.click()
    }
    await page.locator('.topnav-tab:has-text("Chat")').click()
  })

  test('provider selector offers all expected providers', async ({ page }) => {
    const providerSelect = page.locator('select').first()
    const options = providerSelect.locator('option')
    const optionTexts: string[] = []
    for (let i = 0; i < await options.count(); i++) {
      optionTexts.push(await options.nth(i).textContent() || '')
    }

    // Invariant: all documented providers are available
    const expectedProviders = ['Ollama', 'OpenRouter']
    for (const p of expectedProviders) {
      expect(optionTexts.some(t => t.includes(p))).toBe(true)
    }
  })

  test('provider switch updates URL field to correct default', async ({ page }) => {
    // Get initial URL
    const urlInput = page.locator('input[placeholder*="localhost"]').first()
    const initialUrl = await urlInput.inputValue()

    // Switch to deepseek
    const providerSelect = page.locator('select').first()
    await providerSelect.selectOption('deepseek')

    // Invariant: URL changed to deepseek endpoint
    const newUrl = await urlInput.inputValue()
    expect(newUrl).not.toBe(initialUrl)
    expect(newUrl.toLowerCase()).toContain('deepseek')

    // Switch back to ollama
    await providerSelect.selectOption('ollama')
    const restoredUrl = await urlInput.inputValue()
    // Invariant: switching back restores the original URL pattern
    expect(restoredUrl.toLowerCase()).toContain('localhost')
  })

  test('not-connected state shows guidance, not empty page', async ({ page }) => {
    // Invariant: user sees actionable guidance when not connected
    const guidance = page.locator('text=Connect an LLM to chat with your database')
    await expect(guidance).toBeVisible()

    // Invariant: quick-start cards provide setup instructions
    const cards = page.locator('text=Ollama (Local), text=OpenRouter (Cloud)')
    // At least one quick-start card should be visible
    await expect(page.locator('text=Ollama (Local)')).toBeVisible()
  })

  test('connect button exists and is not disabled by default', async ({ page }) => {
    const connectBtn = page.locator('button:has-text("Connect")')
    await expect(connectBtn).toBeVisible()
    // Invariant: connect button is clickable (not disabled)
    await expect(connectBtn).toBeEnabled()
  })

  test('connect to unreachable LLM shows error, not hang', async ({ page }) => {
    // Set URL to something unreachable
    const urlInput = page.locator('input[placeholder*="localhost"]').first()
    await urlInput.fill('http://localhost:19999/v1')

    await page.locator('button:has-text("Connect")').click()

    // Invariant: connection attempt completes within timeout (no infinite spinner)
    await page.waitForTimeout(5000)

    // Invariant: UI is not stuck in "connecting" state
    const connecting = page.locator('text=Connecting...')
    // Should either show error or revert to not-connected
    const isStillConnecting = await connecting.isVisible().catch(() => false)
    // If still showing after 5s, that's a bug (should timeout)
    // Allow some slack since the connection might still be trying
    if (isStillConnecting) {
      await page.waitForTimeout(10000)
      // After 15s total, must not be stuck
      await expect(connecting).not.toBeVisible({ timeout: 5000 })
    }
  })

  test('DeepData context section contains valid system prompt', async ({ page }) => {
    await expect(page.locator('text=DeepData Context for Agents')).toBeVisible()
    await expect(page.locator('text=System prompt snippet')).toBeVisible()

    // Invariant: the context section contains actual DeepData instructions
    const contextText = await page.locator('text=System prompt snippet').locator('..').textContent()
    expect(contextText).toBeTruthy()
    expect(contextText!.length).toBeGreaterThan(20) // not an empty placeholder
  })
})
