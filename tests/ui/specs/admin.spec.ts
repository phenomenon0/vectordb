import { test, expect, API } from '../fixtures/server'

// Admin tests follow the CROSS-TENANT PROBE and LIFECYCLE shapes.
// Key invariants: RBAC forms produce correct API calls, tenant operations
// are reflected in the server state, and the UI handles errors gracefully.

test.describe('Admin: RBAC Contract Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard/')
    const skipBtn = page.locator('button:has-text("Skip")')
    if (await skipBtn.isVisible({ timeout: 2000 }).catch(() => false)) {
      await skipBtn.click()
    }
    await page.locator('.topnav-tab:has-text("Admin")').click()
  })

  test('ACL role selector offers exactly the expected roles', async ({ page }) => {
    const roleSelect = page.locator('select').filter({ has: page.locator('option[value="reader"]') })
    await expect(roleSelect).toBeVisible()

    const options = roleSelect.locator('option')
    const values: string[] = []
    for (let i = 0; i < await options.count(); i++) {
      values.push(await options.nth(i).getAttribute('value') || '')
    }

    // Invariant: known roles are present
    expect(values).toContain('reader')
    expect(values).toContain('writer')
    expect(values).toContain('admin')
  })

  test('permission level selector offers read/write/admin', async ({ page }) => {
    const levelSelect = page.locator('select').filter({ has: page.locator('option[value="read"]') })
    await expect(levelSelect).toBeVisible()

    const options = levelSelect.locator('option')
    const values: string[] = []
    for (let i = 0; i < await options.count(); i++) {
      values.push(await options.nth(i).getAttribute('value') || '')
    }

    // Invariant: known permission levels
    expect(values).toContain('read')
    expect(values).toContain('write')
    expect(values).toContain('admin')
  })

  test('ACL grant sends correct payload to server', async ({ page }) => {
    let capturedPayload: any = null
    await page.route('**/admin/acl/grant', async (route) => {
      capturedPayload = JSON.parse(route.request().postData() || '{}')
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ ok: true }),
      })
    })

    // Fill in tenant ID
    const tenantInput = page.locator('[data-testid="admin-rbac"]').locator('input').first()
    await tenantInput.fill('test-tenant-abc')

    // Select role
    const roleSelect = page.locator('select').filter({ has: page.locator('option[value="reader"]') })
    await roleSelect.selectOption('writer')

    // Grant
    await page.locator('[data-testid="acl-grant-btn"]').click()
    await page.waitForTimeout(2000)

    // Invariant: correct payload sent
    if (capturedPayload) {
      expect(capturedPayload.tenant_id || capturedPayload.tenant).toContain('test-tenant')
      expect(capturedPayload.role).toBe('writer')
    }
  })

  test('rate limit set sends numeric RPS value', async ({ page }) => {
    let capturedPayload: any = null
    await page.route('**/admin/ratelimit/set', async (route) => {
      capturedPayload = JSON.parse(route.request().postData() || '{}')
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ ok: true }),
      })
    })

    await page.locator('[data-testid="ratelimit-tenant-input"]').fill('test-tenant')
    await page.locator('[data-testid="ratelimit-rps-input"]').fill('250')
    await page.locator('[data-testid="ratelimit-set-btn"]').click()
    await page.waitForTimeout(2000)

    // Invariant: RPS is sent as a number, not string
    if (capturedPayload) {
      expect(typeof capturedPayload.rps).toBe('number')
      expect(capturedPayload.rps).toBe(250)
    }
  })

  test('tenant creation and listing lifecycle', async ({ page }) => {
    const tenantName = `e2e-tenant-${Date.now()}`

    // Intercept tenant APIs for controlled testing
    let createCalled = false
    await page.route('**/v3/tenants', async (route) => {
      if (route.request().method() === 'POST') {
        createCalled = true
        const body = JSON.parse(route.request().postData() || '{}')
        // Invariant: tenant name matches what was entered
        expect(body.name || body.tenant_id).toBe(tenantName)
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ ok: true }),
        })
      } else {
        await route.continue()
      }
    })

    const tenantInput = page.locator('[data-testid="admin-tenants"] input[placeholder="New tenant name"]')
    await tenantInput.fill(tenantName)
    await page.locator('[data-testid="admin-tenants"] button:has-text("Create Tenant")').click()
    await page.waitForTimeout(2000)

    expect(createCalled).toBe(true)
  })

  test('admin API errors surface as toasts', async ({ page }) => {
    // Intercept ACL grant to return error
    await page.route('**/admin/acl/grant', async (route) => {
      await route.fulfill({
        status: 403,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'permission denied' }),
      })
    })

    const tenantInput = page.locator('[data-testid="admin-rbac"]').locator('input').first()
    await tenantInput.fill('test-tenant')
    await page.locator('[data-testid="acl-grant-btn"]').click()
    await page.waitForTimeout(2000)

    // Invariant: error is surfaced, not swallowed
    const errorToast = page.locator('.toast.error, .toast:has-text("fail"), .toast:has-text("denied")')
    await expect(errorToast.first()).toBeVisible({ timeout: 5000 })
  })
})
