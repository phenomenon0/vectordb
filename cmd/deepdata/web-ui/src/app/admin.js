// Admin & Security mixin: tenants, RBAC, rate limits, quotas
export const adminMixin = {
  async loadTenants() {
    const r = await this.api('GET', '/v3/tenants')
    if (r.ok) this.tenants = r.data?.tenants ?? r.data ?? []
  },

  async createTenant(name) {
    const r = await this.api('POST', '/v3/tenants', { name })
    if (r.ok) {
      this.toast('Tenant created: ' + name, 'success')
      await this.loadTenants()
    } else {
      this.toast('Failed to create tenant: ' + (r.data?.error || r.status), 'error')
    }
  },

  async deleteTenant(id) {
    const r = await this.api('DELETE', `/v3/tenants/${id}`)
    if (r.ok) {
      this.toast('Tenant deleted', 'success')
      await this.loadTenants()
    } else {
      this.toast('Failed to delete tenant: ' + (r.data?.error || r.status), 'error')
    }
  },

  async loadTenantCollections(tenantId) {
    const r = await this.api('GET', `/v3/tenants/${tenantId}/collections`)
    if (r.ok) {
      this.tenantCollections[tenantId] = r.data?.collections ?? r.data ?? []
    }
  },

  async grantAcl(tenantId, role) {
    const r = await this.api('POST', '/admin/acl/grant', { tenant_id: tenantId, role })
    if (r.ok) {
      this.toast('ACL granted: ' + role + ' on ' + tenantId, 'success')
    } else {
      this.toast('Failed to grant ACL: ' + (r.data?.error || r.status), 'error')
    }
  },

  async revokeAcl(tenantId, role) {
    const r = await this.api('POST', '/admin/acl/revoke', { tenant_id: tenantId, role })
    if (r.ok) {
      this.toast('ACL revoked: ' + role + ' on ' + tenantId, 'success')
    } else {
      this.toast('Failed to revoke ACL: ' + (r.data?.error || r.status), 'error')
    }
  },

  async grantPermission(body) {
    const r = await this.api('POST', '/admin/permission/grant', body)
    if (r.ok) {
      this.toast('Permission granted', 'success')
    } else {
      this.toast('Failed to grant permission: ' + (r.data?.error || r.status), 'error')
    }
  },

  async revokePermission(body) {
    const r = await this.api('POST', '/admin/permission/revoke', body)
    if (r.ok) {
      this.toast('Permission revoked', 'success')
    } else {
      this.toast('Failed to revoke permission: ' + (r.data?.error || r.status), 'error')
    }
  },

  async setRateLimit(tenantId, rps) {
    const r = await this.api('POST', '/admin/ratelimit/set', { tenant_id: tenantId, rps })
    if (r.ok) {
      this.toast('Rate limit set: ' + rps + ' rps', 'success')
    } else {
      this.toast('Failed to set rate limit: ' + (r.data?.error || r.status), 'error')
    }
  },

  async loadQuota(tenantId) {
    const r = await this.api('GET', `/admin/quota/${tenantId}`)
    if (r.ok) {
      this.quotaData = r.data ?? {}
    }
  },

  async setQuota(body) {
    const r = await this.api('POST', '/admin/quota/set', body)
    if (r.ok) {
      this.toast('Quota updated', 'success')
    } else {
      this.toast('Failed to set quota: ' + (r.data?.error || r.status), 'error')
    }
  },
}
