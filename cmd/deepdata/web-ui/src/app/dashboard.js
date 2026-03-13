// Dashboard: health, mode, indexes, costs, feedback, integrity, uptime
export const dashboardMixin = {
  async loadHealth() {
    const r = await this.api('GET', '/health')
    if (r.ok) this.health = r.data
  },
  async loadMode() {
    const r = await this.api('GET', '/api/mode')
    if (r.ok && !r.data.error) this.mode = r.data
  },
  async loadIndexes() {
    const r = await this.api('GET', '/api/index/list')
    if (r.ok) this.indexes = r.data.indexes || {}
  },
  async loadDashCosts() {
    const r = await this.api('GET', '/api/costs')
    if (r.ok) this.dashCosts = r.data
  },
  async loadDashFeedback() {
    const r = await this.api('GET', '/v2/feedback/stats')
    if (r.ok) this.dashFeedback = r.data
  },
  async loadDashIntegrity() {
    const r = await this.api('GET', '/integrity')
    this.dashIntegrityResult = r.ok ? r.data : { ok: false, error: 'request failed' }
  },
  updateUptime() {
    if (!this.startTime) return
    const s = Math.floor((Date.now() - this.startTime) / 1000)
    const h = Math.floor(s / 3600)
    const m = Math.floor((s % 3600) / 60)
    const sec = s % 60
    this.uptime = (h ? h + 'h ' : '') + m + 'm ' + sec + 's'
  },

  // ---- Monitoring ----
  async loadMetrics() {
    try {
      const r = await fetch(this.apiBase + '/metrics')
      if (!r.ok) return
      const text = await r.text()
      const m = {}
      for (const line of text.split('\n')) {
        if (line.startsWith('#') || !line.trim()) continue
        // Parse: deepdata_request_duration_seconds{quantile="0.5"} 0.021
        const match = line.match(/^(\S+?)(?:\{([^}]*)\})?\s+(\S+)$/)
        if (!match) continue
        const [, name, labels, value] = match
        const val = parseFloat(value)
        if (isNaN(val)) continue
        if (name === 'deepdata_request_duration_seconds') {
          const qm = (labels || '').match(/quantile="([^"]+)"/)
          if (qm) {
            if (qm[1] === '0.5') m.request_duration_p50 = val
            else if (qm[1] === '0.95') m.request_duration_p95 = val
          }
        } else if (name === 'deepdata_requests_total') {
          m.request_qps = val
        } else if (name === 'deepdata_memory_bytes' || name === 'process_resident_memory_bytes') {
          m.memory_bytes = val
        } else if (name === 'deepdata_disk_bytes') {
          m.disk_bytes = val
        }
      }
      this.metrics = m
      this.metricsLoaded = true
    } catch {
      // /metrics may not be available — silently ignore
    }
  },

  async loadIndexTypes() {
    const r = await this.api('GET', '/api/index/types')
    if (r.ok) this.indexTypes = r.data || []
  },

  async loadIndexStats() {
    const r = await this.api('GET', '/api/index/stats')
    if (r.ok) this.indexDetailedStats = r.data || {}
  },

  fmtMetric(val, unit) {
    if (val === undefined || val === null) return '-'
    if (unit === 'ms') return (val * 1000).toFixed(1) + ' ms'
    if (unit === 'bytes') {
      if (val > 1073741824) return (val / 1073741824).toFixed(1) + ' GB'
      if (val > 1048576) return (val / 1048576).toFixed(1) + ' MB'
      if (val > 1024) return (val / 1024).toFixed(1) + ' KB'
      return val + ' B'
    }
    if (unit === 'qps') return val.toLocaleString()
    return String(val)
  },
}
