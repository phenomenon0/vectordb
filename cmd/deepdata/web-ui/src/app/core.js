// Core utilities: API helper, toast, formatters, navigation, onboarding
export const coreMixin = {
  navigateTo(pageId) {
    const prevPage = this.page
    this.page = pageId
    if (prevPage === 'chat' && pageId !== 'chat') {
      this.stopAgentPolling()
    }
  },

  // ---- Onboarding ----
  onboardingNext() {
    if (this.onboardingStep < this.onboardingSteps.length - 1) {
      this.onboardingStep++
    } else {
      this.completeOnboarding()
    }
  },
  onboardingBack() {
    if (this.onboardingStep > 0) this.onboardingStep--
  },
  skipOnboarding() {
    this.completeOnboarding()
  },
  completeOnboarding() {
    localStorage.setItem('deepdata_onboarded', '1')
    this.showOnboarding = false
  },
  restartOnboarding() {
    this.onboardingStep = 0
    this.showOnboarding = true
  },

  // ---- API helper ----
  async api(method, path, body = null) {
    const t0 = Date.now()
    const opts = { method, headers: {} }
    if (body) {
      opts.headers['Content-Type'] = 'application/json'
      opts.body = JSON.stringify(body)
    }
    try {
      const r = await fetch(this.apiBase + path, opts)
      const ms = Date.now() - t0
      let data = null
      const ct = r.headers.get('content-type') || ''
      if (ct.includes('json')) data = await r.json()
      else data = await r.text()
      this.requestLog.push({ method, path, status: r.status, ok: r.ok, ms })
      if (this.requestLog.length > 100) this.requestLog.shift()
      return { ok: r.ok, status: r.status, data, ms }
    } catch (e) {
      const ms = Date.now() - t0
      this.requestLog.push({ method, path, status: 0, ok: false, ms })
      return { ok: false, status: 0, data: null, ms, error: e.message }
    }
  },

  async agentApi(method, path, body = null) {
    const opts = { method, headers: {} }
    if (body) {
      opts.headers['Content-Type'] = 'application/json'
      opts.body = JSON.stringify(body)
    }
    try {
      const r = await fetch(this.agentGoUrl + path, opts)
      let data = null
      const ct = r.headers.get('content-type') || ''
      if (ct.includes('json')) data = await r.json()
      else data = await r.text()
      return { ok: r.ok, status: r.status, data }
    } catch (e) {
      return { ok: false, status: 0, data: null, error: e.message }
    }
  },

  toast(msg, type = '') { this.toasts.push({ msg, type }) },

  copyText(text) {
    navigator.clipboard.writeText(text).then(() => this.toast('Copied', 'success'))
  },

  // ---- Formatters ----
  fmt(n) { return n == null ? '0' : Number(n).toLocaleString() },
  fmtBytes(b) {
    if (!b || b === 0) return '0 B'
    const u = ['B', 'KB', 'MB', 'GB']
    const i = Math.min(Math.floor(Math.log(b) / Math.log(1024)), u.length - 1)
    return (b / Math.pow(1024, i)).toFixed(i ? 1 : 0) + ' ' + u[i]
  },
  fmtAge(ms) {
    if (!ms && ms !== 0) return '-'
    if (ms < 1000) return ms + 'ms'
    const s = Math.floor(ms / 1000)
    if (s < 60) return s + 's'
    const m = Math.floor(s / 60)
    if (m < 60) return m + 'm ' + (s % 60) + 's'
    const h = Math.floor(m / 60)
    return h + 'h ' + (m % 60) + 'm'
  },
  fmtTimestamp(s) {
    if (s === undefined || s === null) return '0:00'
    const m = Math.floor(s / 60)
    const sec = Math.floor(s % 60)
    return m + ':' + (sec < 10 ? '0' : '') + sec
  },
  indexDetail(stats) {
    if (!stats) return '-'
    const parts = []
    const count = stats.Count ?? stats.vectors ?? stats.count
    const mem = stats.MemoryUsed ?? stats.memory_bytes ?? stats.memory_used
    const efSearch = stats.Extra?.ef_search ?? stats.ef_search
    const m = stats.Extra?.m ?? stats.m
    const levels = stats.levels ?? stats.Levels
    const clusters = stats.clusters ?? stats.Clusters

    if (count != null) parts.push(this.fmt(count) + ' vecs')
    if (mem) parts.push(this.fmtBytes(mem))
    if (efSearch) parts.push('ef=' + efSearch)
    if (m) parts.push('M=' + m)
    if (levels) parts.push(levels + ' levels')
    if (clusters) parts.push(clusters + ' clusters')
    return parts.join(', ') || '-'
  },

  escapeHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')
  },

  extractSnippet(text, query, windowSize = 300) {
    if (!text) return ''
    let clean = text.replace(/[\x00-\x08\x0E-\x1F\x7F-\x9F]/g, '')
    clean = clean.replace(/[\uFFFD\uFFFE\uFFFF]/g, '')
    clean = clean.replace(/[\u200B-\u200F\u2028-\u202F\uFEFF]/g, '')
    clean = clean.replace(/.*?This resource fork intentionally left blank\s*/s, '')
    clean = clean.replace(/com\.apple\.\w+/g, '')
    clean = clean.replace(/\s+/g, ' ').trim()
    if (!clean) return ''

    const tokens = (query || '').toLowerCase().split(/\s+/).filter(Boolean)
    if (!tokens.length) return clean.substring(0, windowSize) + (clean.length > windowSize ? '\u2026' : '')

    const lower = clean.toLowerCase()
    let bestPos = -1
    for (const tok of tokens) {
      const idx = lower.indexOf(tok)
      if (idx !== -1) { bestPos = idx; break }
    }

    if (bestPos === -1) {
      return clean.substring(0, windowSize) + (clean.length > windowSize ? '\u2026' : '')
    }

    let start = Math.max(0, bestPos - Math.floor(windowSize / 2))
    let end = Math.min(clean.length, start + windowSize)
    if (start > 0) {
      const sp = clean.indexOf(' ', start)
      if (sp !== -1 && sp < start + 30) start = sp + 1
    }
    if (end < clean.length) {
      const sp = clean.lastIndexOf(' ', end)
      if (sp > end - 30) end = sp
    }

    let snippet = clean.substring(start, end).trim()
    if (start > 0) snippet = '\u2026' + snippet
    if (end < clean.length) snippet = snippet + '\u2026'
    return snippet
  },

  highlightTerms(snippet, query) {
    if (!snippet || !query) return snippet || ''
    const tokens = query.split(/\s+/).filter(Boolean)
    if (!tokens.length) return this.escapeHtml(snippet)
    const escaped = tokens.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
    const re = new RegExp('(' + escaped.join('|') + ')', 'gi')
    const safe = this.escapeHtml(snippet)
    return safe.replace(re, '<mark class="search-highlight">$1</mark>')
  },
}
