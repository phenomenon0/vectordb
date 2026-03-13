// Settings: maintenance, costs, feedback, embedder, API keys, tools
export const settingsMixin = {
  // ---- Costs ----
  async loadCosts() {
    const [r1, r2] = await Promise.all([
      this.api('GET', '/api/costs'),
      this.api('GET', '/api/costs/daily?days=30'),
    ])
    if (r1.ok) this.costs = r1.data
    if (r2.ok) this.dailyCosts = r2.data.daily || r2.data || []
  },

  // ---- Feedback ----
  async loadFeedbackStats() {
    const r = await this.api('GET', '/v2/feedback/stats')
    if (r.ok) this.feedbackStats = r.data
  },
  async submitFeedback() {
    if (!this.fbInteractionId) return
    const r = await this.api('POST', '/v2/feedback', {
      interaction_id: this.fbInteractionId,
      type: this.fbType,
      rating: this.fbRating,
      text: this.fbText,
    })
    if (r.ok) {
      this.toast('Feedback submitted', 'success')
      this.fbInteractionId = ''
      this.fbText = ''
      this.loadFeedbackStats()
    } else {
      this.toast('Failed: ' + (r.data?.error || r.status), 'error')
    }
  },
  async recordInteraction() {
    if (!this.intQuery) return
    const r = await this.api('POST', '/v2/interaction', {
      query: this.intQuery,
      result_ids: this.intResults.split(',').map(s => s.trim()).filter(Boolean),
      collection: this.intCollection || 'default',
    })
    if (r.ok) {
      this.toast('Interaction recorded: ' + (r.data?.interaction_id || ''), 'success')
      this.intQuery = ''
      this.intResults = ''
      this.loadFeedbackStats()
    } else {
      this.toast('Failed', 'error')
    }
  },

  // ---- Maintenance ----
  async runIntegrity() {
    this.integrityRunning = true
    this.integrityResult = null
    const r = await this.api('GET', '/integrity')
    this.integrityRunning = false
    this.integrityResult = r.ok ? r.data : { ok: false, error: r.data?.error || 'request failed' }
  },
  async runCompact() {
    this.compactRunning = true
    this.compactResult = null
    const r = await this.api('POST', '/compact')
    this.compactRunning = false
    if (r.ok) {
      this.compactResult = 'Compacted successfully'
      this.loadHealth()
    } else {
      this.compactResult = 'Failed'
    }
  },
  async importSnapshot(event) {
    const file = event.target.files[0]
    if (!file) return
    this.importRunning = true
    const formData = new FormData()
    formData.append('file', file)
    try {
      const r = await fetch(this.apiBase + '/import', { method: 'POST', body: formData })
      if (r.ok) {
        this.toast('Import successful', 'success')
        this.loadHealth()
      } else {
        this.toast('Import failed', 'error')
      }
    } catch (e) {
      this.toast('Import error: ' + e.message, 'error')
    }
    this.importRunning = false
    event.target.value = ''
  },
  async createIndex() {
    let config = {}
    if (this.newIdxConfig) {
      try { config = JSON.parse(this.newIdxConfig) } catch { this.toast('Invalid config JSON', 'error'); return }
    }
    const r = await this.api('POST', '/api/index/create', {
      collection: this.newIdxCollection || 'default',
      index_type: this.newIdxType,
      config,
    })
    if (r.ok) {
      this.toast('Index created', 'success')
      this.loadIndexes()
    } else {
      this.toast('Failed: ' + (r.data?.error || r.status), 'error')
    }
  },

  // ---- Embedder Config ----
  async loadEmbedderConfig() {
    const r = await this.api('GET', '/api/config/embedder')
    if (r.ok) {
      this.embType = r.data.type || 'hash'
      this.embModel = r.data.model || ''
      this.embDim = r.data.dimension || 0
    }
  },
  async saveEmbedderConfig() {
    this.embSaving = true
    this.embTestResult = null
    const body = { type: this.embType, model: this.embModel }
    if (this.embType === 'ollama') body.url = this.embUrl || 'http://localhost:11434'
    if (this.embType === 'openai') body.key = this.embKey
    if (this.embType === 'hash') body.model = this.embModel || '384'
    const r = await this.api('POST', '/api/config/embedder', body)
    this.embSaving = false
    if (r.ok && r.data.ok) {
      this.embDim = r.data.dimension
      this.embModel = r.data.model
      this.embTestResult = { ok: true, msg: 'Switched to ' + r.data.type + ' (dim=' + r.data.dimension + ')' }
      if (r.data.warning) this.embTestResult.msg += ' \u2014 ' + r.data.warning
      this.loadMode()
    } else {
      this.embTestResult = { ok: false, msg: r.data?.error || 'Failed' }
    }
  },

  // ---- API Keys ----
  async loadApiKeys() {
    const r = await this.api('GET', '/api/config/keys')
    if (r.ok) this.apiKeysSet = r.data || {}
  },
  async saveApiKeys() {
    this.apiKeysSaving = true
    const body = {}
    for (const [k, v] of Object.entries(this.apiKeyInputs)) {
      if (v.trim()) body[k] = v.trim()
    }
    const r = await this.api('POST', '/api/config/keys', body)
    this.apiKeysSaving = false
    if (r.ok && r.data.ok) {
      this.toast('Keys saved: ' + (r.data.set?.join(', ') || 'none'), 'success')
      this.apiKeyInputs = { openai: '', deepseek: '', anthropic: '', openrouter: '', cerebras: '' }
      this.loadApiKeys()
    } else {
      this.toast('Failed to save keys', 'error')
    }
  },

  // ---- Tools ----
  async runEmbed() {
    if (!this.embedText.trim()) return
    this.embedRunning = true
    this.embedResult = ''
    const r = await this.api('POST', '/api/embed', { text: this.embedText })
    this.embedRunning = false
    if (r.ok) {
      const vec = r.data.vector || r.data.embedding || []
      this.embedResult = `dim: ${vec.length}\n[${vec.slice(0, 20).map(v => v.toFixed(6)).join(', ')}${vec.length > 20 ? ', ...' : ''}]`
    } else {
      this.embedResult = 'Error: ' + (r.data?.error || r.status)
    }
  },
  async runExtract() {
    if (!this.extractText.trim()) return
    this.extractRunning = true
    this.extractResult = ''
    const body = { content: this.extractText }
    if (this.extractTypes.trim()) {
      body.entity_types = this.extractTypes.split(',').map(s => s.trim()).filter(Boolean)
    }
    const r = await this.api('POST', '/v2/extract', body)
    this.extractRunning = false
    if (r.ok) {
      this.extractResult = JSON.stringify(r.data, null, 2)
    } else {
      this.extractResult = 'Error: ' + (r.data || r.status)
    }
  },
  async batchInsert() {
    if (!this.batchText.trim()) return
    const doc = {
      text: this.batchText,
      collection: this.batchCollection || 'default',
    }
    if (this.batchId) doc.id = this.batchId
    if (this.batchMeta) {
      try { doc.metadata = JSON.parse(this.batchMeta) } catch {}
    }
    const r = await this.api('POST', '/insert', doc)
    if (r.ok) {
      this.toast('Inserted', 'success')
      this.batchText = ''
      this.batchId = ''
      this.batchMeta = ''
      this.loadHealth()
    } else {
      this.toast('Insert failed: ' + (r.data?.error || r.status), 'error')
    }
  },
}
