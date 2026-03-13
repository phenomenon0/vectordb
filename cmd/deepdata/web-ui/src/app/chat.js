// Chat: LLM provider, chat interface, Agent-GO
export const chatMixin = {
  // ---- LLM Provider ----
  getLlmProviderDefaults(provider) {
    const presets = {
      ollama:     { url: 'http://localhost:11434/v1', needsKey: false },
      atlasd:     { url: 'http://localhost:11435/v1', needsKey: false },
      openrouter: { url: 'https://openrouter.ai/api/v1', needsKey: true },
      deepseek:   { url: 'https://api.deepseek.com/v1', needsKey: true },
      cerebras:   { url: 'https://api.cerebras.ai/v1', needsKey: true },
      openai:     { url: 'https://api.openai.com/v1', needsKey: true },
      custom:     { url: '', needsKey: false },
    }
    return presets[provider] || presets.custom
  },

  onLlmProviderChange() {
    const d = this.getLlmProviderDefaults(this.llmProvider)
    this.llmUrl = d.url
    this.llmNeedsKey = d.needsKey
    if (!d.needsKey) this.llmApiKey = ''
    this.llmConnected = false
    this.llmModels = []
  },

  async connectLlm() {
    this.llmConnecting = true
    localStorage.setItem('llmProvider', this.llmProvider)
    localStorage.setItem('llmUrl', this.llmUrl)
    if (this.llmApiKey) localStorage.setItem('llmApiKey', this.llmApiKey)

    try {
      const headers = {}
      if (this.llmApiKey) headers['Authorization'] = 'Bearer ' + this.llmApiKey
      const ctrl = new AbortController()
      const timeoutId = setTimeout(() => ctrl.abort(), 8000)
      const r = await fetch(this.llmUrl + '/models', { headers, signal: ctrl.signal })
      clearTimeout(timeoutId)
      if (r.ok) {
        const data = await r.json()
        const models = (data.data || data.models || []).map(m => m.id || m.name || m.model || m)
        this.llmModels = models.filter(Boolean)
        if (this.llmModels.length && !this.llmModel) {
          this.llmModel = this.llmModels[0]
        }
        const saved = localStorage.getItem('llmModel')
        if (saved && this.llmModels.includes(saved)) this.llmModel = saved

        this.llmConnected = true
        this.chatSystemPromptOverride = this.buildSystemPrompt()
        this.chatMessages = [{ role: 'system', content: this.chatSystemPromptOverride }]
        this.toast('Connected \u2014 ' + this.llmModels.length + ' models', 'success')
      } else {
        this.llmConnected = false
        this.toast('Failed to reach ' + this.llmUrl + ' (' + r.status + ')', 'error')
      }
    } catch (e) {
      this.llmConnected = false
      const msg = e.name === 'AbortError' ? 'Connection timed out' : e.message
      this.toast('Connection failed: ' + msg, 'error')
    }
    this.llmConnecting = false
  },

  buildSystemPrompt() {
    const colls = (this.health.collections || []).map(c => c.name + ' (' + c.vector_count + ' vectors)').join(', ') || 'none'
    const focused = this.chatCollection
      ? `\nFocused collection: "${this.chatCollection}" \u2014 scope all queries/inserts to this collection unless asked otherwise.`
      : ''
    return `You are a helpful database assistant for DeepData, a vector database.

Instance: ${this.apiBase}
Mode: ${this.mode.mode || 'local'}
Embedder: ${this.mode.embedder_model || this.mode.embedder_type || 'unknown'}
Dimension: ${this.mode.dimension || '?'}
Collections: ${colls}
Vectors: ${this.health.active || 0}${focused}

Available API endpoints (all at ${this.apiBase}):
- GET /health \u2014 status, vector count, collections
- POST /query \u2014 vector search: {"text":"...","top_k":5,"collection":"${this.chatCollection || 'default'}"}
- POST /insert \u2014 add document: {"text":"...","collection":"${this.chatCollection || 'default'}","metadata":{}}
- GET /scroll?collection=${this.chatCollection || 'default'}&limit=50&offset=0 \u2014 browse documents
- POST /api/embed \u2014 get embedding: {"text":"..."}
- GET /integrity \u2014 integrity check
- POST /compact \u2014 compact storage
- POST /v2/search \u2014 hybrid/keyword: {"text":"...","mode":"hybrid","alpha":0.5,"collection":"${this.chatCollection || 'default'}"}

When the user asks about their data, reference the actual collections and counts above. Give concrete curl commands or API calls they can run. Be concise.`
  },

  applyChatSystem() {
    this.chatSystemPromptOverride = this.buildSystemPrompt()
    const sysIdx = this.chatMessages.findIndex(m => m.role === 'system')
    if (sysIdx >= 0) {
      this.chatMessages[sysIdx].content = this.chatSystemPromptOverride
    } else {
      this.chatMessages.unshift({ role: 'system', content: this.chatSystemPromptOverride })
    }
  },

  async sendChat() {
    if (!this.chatInput.trim() || this.chatStreaming) return
    const userMsg = this.chatInput.trim()
    this.chatInput = ''
    this.chatMessages.push({ role: 'user', content: userMsg })
    this.chatStreaming = true
    this.chatStreamContent = ''

    localStorage.setItem('llmModel', this.llmModel)

    const headers = { 'Content-Type': 'application/json' }
    if (this.llmApiKey) headers['Authorization'] = 'Bearer ' + this.llmApiKey

    this.chatAbortCtrl = new AbortController()

    try {
      const r = await fetch(this.llmUrl + '/chat/completions', {
        method: 'POST',
        headers,
        signal: this.chatAbortCtrl.signal,
        body: JSON.stringify({
          model: this.llmModel,
          messages: this.chatMessages,
          stream: true,
        }),
      })

      if (!r.ok) {
        const errText = await r.text()
        this.chatMessages.push({ role: 'assistant', content: 'Error: ' + r.status + ' \u2014 ' + errText })
        this.chatStreaming = false
        return
      }

      const reader = r.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })

        const lines = buf.split('\n')
        buf = lines.pop() || ''

        for (const line of lines) {
          const trimmed = line.trim()
          if (!trimmed || trimmed === 'data: [DONE]') continue
          if (!trimmed.startsWith('data: ')) continue
          try {
            const json = JSON.parse(trimmed.slice(6))
            const delta = json.choices?.[0]?.delta?.content
            if (delta) {
              this.chatStreamContent += delta
              this.$nextTick(() => {
                const el = this.$refs.chatScroll
                if (el) el.scrollTop = el.scrollHeight
              })
            }
          } catch {}
        }
      }

      if (this.chatStreamContent) {
        this.chatMessages.push({ role: 'assistant', content: this.chatStreamContent })
      }
    } catch (e) {
      if (e.name !== 'AbortError') {
        this.chatMessages.push({ role: 'assistant', content: 'Error: ' + e.message })
      } else if (this.chatStreamContent) {
        this.chatMessages.push({ role: 'assistant', content: this.chatStreamContent + '\n\n[stopped]' })
      }
    }

    this.chatStreamContent = ''
    this.chatStreaming = false
    this.chatAbortCtrl = null
    this.$nextTick(() => {
      const el = this.$refs.chatScroll
      if (el) el.scrollTop = el.scrollHeight
    })
  },

  abortChat() {
    if (this.chatAbortCtrl) this.chatAbortCtrl.abort()
  },

  // ---- Agent-GO ----
  async connectAgentGo() {
    this.agentGoConnecting = true
    localStorage.setItem('agentGoUrl', this.agentGoUrl)
    const r = await this.agentApi('GET', '/health')
    this.agentGoConnecting = false
    if (r.ok) {
      this.agentGoConnected = true
      this.agentGoHealth = r.data
      await this.loadAgentRuns()
      this.startRunPolling()
      this.toast('Agent-GO connected', 'success')
    } else {
      this.agentGoConnected = false
      this.toast('Cannot reach Agent-GO', 'error')
    }
  },

  async loadAgentRuns() {
    const r = await this.agentApi('GET', '/runs')
    if (r.ok) this.agentRuns = r.data.runs || []
  },

  startRunPolling() {
    this.stopRunPolling()
    this.runPollId = setInterval(() => { if (this.agentGoConnected) this.loadAgentRuns() }, 5000)
  },

  stopRunPolling() {
    if (this.runPollId) { clearInterval(this.runPollId); this.runPollId = null }
  },

  stopAgentPolling() {
    this.stopRunPolling()
  },

  async selectRun(runId) {
    const r = await this.agentApi('GET', '/runs/' + runId)
    if (r.ok) {
      this.toast('Run ' + runId.substring(0, 8) + ': ' + (r.data.status || '?') + ', ' + (r.data.event_count || 0) + ' events')
    }
  },

  runStatusBadge(status) {
    switch (status) {
      case 'running': return 'badge-green'
      case 'completed': case 'done': return 'badge-blue'
      case 'paused': case 'pause_requested': return 'badge-yellow'
      case 'failed': case 'error': return 'badge-red'
      default: return 'badge-purple'
    }
  },
}
