import Alpine from 'alpinejs'
import 'highlight.js/styles/monokai-sublime.css'
import './style.css'

import { initialState } from './app/state.js'
import { coreMixin } from './app/core.js'
import { dashboardMixin } from './app/dashboard.js'
import { searchMixin } from './app/search.js'
import { dataMixin } from './app/data.js'
import { settingsMixin } from './app/settings.js'
import { chatMixin } from './app/chat.js'
import { adminMixin } from './app/admin.js'
import { knowledgeGraphMixin } from './app/knowledge-graph.js'

// Compose the app from all mixins
function app() {
  return {
    ...initialState(),
    ...coreMixin,
    ...dashboardMixin,
    ...searchMixin,
    ...dataMixin,
    ...settingsMixin,
    ...chatMixin,
    ...adminMixin,
    ...knowledgeGraphMixin,

    async init() {
      this.startTime = Date.now()
      this.llmNeedsKey = this.getLlmProviderDefaults(this.llmProvider).needsKey
      await Promise.all([
        this.loadHealth(),
        this.loadMode(),
        this.loadIndexes(),
        this.loadDashCosts(),
        this.loadDashFeedback(),
        this.loadDashIntegrity(),
        this.loadEmbedderConfig(),
        this.loadApiKeys(),
        this.loadMetrics(),
      ])
      this.updateUptime()
      setInterval(() => this.loadHealth(), 15000)
      setInterval(() => this.updateUptime(), 1000)
      this.metricsRefreshId = setInterval(() => this.loadMetrics(), 10000)
    },
  }
}

// Register globally so Alpine x-data="app()" works
window.app = app

Alpine.start()
