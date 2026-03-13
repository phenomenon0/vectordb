// Explorer, collections, vault browser
export const dataMixin = {
  // ---- Vault Browser ----
  async browseVault() {
    const r = await this.api('GET', '/vault/browse?dir=' + encodeURIComponent(this.vaultBrowseDir))
    if (r.ok) {
      this.vaultFiles = r.data.files || []
    } else {
      this.vaultFiles = []
      if (r.status === 404) {
        this.toast('No vault configured. Enable Obsidian sync in Settings first.', 'error')
      }
    }
  },

  openVaultFile(f) {
    const synth = {
      id: f.path,
      text: '',
      document: '',
      metadata: { path: f.path },
      score: null,
    }
    this.dataSection = 'search'
    this.$nextTick(() => this.openViewer(synth))
  },

  // ---- Explorer ----
  async loadDocs() {
    const q = new URLSearchParams({
      collection: this.explorerCollection || '',
      limit: this.explorerLimit,
      offset: this.explorerOffset,
      include_meta: 'true',
    })
    const r = await this.api('GET', '/scroll?' + q.toString())
    if (r.ok) {
      const ids = r.data.ids || []
      const docs = r.data.documents || []
      const metas = r.data.metadata || []
      this.explorerTotal = r.data.total || 0
      this.explorerDocs = ids.map((id, i) => ({
        id,
        text: docs[i] || '',
        metadata: metas[i] || null,
        collection: this.explorerCollection,
        _expanded: false,
      }))
    }
  },

  async deleteDoc(id, collection) {
    const r = await this.api('POST', '/delete', { ids: [id], collection: collection || 'default' })
    if (r.ok) {
      this.toast('Deleted ' + id, 'success')
      this.loadDocs()
      this.loadHealth()
    } else {
      this.toast('Delete failed', 'error')
    }
  },

  // ---- Collections ----
  async loadV2Collections() {
    const r = await this.api('GET', '/v2/collections')
    if (r.ok) {
      this.v2Collections = (r.data.collections || []).map(c => ({ ...c, _confirmDelete: false }))
    }
  },
  async createCollection() {
    if (!this.newCollName) return
    let fields = []
    if (this.newCollFields) {
      try { fields = JSON.parse(this.newCollFields) } catch { this.toast('Invalid fields JSON', 'error'); return }
    }
    const r = await this.api('POST', '/v2/collections', { name: this.newCollName, fields })
    if (r.ok) {
      this.toast('Created ' + this.newCollName, 'success')
      this.newCollName = ''
      this.newCollFields = ''
      this.loadV2Collections()
      this.loadHealth()
    } else {
      this.toast('Create failed: ' + (r.data?.message || r.status), 'error')
    }
  },
  async deleteCollection(name) {
    const r = await this.api('DELETE', '/v2/collections/' + encodeURIComponent(name))
    if (r.ok) {
      this.toast('Deleted ' + name, 'success')
      this.loadV2Collections()
      this.loadHealth()
    } else {
      this.toast('Delete failed', 'error')
    }
  },
}
