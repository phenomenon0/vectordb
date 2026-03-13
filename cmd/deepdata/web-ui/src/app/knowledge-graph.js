// Knowledge Graph extraction mixin
export const knowledgeGraphMixin = {
  async extractKnowledge() {
    if (!this.kgInputText.trim()) {
      this.toast('Enter text to extract knowledge from', 'error')
      return
    }
    this.kgExtracting = true
    this.kgEntities = []
    this.kgRelationships = []
    this.kgTemporalEvents = []
    this.kgJobId = null

    try {
      if (this.kgBatchMode) {
        await this.extractBatch()
      } else if (this.kgTemporalMode) {
        await this.extractTemporal()
      } else {
        await this.extractSingle()
      }
    } finally {
      this.kgExtracting = false
    }
  },

  async extractSingle() {
    const r = await this.api('POST', '/v2/extract', {
      content: this.kgInputText,
      entity_types: this.kgEntityTypes,
    })
    if (r.ok) {
      this.kgEntities = r.data.entities || []
      this.kgRelationships = r.data.relationships || []
      this.toast('Extracted ' + this.kgEntities.length + ' entities, ' + this.kgRelationships.length + ' relationships', 'success')
    } else {
      this.toast('Extraction failed: ' + (r.data?.message || r.status), 'error')
    }
  },

  async extractBatch() {
    const contents = this.kgInputText.split('\n---\n').filter(c => c.trim())
    if (!contents.length) {
      this.toast('No content segments found (split by ---)', 'error')
      return
    }
    const r = await this.api('POST', '/v2/extract/batch', {
      contents,
      entity_types: this.kgEntityTypes,
    })
    if (r.ok) {
      if (r.data.job_id) {
        this.kgJobId = r.data.job_id
        this.toast('Batch job started: ' + r.data.job_id, 'success')
        this.pollExtractStatus(r.data.job_id)
      } else {
        this.kgEntities = r.data.entities || []
        this.kgRelationships = r.data.relationships || []
        this.toast('Batch extracted ' + this.kgEntities.length + ' entities', 'success')
      }
    } else {
      this.toast('Batch extraction failed: ' + (r.data?.message || r.status), 'error')
    }
  },

  async extractTemporal() {
    const r = await this.api('POST', '/v2/extract/temporal', {
      content: this.kgInputText,
    })
    if (r.ok) {
      this.kgEntities = r.data.entities || []
      this.kgRelationships = r.data.relationships || []
      this.kgTemporalEvents = (r.data.temporal_events || []).sort((a, b) => {
        return (a.date || '').localeCompare(b.date || '')
      })
      this.toast('Temporal extraction: ' + this.kgEntities.length + ' entities, ' + this.kgTemporalEvents.length + ' events', 'success')
    } else {
      this.toast('Temporal extraction failed: ' + (r.data?.message || r.status), 'error')
    }
  },

  async pollExtractStatus(jobId) {
    const poll = async () => {
      const r = await this.api('GET', '/v2/extract/status?job_id=' + encodeURIComponent(jobId))
      if (!r.ok) {
        this.toast('Status check failed', 'error')
        this.kgJobId = null
        return
      }
      if (r.data.status === 'completed') {
        this.kgEntities = r.data.entities || []
        this.kgRelationships = r.data.relationships || []
        this.kgTemporalEvents = (r.data.temporal_events || []).sort((a, b) => {
          return (a.date || '').localeCompare(b.date || '')
        })
        this.kgJobId = null
        this.kgExtracting = false
        this.toast('Extraction complete: ' + this.kgEntities.length + ' entities', 'success')
      } else if (r.data.status === 'failed') {
        this.kgJobId = null
        this.kgExtracting = false
        this.toast('Extraction job failed: ' + (r.data.error || 'unknown'), 'error')
      } else {
        setTimeout(poll, 2000)
      }
    }
    setTimeout(poll, 2000)
  },

  kgEntityTypeColor(type) {
    const colors = {
      person: '#3b82f6',
      organization: '#8b5cf6',
      location: '#10b981',
      concept: '#f59e0b',
      event: '#ef4444',
      product: '#06b6d4',
      technology: '#ec4899',
    }
    return colors[type] || '#6b7280'
  },

  toggleKgEntityType(type) {
    const idx = this.kgEntityTypes.indexOf(type)
    if (idx >= 0) {
      this.kgEntityTypes.splice(idx, 1)
    } else {
      this.kgEntityTypes.push(type)
    }
  },
}
