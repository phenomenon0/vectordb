import { marked } from 'marked'
import hljs from 'highlight.js'

// Search, viewer, annotations, transcription
export const searchMixin = {
  // ---- Search ----
  async doSearch() {
    if (!this.searchQuery.trim()) return
    this.searching = true
    this.searchResults = []
    this.searchLatency = null

    let path = '/query'
    let body = {
      query: this.searchQuery,
      top_k: this.searchTopK,
      collection: this.searchCollection || '',
      include_meta: true,
    }

    if (this.searchMode === 'hybrid') {
      body.score_mode = 'hybrid'
      body.hybrid_alpha = this.searchAlpha
      body.hybrid_params = {
        strategy: this.hybridStrategy,
        weights: {
          dense: this.hybridWeightDense,
          sparse: this.hybridWeightSparse,
        },
      }
    } else if (this.searchMode === 'keyword') {
      body.score_mode = 'lexical'
      body.mode = 'lex'
    }

    let metaFilter = null
    if (this.searchMeta) {
      try { metaFilter = JSON.parse(this.searchMeta) } catch {}
    }

    const geoFilter = this.buildGeoFilter()
    if (geoFilter) {
      metaFilter = metaFilter ? { ...metaFilter, ...geoFilter } : geoFilter
    }

    if (metaFilter) {
      body.metadata = metaFilter
    }

    const r = await this.api('POST', path, body)
    this.searching = false
    if (r.ok) {
      this.searchResults = (r.data.results || []).map(x => ({ ...x, _expanded: false }))
      this.searchLatency = r.ms
    } else {
      this.toast('Search failed: ' + (r.data?.error || r.status), 'error')
    }
  },

  // ---- Geo Filter Builder ----
  buildGeoFilter() {
    if (!this.geoFilterEnabled) return null
    if (this.geoFilterMode === 'radius') {
      const lat = parseFloat(this.geoFilterLat)
      const lon = parseFloat(this.geoFilterLon)
      const dist = parseFloat(this.geoFilterDistanceKm)
      if (isNaN(lat) || isNaN(lon) || isNaN(dist)) return null
      return { $geo_radius: { center: { lat, lon }, distance_km: dist } }
    } else {
      // bbox mode
      try {
        const tl = JSON.parse(this.geoFilterTopLeft)
        const br = JSON.parse(this.geoFilterBottomRight)
        return { $geo_bbox: { top_left: tl, bottom_right: br } }
      } catch {
        return null
      }
    }
  },

  // ---- Recommend ----
  async doRecommend() {
    const posIds = this.recommendPositiveIds.split(',').map(s => s.trim()).filter(Boolean)
    if (!posIds.length) {
      this.toast('Enter at least one positive ID', 'error')
      return
    }
    this.recommending = true
    this.recommendResults = []

    const body = {
      positive_ids: posIds,
      negative_ids: this.recommendNegativeIds.split(',').map(s => s.trim()).filter(Boolean),
      negative_weight: this.recommendNegativeWeight,
      top_k: this.recommendTopK,
      collection: this.searchCollection || '',
    }

    const r = await this.api('POST', '/v2/recommend', body)
    this.recommending = false
    if (r.ok) {
      this.recommendResults = (r.data.results || []).map(x => ({ ...x, _expanded: false }))
    } else {
      this.toast('Recommend failed: ' + (r.data?.error || r.status), 'error')
    }
  },

  // ---- Discover ----
  addDiscoverPair() {
    this.discoverContextPairs.push({ positive: '', negative: '' })
  },

  removeDiscoverPair(idx) {
    this.discoverContextPairs.splice(idx, 1)
  },

  async doDiscover() {
    if (!this.discoverTargetId.trim()) {
      this.toast('Enter a target ID', 'error')
      return
    }
    this.discovering = true
    this.discoverResults = []

    const pairs = this.discoverContextPairs
      .filter(p => p.positive.trim() || p.negative.trim())
      .map(p => ({ positive: p.positive.trim(), negative: p.negative.trim() }))

    const body = {
      target_id: this.discoverTargetId.trim(),
      context_pairs: pairs,
      top_k: this.recommendTopK,
      collection: this.searchCollection || '',
    }

    const r = await this.api('POST', '/v2/discover', body)
    this.discovering = false
    if (r.ok) {
      this.discoverResults = (r.data.results || []).map(x => ({ ...x, _expanded: false }))
    } else {
      this.toast('Discover failed: ' + (r.data?.error || r.status), 'error')
    }
  },

  // ---- Viewer ----
  detectModality(path) {
    if (!path) return 'text'
    const ext = path.split('.').pop().toLowerCase()
    if (ext === 'md') return 'markdown'
    if (['png','jpg','jpeg','gif','webp','svg','bmp','ico'].includes(ext)) return 'image'
    if (['mp3','wav','ogg','m4a','flac','aac','wma'].includes(ext)) return 'audio'
    if (['mp4','webm','mov','avi','mkv'].includes(ext)) return 'video'
    return 'text'
  },

  async openViewer(r, index) {
    this.viewerResult = r
    this.viewerIndex = (index !== undefined) ? index : this.searchResults.indexOf(r)
    const path = r.metadata?.path || ''
    this.viewerModality = this.detectModality(path)
    this.viewerOpen = true
    this.viewerLoading = true
    this.viewerContent = null
    this.activeAnnotation = null
    this.imgReset()

    const docId = path || r.id || r.doc_id || ''
    await this.loadAnnotations(docId)

    if (path) {
      await this.loadViewerContent(path)
    } else {
      this.viewerModality = 'text'
      this.viewerContent = r.text || r.document || ''
      this.viewerLoading = false
    }
  },

  viewerNext() {
    if (this.viewerIndex < this.searchResults.length - 1) {
      this.openViewer(this.searchResults[this.viewerIndex + 1], this.viewerIndex + 1)
    }
  },

  viewerPrev() {
    if (this.viewerIndex > 0) {
      this.openViewer(this.searchResults[this.viewerIndex - 1], this.viewerIndex - 1)
    }
  },

  async loadViewerContent(path) {
    const modality = this.viewerModality
    try {
      if (modality === 'markdown' || modality === 'text') {
        const resp = await fetch(this.apiBase + '/vault/file?path=' + encodeURIComponent(path))
        if (!resp.ok) {
          this.viewerContent = this.viewerResult?.text || this.viewerResult?.document || 'Failed to load file'
          this.viewerModality = 'text'
        } else {
          let text = await resp.text()
          text = text.replace(/[\x00-\x08\x0E-\x1F\x7F-\x9F]/g, '')
          text = text.replace(/.*?This resource fork intentionally left blank\s*/s, '')
          text = text.replace(/com\.apple\.\w+/g, '')
          this.viewerContent = text.trim()
        }
      } else {
        this.viewerContent = this.apiBase + '/vault/file?path=' + encodeURIComponent(path)
      }
    } catch (e) {
      this.viewerContent = this.viewerResult?.text || this.viewerResult?.document || 'Error: ' + e.message
      this.viewerModality = 'text'
    }
    this.viewerLoading = false
  },

  closeViewer() {
    this.viewerOpen = false
    this.viewerResult = null
    this.viewerContent = null
    this.viewerModality = null
    this.viewerLoading = false
    this.viewerIndex = -1
    this.annotationMode = null
    this.activeAnnotation = null
    this.transcription = ''
    this.transcribing = false
  },

  renderMarkdown(text) {
    if (!text) return text || ''
    const processed = text.replace(/\[\[([^\]|]+)(?:\|([^\]]+))?\]\]/g, (_, target, display) => {
      return '<a class="wiki-link" data-wiki-target="' + target.replace(/"/g, '&quot;') + '">' + (display || target) + '</a>'
    })
    try {
      const renderer = new marked.Renderer()
      renderer.code = function(obj) {
        const code = typeof obj === 'string' ? obj : (obj.text || obj)
        const lang = typeof obj === 'string' ? arguments[1] : (obj.lang || '')
        if (lang && hljs.getLanguage(lang)) {
          const highlighted = hljs.highlight(code, { language: lang }).value
          return '<pre><code class="hljs language-' + lang + '">' + highlighted + '</code></pre>'
        }
        const escaped = String(code).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
        return '<pre><code>' + escaped + '</code></pre>'
      }
      const html = marked.parse(processed, { breaks: true, gfm: true, renderer })
      return html
    } catch (e) {
      return '<pre>' + text + '</pre>'
    }
  },

  handleViewerClick(e) {
    const link = e.target.closest('.wiki-link')
    if (link) {
      e.preventDefault()
      const target = link.dataset.wikiTarget
      if (target) {
        this.searchQuery = target
        this.doSearch()
      }
    }
  },

  // Image zoom/pan
  imgZoom(delta) {
    this.imgScale = Math.max(0.1, Math.min(10, this.imgScale + delta))
  },
  imgPan(dx, dy) {
    this.imgX += dx / this.imgScale
    this.imgY += dy / this.imgScale
  },
  imgReset() {
    this.imgScale = 1; this.imgX = 0; this.imgY = 0; this.imgDragging = false
  },

  // ---- Annotations ----
  async loadAnnotations(docId) {
    if (!docId) return
    const r = await this.api('GET', '/vault/annotations?doc_id=' + encodeURIComponent(docId))
    if (r.ok) {
      this.annotations[docId] = r.data.annotations || []
    }
  },

  async saveAnnotation(ann) {
    const r = await this.api('POST', '/vault/annotations', ann)
    if (r.ok) {
      const docId = ann.doc_id
      if (!this.annotations[docId]) this.annotations[docId] = []
      this.annotations[docId].push(ann)
      this.annotations = { ...this.annotations }
      this.annotationNote = ''
      this.toast('Annotation saved', 'success')
    } else {
      this.toast('Failed to save annotation', 'error')
    }
  },

  async deleteAnnotation(docId, annId) {
    const r = await this.api('DELETE', '/vault/annotations?doc_id=' + encodeURIComponent(docId) + '&id=' + encodeURIComponent(annId))
    if (r.ok) {
      this.annotations[docId] = (this.annotations[docId] || []).filter(a => a.id !== annId)
      this.annotations = { ...this.annotations }
      this.toast('Annotation deleted', 'success')
    }
  },

  startTextHighlight() {
    const sel = window.getSelection()
    if (!sel || sel.isCollapsed) {
      this.toast('Select text first', 'error')
      return
    }
    const text = sel.toString()
    const docId = this.viewerResult?.metadata?.path || this.viewerResult?.id || ''
    const ann = {
      id: crypto.randomUUID(),
      doc_id: docId,
      modality: this.viewerModality,
      created_at: new Date().toISOString(),
      note: this.annotationNote || '',
      color: this.annotationColor,
      highlighted_text: text.substring(0, 500),
      text_start: 0,
      text_end: text.length,
    }
    this.saveAnnotation(ann)
    sel.removeAllRanges()
  },

  startImagePin(e) {
    const container = e.currentTarget
    const rect = container.getBoundingClientRect()
    const x = (e.clientX - rect.left) / rect.width
    const y = (e.clientY - rect.top) / rect.height
    const docId = this.viewerResult?.metadata?.path || this.viewerResult?.id || ''
    const note = prompt('Pin note:')
    if (note === null) return
    const ann = {
      id: crypto.randomUUID(),
      doc_id: docId,
      modality: 'image',
      created_at: new Date().toISOString(),
      note: note,
      color: this.annotationColor,
      pin: { x, y },
    }
    this.saveAnnotation(ann)
    this.annotationMode = null
  },

  addTimestampAnnotation(type) {
    const player = type === 'audio' ? this.$refs.audioPlayer : this.$refs.videoPlayer
    if (!player) return
    const ts = player.currentTime
    const docId = this.viewerResult?.metadata?.path || this.viewerResult?.id || ''
    const note = prompt('Note for ' + this.fmtTimestamp(ts) + ':')
    if (note === null) return
    const ann = {
      id: crypto.randomUUID(),
      doc_id: docId,
      modality: type,
      created_at: new Date().toISOString(),
      note: note,
      color: this.annotationColor,
      timestamp: ts,
    }
    this.saveAnnotation(ann)
  },

  // ---- Transcription ----
  async transcribeMedia() {
    const path = this.viewerResult?.metadata?.path
    if (!path) { this.toast('No file path for transcription', 'error'); return }
    this.transcribing = true
    this.transcription = ''
    try {
      const resp = await fetch(this.apiBase + '/vault/transcribe?path=' + encodeURIComponent(path), { method: 'POST' })
      if (!resp.ok) {
        const err = await resp.text()
        this.toast('Transcription failed: ' + err, 'error')
        this.transcribing = false
        return
      }
      const data = await resp.json()
      this.transcription = data.text || ''
      if (data.segments && data.segments.length > 0) {
        this.toast('Transcribed ' + data.segments.length + ' segments', 'success')
      } else {
        this.toast('Transcription complete', 'success')
      }
    } catch (e) {
      this.toast('Transcription error: ' + e.message, 'error')
    }
    this.transcribing = false
  },
}
