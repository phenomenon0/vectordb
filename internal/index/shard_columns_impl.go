package index

// ColumnExportable implementations for HNSW, FLAT, and IVF indexes.
// These enable columnar Shard v2 persistence with selective field loading.

// --- HNSW ---

func (h *HNSWIndex) ExportConfig() map[string]interface{} {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return map[string]interface{}{
		"m":         h.m,
		"ml":        h.ml,
		"ef_search": h.efSearch,
	}
}

func (h *HNSWIndex) ExportVectors() []VectorEntry {
	h.mu.RLock()
	defer h.mu.RUnlock()
	entries := make([]VectorEntry, 0, len(h.vectors))
	for id, vec := range h.vectors {
		entries = append(entries, VectorEntry{ID: id, Vector: vec})
	}
	return entries
}

func (h *HNSWIndex) ExportDeleted() []uint64 {
	h.mu.RLock()
	defer h.mu.RUnlock()
	ids := make([]uint64, 0, len(h.deleted))
	for id := range h.deleted {
		ids = append(ids, id)
	}
	return ids
}

// --- FLAT ---

func (flat *FLATIndex) ExportConfig() map[string]interface{} {
	flat.mu.RLock()
	defer flat.mu.RUnlock()
	return map[string]interface{}{
		"metric": flat.metric,
	}
}

func (flat *FLATIndex) ExportVectors() []VectorEntry {
	flat.mu.RLock()
	defer flat.mu.RUnlock()
	entries := make([]VectorEntry, 0, len(flat.vectors))
	for id, vec := range flat.vectors {
		entries = append(entries, VectorEntry{ID: id, Vector: vec})
	}
	return entries
}

func (flat *FLATIndex) ExportDeleted() []uint64 {
	flat.mu.RLock()
	defer flat.mu.RUnlock()
	ids := make([]uint64, 0, len(flat.deleted))
	for id := range flat.deleted {
		ids = append(ids, id)
	}
	return ids
}

// --- IVF ---

func (ivf *IVFIndex) ExportConfig() map[string]interface{} {
	ivf.mu.RLock()
	defer ivf.mu.RUnlock()
	return map[string]interface{}{
		"nlist":  ivf.nlist,
		"nprobe": ivf.nprobe,
	}
}

func (ivf *IVFIndex) ExportVectors() []VectorEntry {
	ivf.mu.RLock()
	defer ivf.mu.RUnlock()
	entries := make([]VectorEntry, 0, len(ivf.vectors))
	for id, vec := range ivf.vectors {
		entries = append(entries, VectorEntry{ID: id, Vector: vec})
	}
	return entries
}

func (ivf *IVFIndex) ExportDeleted() []uint64 {
	ivf.mu.RLock()
	defer ivf.mu.RUnlock()
	ids := make([]uint64, 0, len(ivf.deleted))
	for id := range ivf.deleted {
		ids = append(ids, id)
	}
	return ids
}
