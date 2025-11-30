package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/coder/hnsw"
)

// ===========================================================================================
// COLLECTION MIGRATION SUPPORT
// Enables moving collections between shards for rebalancing
// ===========================================================================================

// CollectionExport represents exportable collection data
type CollectionExport struct {
	Collection string              `json:"collection"`
	Vectors    []VectorRecord      `json:"vectors"`
	Count      int                 `json:"count"`
	Dimension  int                 `json:"dimension"`
	ExportTime time.Time           `json:"export_time"`
	Checksum   string              `json:"checksum"`
}

// VectorRecord is a single exportable vector with all metadata
type VectorRecord struct {
	ID       string            `json:"id"`
	Vector   []float32         `json:"vector"`
	Doc      string            `json:"doc"`
	Meta     map[string]string `json:"meta,omitempty"`
	TenantID string            `json:"tenant_id,omitempty"`
	NumMeta  map[string]float64    `json:"num_meta,omitempty"`
	TimeMeta map[string]time.Time  `json:"time_meta,omitempty"`
}

// ExportCollection exports all vectors for a collection
func (vs *VectorStore) ExportCollection(collection string) (*CollectionExport, error) {
	vs.RLock()
	defer vs.RUnlock()

	export := &CollectionExport{
		Collection: collection,
		Vectors:    make([]VectorRecord, 0),
		Dimension:  vs.Dim,
		ExportTime: time.Now(),
	}

	for i := 0; i < vs.Count; i++ {
		id := vs.IDs[i]
		hid := hashID(id)

		// Skip deleted
		if vs.Deleted[hid] {
			continue
		}

		// Check collection
		coll := vs.Coll[hid]
		if coll == "" {
			coll = "default"
		}
		if coll != collection {
			continue
		}

		// Extract vector
		start := i * vs.Dim
		end := start + vs.Dim
		vec := make([]float32, vs.Dim)
		copy(vec, vs.Data[start:end])

		// Build record
		record := VectorRecord{
			ID:       id,
			Vector:   vec,
			Doc:      vs.Docs[i],
			TenantID: vs.TenantID[hid],
		}

		// Copy metadata
		if meta, ok := vs.Meta[hid]; ok && len(meta) > 0 {
			record.Meta = make(map[string]string, len(meta))
			for k, v := range meta {
				record.Meta[k] = v
			}
		}

		// Copy numeric metadata
		if numMeta, ok := vs.NumMeta[hid]; ok && len(numMeta) > 0 {
			record.NumMeta = make(map[string]float64, len(numMeta))
			for k, v := range numMeta {
				record.NumMeta[k] = v
			}
		}

		// Copy time metadata
		if timeMeta, ok := vs.TimeMeta[hid]; ok && len(timeMeta) > 0 {
			record.TimeMeta = make(map[string]time.Time, len(timeMeta))
			for k, v := range timeMeta {
				record.TimeMeta[k] = v
			}
		}

		export.Vectors = append(export.Vectors, record)
	}

	export.Count = len(export.Vectors)
	export.Checksum = fmt.Sprintf("coll-%s-n%d", collection, export.Count)

	return export, nil
}

// ImportCollection imports vectors for a collection
func (vs *VectorStore) ImportCollection(export *CollectionExport) error {
	if export.Dimension != vs.Dim {
		return fmt.Errorf("dimension mismatch: store=%d, import=%d", vs.Dim, export.Dimension)
	}

	vs.Lock()
	defer vs.Unlock()

	imported := 0
	for _, rec := range export.Vectors {
		// Validate vector dimension
		if len(rec.Vector) != vs.Dim {
			continue
		}

		tenantID := rec.TenantID
		if tenantID == "" {
			tenantID = "default"
		}

		// Add vector (without lock since we already hold it)
		id := rec.ID
		if id == "" {
			id = fmt.Sprintf("doc-%d", vs.next)
			vs.next++
		}

		hid := hashID(id)

		// Check if already exists
		if _, exists := vs.idToIx[hid]; exists {
			// Skip duplicates
			continue
		}

		// Store vector
		vs.Data = append(vs.Data, rec.Vector...)
		vs.Docs = append(vs.Docs, rec.Doc)
		vs.IDs = append(vs.IDs, id)
		vs.Seqs = append(vs.Seqs, uint64(vs.next))
		vs.next++

		idx := vs.Count
		vs.idToIx[hid] = idx
		vs.Count++

		// Store metadata
		vs.Meta[hid] = rec.Meta
		vs.Coll[hid] = export.Collection
		vs.TenantID[hid] = tenantID

		if len(rec.NumMeta) > 0 {
			vs.NumMeta[hid] = rec.NumMeta
		}
		if len(rec.TimeMeta) > 0 {
			vs.TimeMeta[hid] = rec.TimeMeta
		}

		// Add to HNSW
		vs.hnsw.Add(hnsw.MakeNode(hid, rec.Vector))

		// Update lexical index
		tokens := tokenize(rec.Doc)
		vs.lexTF[hid] = make(map[string]int)
		for _, t := range tokens {
			vs.lexTF[hid][t]++
		}
		vs.docLen[hid] = len(tokens)
		vs.sumDocL += len(tokens)
		for t := range vs.lexTF[hid] {
			vs.df[t]++
		}

		imported++
	}

	// Update checksum
	vs.checksum = vs.computeChecksum()

	return nil
}

// DeleteCollection removes all vectors for a collection
func (vs *VectorStore) DeleteCollection(collection string) (int, error) {
	vs.Lock()
	defer vs.Unlock()

	deleted := 0
	for i := 0; i < vs.Count; i++ {
		id := vs.IDs[i]
		hid := hashID(id)

		// Skip already deleted
		if vs.Deleted[hid] {
			continue
		}

		// Check collection
		coll := vs.Coll[hid]
		if coll == "" {
			coll = "default"
		}
		if coll != collection {
			continue
		}

		// Mark as deleted
		vs.Deleted[hid] = true
		deleted++
	}

	return deleted, nil
}

// CollectionStats returns statistics for a collection
func (vs *VectorStore) CollectionStats(collection string) map[string]any {
	vs.RLock()
	defer vs.RUnlock()

	total := 0
	deleted := 0
	tenants := make(map[string]int)

	for i := 0; i < vs.Count; i++ {
		id := vs.IDs[i]
		hid := hashID(id)

		// Check collection
		coll := vs.Coll[hid]
		if coll == "" {
			coll = "default"
		}
		if coll != collection {
			continue
		}

		total++
		if vs.Deleted[hid] {
			deleted++
		}

		tenant := vs.TenantID[hid]
		if tenant == "" {
			tenant = "default"
		}
		tenants[tenant]++
	}

	return map[string]any{
		"collection": collection,
		"total":      total,
		"deleted":    deleted,
		"active":     total - deleted,
		"tenants":    tenants,
	}
}

// ===========================================================================================
// HTTP MIGRATION CLIENT
// For calling migration endpoints on remote shards
// ===========================================================================================

// MigrationClient handles HTTP calls for data migration
type MigrationClient struct {
	httpClient *http.Client
}

// NewMigrationClient creates a new migration client
func NewMigrationClient() *MigrationClient {
	return &MigrationClient{
		httpClient: &http.Client{
			Timeout: 5 * time.Minute, // Long timeout for large migrations
		},
	}
}

// ExportFromShard exports a collection from a remote shard
func (mc *MigrationClient) ExportFromShard(shardAddr, collection string) (*CollectionExport, error) {
	url := fmt.Sprintf("%s/migration/export?collection=%s", shardAddr, collection)

	resp, err := mc.httpClient.Get(url)
	if err != nil {
		return nil, fmt.Errorf("export request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("export failed with status %d: %s", resp.StatusCode, string(body))
	}

	var export CollectionExport
	if err := json.NewDecoder(resp.Body).Decode(&export); err != nil {
		return nil, fmt.Errorf("failed to decode export: %w", err)
	}

	return &export, nil
}

// ImportToShard imports a collection to a remote shard
func (mc *MigrationClient) ImportToShard(shardAddr string, export *CollectionExport) error {
	body, err := json.Marshal(export)
	if err != nil {
		return fmt.Errorf("failed to marshal export: %w", err)
	}

	url := fmt.Sprintf("%s/migration/import", shardAddr)
	resp, err := mc.httpClient.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("import request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("import failed with status %d: %s", resp.StatusCode, string(respBody))
	}

	return nil
}

// DeleteFromShard deletes a collection from a remote shard
func (mc *MigrationClient) DeleteFromShard(shardAddr, collection string) error {
	url := fmt.Sprintf("%s/migration/delete?collection=%s", shardAddr, collection)

	req, err := http.NewRequest(http.MethodDelete, url, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := mc.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("delete request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("delete failed with status %d: %s", resp.StatusCode, string(body))
	}

	return nil
}

// GetCollectionStats gets statistics for a collection on a remote shard
func (mc *MigrationClient) GetCollectionStats(shardAddr, collection string) (map[string]any, error) {
	url := fmt.Sprintf("%s/migration/stats?collection=%s", shardAddr, collection)

	resp, err := mc.httpClient.Get(url)
	if err != nil {
		return nil, fmt.Errorf("stats request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("stats failed with status %d: %s", resp.StatusCode, string(body))
	}

	var stats map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&stats); err != nil {
		return nil, fmt.Errorf("failed to decode stats: %w", err)
	}

	return stats, nil
}

// ===========================================================================================
// HTTP HANDLERS FOR MIGRATION
// ===========================================================================================

// registerMigrationHandlers adds migration endpoints to HTTP mux
func registerMigrationHandlers(mux *http.ServeMux, store *VectorStore) {
	// Export collection
	mux.HandleFunc("/migration/export", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		collection := r.URL.Query().Get("collection")
		if collection == "" {
			http.Error(w, "collection parameter required", http.StatusBadRequest)
			return
		}

		export, err := store.ExportCollection(collection)
		if err != nil {
			http.Error(w, fmt.Sprintf("export failed: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(export)
	})

	// Import collection
	mux.HandleFunc("/migration/import", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Limit request size (100MB for large migrations)
		r.Body = http.MaxBytesReader(w, r.Body, 100*1024*1024)

		var export CollectionExport
		if err := json.NewDecoder(r.Body).Decode(&export); err != nil {
			http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
			return
		}

		if err := store.ImportCollection(&export); err != nil {
			http.Error(w, fmt.Sprintf("import failed: %v", err), http.StatusInternalServerError)
			return
		}

		json.NewEncoder(w).Encode(map[string]any{
			"ok":         true,
			"collection": export.Collection,
			"imported":   export.Count,
		})
	})

	// Delete collection
	mux.HandleFunc("/migration/delete", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodDelete {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		collection := r.URL.Query().Get("collection")
		if collection == "" {
			http.Error(w, "collection parameter required", http.StatusBadRequest)
			return
		}

		deleted, err := store.DeleteCollection(collection)
		if err != nil {
			http.Error(w, fmt.Sprintf("delete failed: %v", err), http.StatusInternalServerError)
			return
		}

		json.NewEncoder(w).Encode(map[string]any{
			"ok":         true,
			"collection": collection,
			"deleted":    deleted,
		})
	})

	// Collection statistics
	mux.HandleFunc("/migration/stats", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		collection := r.URL.Query().Get("collection")
		if collection == "" {
			http.Error(w, "collection parameter required", http.StatusBadRequest)
			return
		}

		stats := store.CollectionStats(collection)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(stats)
	})
}
