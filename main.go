package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
	"unicode"

	"agentscope/core"
	"agentscope/tools"
	"agentscope/vectordb/index"
	"agentscope/vectordb/logging"
	"agentscope/vectordb/storage"
	"agentscope/vectordb/telemetry"

	"github.com/coder/hnsw"
)

// ======================================================================================
// Vector Store (HNSW + metadata + persistence)
// ======================================================================================

type VectorStore struct {
	sync.RWMutex
	Data  []float32
	Dim   int
	Count int
	Docs  []string
	IDs   []string
	Seqs  []uint64
	next  int64
	// Index abstraction (NEW)
	indexes map[string]index.Index // Collection -> Index mapping
	// Legacy HNSW graph (deprecated, kept for backward compatibility)
	hnsw        *hnsw.Graph[uint64]
	idToIx      map[uint64]int
	Meta        map[uint64]map[string]string
	Deleted     map[uint64]bool
	Coll        map[uint64]string
	NumMeta     map[uint64]map[string]float64
	TimeMeta    map[uint64]map[string]time.Time
	numIndex    map[string][]numEntry
	timeIndex   map[string][]timeEntry
	walPath     string
	walMu       sync.Mutex
	walMaxBytes int64
	walMaxOps   int
	walOps      int
	walRotate   int64
	walHook     func(walEntry) // Optional hook to forward WAL events (e.g., to replication stream)
	apiToken    string
	rl          *rateLimiter
	checksum    string
	lastSaved   time.Time
	// Lexical stats for hybrid/BM25
	lexTF   map[uint64]map[string]int
	docLen  map[uint64]int
	df      map[string]int
	sumDocL int
	// Multi-tenancy support
	TenantID    map[uint64]string  // vector hash -> tenant ID
	acl         *ACL               // access control lists
	quotas      *TenantQuota       // storage quotas per tenant
	tenantRL    *tenantRateLimiter // per-tenant rate limiting
	jwtMgr      *JWTManager        // JWT token manager
	requireAuth bool               // Require JWT authentication
	// Storage format (gob, sjson, sjson-zstd)
	storageFormat storage.Format
	// Metadata bitmap index for fast pre-filtering
	// Metadata bitmap index for fast pre-filtering
	metaIndex *MetadataIndex

	// Replication
	replicationMgr *ReplicationManager
	replicas       []*ReplicaNode
}

func NewVectorStore(capacity int, dim int) *VectorStore {
	cfg := loadHNSWConfig()

	// Create default HNSW index using index abstraction
	defaultIdx, err := index.NewHNSWIndex(dim, map[string]interface{}{
		"m":         cfg.M,
		"ml":        cfg.Ml,
		"ef_search": cfg.EfSearch,
	})
	if err != nil {
		panic(fmt.Sprintf("failed to create default HNSW index: %v", err))
	}

	// Legacy HNSW graph (kept for backward compatibility during migration)
	g := hnsw.NewGraph[uint64]()
	g.Distance = hnsw.CosineDistance
	g.M = cfg.M
	g.Ml = cfg.Ml
	g.EfSearch = cfg.EfSearch

	// Initialize JWT manager if configured
	var jwtMgr *JWTManager
	if secret := os.Getenv("JWT_SECRET"); secret != "" {
		issuer := os.Getenv("JWT_ISSUER")
		if issuer == "" {
			issuer = "vectordb"
		}
		jwtMgr = NewJWTManager(secret, issuer)
	}

	// Select storage format (default: gob for backward compatibility)
	// Options: "gob", "sjson", "sjson-zstd"
	storageFormat := storage.Default()
	if formatName := os.Getenv("STORAGE_FORMAT"); formatName != "" {
		if f := storage.Get(formatName); f != nil {
			storageFormat = f
		}
	}

	apiToken := os.Getenv("API_TOKEN")
	requireAuth := os.Getenv("REQUIRE_AUTH") == "1" || jwtMgr != nil || apiToken != ""

	return &VectorStore{
		Data:        make([]float32, 0, capacity*dim),
		Dim:         dim,
		Count:       0,
		indexes:     map[string]index.Index{"default": defaultIdx},
		hnsw:        g, // Legacy, deprecated
		idToIx:      make(map[uint64]int),
		Meta:        make(map[uint64]map[string]string),
		Deleted:     make(map[uint64]bool),
		Coll:        make(map[uint64]string),
		Seqs:        make([]uint64, 0, capacity),
		NumMeta:     make(map[uint64]map[string]float64),
		TimeMeta:    make(map[uint64]map[string]time.Time),
		numIndex:    make(map[string][]numEntry),
		timeIndex:   make(map[string][]timeEntry),
		walMaxBytes: 0,
		walMaxOps:   0,
		apiToken:    apiToken,
		requireAuth: requireAuth,
		lexTF:       make(map[uint64]map[string]int),
		docLen:      make(map[uint64]int),
		df:          make(map[string]int),
		sumDocL:     0,
		// Multi-tenancy
		TenantID: make(map[uint64]string),
		acl:      NewACL(),
		quotas:   NewTenantQuota(),
		tenantRL: newTenantRateLimiter(envInt("TENANT_RPS", 100), envInt("TENANT_BURST", 100), time.Minute),
		jwtMgr:   jwtMgr,
		// Storage
		storageFormat: storageFormat,
		// Metadata index for fast pre-filtering
		metaIndex: NewMetadataIndex(),
	}
}

// Add appends a vector/doc/meta with tenant ownership.
func (vs *VectorStore) Add(v []float32, doc string, id string, meta map[string]string, collection string, tenantID string) (string, error) {
	vs.Lock()
	defer vs.Unlock()
	if len(v) != vs.Dim {
		return "", fmt.Errorf("dimension mismatch: expected %d, got %d", vs.Dim, len(v))
	}
	if id == "" {
		id = fmt.Sprintf("doc-%d", vs.next)
		vs.next++
	}
	if tenantID == "" {
		tenantID = "default" // Default tenant for backward compatibility
	}

	// Check quota before adding
	vectorBytes := int64(len(v) * 4) // 4 bytes per float32
	docBytes := int64(len(doc))
	totalBytes := vectorBytes + docBytes
	if err := vs.quotas.AddUsage(tenantID, totalBytes, 1); err != nil {
		return "", fmt.Errorf("quota check failed: %w", err)
	}

	vs.Data = append(vs.Data, v...)
	vs.Docs = append(vs.Docs, doc)
	vs.IDs = append(vs.IDs, id)
	vs.Seqs = append(vs.Seqs, uint64(vs.Count))
	vs.Count++

	toks := tokenize(doc)
	vs.ingestLex(hashID(id), toks)

	hid := hashID(id)

	// Use index abstraction (NEW)
	if collection == "" {
		collection = "default"
	}
	idx, ok := vs.indexes[collection]
	if !ok {
		idx = vs.indexes["default"] // Fall back to default index
	}
	if idx != nil {
		if err := idx.Add(context.Background(), hid, v); err != nil {
			// Rollback quota on index add failure
			vs.quotas.RemoveUsage(tenantID, vectorBytes+docBytes, 1)
			return "", fmt.Errorf("failed to add vector to index: %w", err)
		}
	}

	// Legacy HNSW (deprecated - keep for backward compatibility during migration)
	vs.hnsw.Add(hnsw.MakeNode(hid, v))
	vs.idToIx[hid] = vs.Count - 1
	if meta != nil {
		vs.Meta[hid] = meta
		vs.ingestMeta(hid, meta)
	}
	if collection == "" {
		collection = "default"
	}
	vs.Coll[hid] = collection
	vs.TenantID[hid] = tenantID // Store tenant ownership
	delete(vs.Deleted, hid)
	if err := vs.appendWAL("insert", id, doc, meta, v, tenantID); err != nil {
		// Rollback quota on WAL failure
		vs.quotas.RemoveUsage(tenantID, totalBytes, 1)
		return "", fmt.Errorf("failed to append to WAL: %w", err)
	}
	return id, nil
}

// Upsert replaces existing vector/doc/meta if ID exists; otherwise adds new with tenant ownership.
func (vs *VectorStore) Upsert(v []float32, doc string, id string, meta map[string]string, collection string, tenantID string) (string, error) {
	if id == "" {
		return vs.Add(v, doc, id, meta, collection, tenantID)
	}
	vs.Lock()
	defer vs.Unlock()
	if len(v) != vs.Dim {
		return "", fmt.Errorf("dimension mismatch: expected %d, got %d", vs.Dim, len(v))
	}
	if tenantID == "" {
		tenantID = "default"
	}

	hid := hashID(id)
	if ix, ok := vs.idToIx[hid]; ok && ix >= 0 && ix < len(vs.IDs) {
		// Verify tenant ownership for updates
		if existingTenant := vs.TenantID[hid]; existingTenant != "" && existingTenant != tenantID {
			return "", fmt.Errorf("access denied: vector belongs to different tenant")
		}

		vs.ejectLex(hid)
		vs.ejectMeta(hid)
		copy(vs.Data[ix*vs.Dim:(ix+1)*vs.Dim], v)
		vs.Docs[ix] = doc
		vs.ingestLex(hid, tokenize(doc))
		if meta != nil {
			vs.Meta[hid] = meta
			vs.ingestMeta(hid, meta)
		} else {
			delete(vs.Meta, hid)
			vs.ejectMeta(hid)
		}
		if collection != "" {
			vs.Coll[hid] = collection
		}
		delete(vs.Deleted, hid)
		if err := vs.appendWAL("upsert", id, doc, meta, v, tenantID); err != nil {
			return "", fmt.Errorf("failed to append to WAL: %w", err)
		}
		return id, nil
	}

	// Adding new vector - check quota
	vectorBytes := int64(len(v) * 4)
	docBytes := int64(len(doc))
	totalBytes := vectorBytes + docBytes
	if err := vs.quotas.AddUsage(tenantID, totalBytes, 1); err != nil {
		return "", fmt.Errorf("quota check failed: %w", err)
	}

	vs.Data = append(vs.Data, v...)
	vs.Docs = append(vs.Docs, doc)
	vs.IDs = append(vs.IDs, id)
	vs.Seqs = append(vs.Seqs, uint64(vs.Count))
	vs.Count++

	// Use index abstraction (NEW)
	if collection == "" {
		collection = "default"
	}
	idx, ok := vs.indexes[collection]
	if !ok {
		idx = vs.indexes["default"]
	}
	if idx != nil {
		if err := idx.Add(context.Background(), hid, v); err != nil {
			vs.quotas.RemoveUsage(tenantID, vectorBytes+docBytes, 1)
			return "", fmt.Errorf("failed to add vector to index: %w", err)
		}
	}

	// Legacy HNSW (deprecated)
	vs.hnsw.Add(hnsw.MakeNode(hid, v))
	vs.idToIx[hid] = vs.Count - 1
	vs.ingestLex(hid, tokenize(doc))
	if meta != nil {
		vs.Meta[hid] = meta
		vs.ingestMeta(hid, meta)
	}
	if collection == "" {
		collection = "default"
	}
	vs.Coll[hid] = collection
	vs.TenantID[hid] = tenantID // Store tenant ownership
	delete(vs.Deleted, hid)
	if err := vs.appendWAL("insert", id, doc, meta, v, tenantID); err != nil {
		// Rollback quota on WAL failure
		vs.quotas.RemoveUsage(tenantID, totalBytes, 1)
		return "", fmt.Errorf("failed to append to WAL: %w", err)
	}
	return id, nil
}

func (vs *VectorStore) Delete(id string) error {
	vs.Lock()
	defer vs.Unlock()
	hid := hashID(id)
	tenant := vs.TenantID[hid]
	if tenant == "" {
		tenant = "default"
	}
	vs.Deleted[hid] = true
	vs.ejectLex(hid)
	vs.ejectMeta(hid)
	delete(vs.Meta, hid)
	delete(vs.Coll, hid)
	if err := vs.appendWAL("delete", id, "", nil, nil, tenant); err != nil {
		return fmt.Errorf("failed to append to WAL: %w", err)
	}
	return nil
}

func (vs *VectorStore) Get(index int) []float32 {
	offset := index * vs.Dim
	return vs.Data[offset : offset+vs.Dim]
}

func (vs *VectorStore) GetDoc(index int) string {
	if index < 0 || index >= len(vs.Docs) {
		return ""
	}
	return vs.Docs[index]
}

func (vs *VectorStore) GetID(index int) string {
	if index < 0 || index >= len(vs.IDs) {
		return ""
	}
	return vs.IDs[index]
}

// Brute-force scan (used for debugging).
func (vs *VectorStore) Search(query []float32, k int) []int {
	vs.RLock()
	defer vs.RUnlock()
	if k <= 0 || vs.Count == 0 {
		return nil
	}
	bestIDs := make([]int, 0, k)
	bestScores := make([]float32, 0, k)
	for i := 0; i < vs.Count; i++ {
		hid := hashID(vs.IDs[i])
		if vs.Deleted[hid] {
			continue
		}
		vec := vs.Data[i*vs.Dim : (i+1)*vs.Dim]
		score := DotProduct(query, vec)
		if len(bestIDs) < k {
			bestIDs = append(bestIDs, i)
			bestScores = append(bestScores, score)
			continue
		}
		minIdx := 0
		for j := 1; j < k; j++ {
			if bestScores[j] < bestScores[minIdx] {
				minIdx = j
			}
		}
		if score > bestScores[minIdx] {
			bestScores[minIdx] = score
			bestIDs[minIdx] = i
		}
	}
	return bestIDs
}

// ANN search via HNSW.
func (vs *VectorStore) SearchANN(query []float32, k int) []int {
	vs.RLock()
	defer vs.RUnlock()

	// Use index abstraction (NEW)
	idx := vs.indexes["default"]
	if idx != nil {
		results, err := idx.Search(context.Background(), query, k, index.DefaultSearchParams{})
		if err == nil && len(results) > 0 {
			ixs := make([]int, 0, len(results))
			for _, r := range results {
				if vs.Deleted[r.ID] {
					continue
				}
				if ix, ok := vs.idToIx[r.ID]; ok {
					ixs = append(ixs, ix)
				}
			}
			return ixs
		}
	}

	// Fall back to legacy HNSW if index abstraction fails
	nodes := vs.hnsw.Search(query, k)
	ixs := make([]int, 0, len(nodes))
	for _, n := range nodes {
		if vs.Deleted[n.Key] {
			continue
		}
		if ix, ok := vs.idToIx[n.Key]; ok {
			ixs = append(ixs, ix)
		}
	}
	return ixs
}

// Lexical-only search using BM25 over all active docs.
func (vs *VectorStore) SearchLex(qTokens []string, k int) []int {
	vs.RLock()
	defer vs.RUnlock()
	type scored struct {
		ix    int
		score float64
	}
	best := make([]scored, 0, k)
	for i, id := range vs.IDs {
		hid := hashID(id)
		if vs.Deleted[hid] {
			continue
		}
		score := vs.bm25(hid, qTokens)
		if len(best) < k {
			best = append(best, scored{ix: i, score: score})
			continue
		}
		minIdx := 0
		for j := 1; j < len(best); j++ {
			if best[j].score < best[minIdx].score {
				minIdx = j
			}
		}
		if score > best[minIdx].score {
			best[minIdx] = scored{ix: i, score: score}
		}
	}
	sort.Slice(best, func(i, j int) bool { return best[i].score > best[j].score })
	ixs := make([]int, 0, len(best))
	for _, b := range best {
		ixs = append(ixs, b.ix)
	}
	return ixs
}

// GetPreFilteredIDs returns document IDs matching the metadata filter.
// Use this to get candidates before searching.
func (vs *VectorStore) GetPreFilteredIDs(filter map[string]string) []uint64 {
	if vs.metaIndex == nil || len(filter) == 0 {
		return nil
	}
	return vs.metaIndex.GetMatchingDocs(filter).ToSlice()
}

// GetMetadataIndexStats returns statistics about the metadata index.
func (vs *VectorStore) GetMetadataIndexStats() map[string]any {
	if vs.metaIndex == nil {
		return nil
	}
	return vs.metaIndex.GetStats()
}

// AnalyzeMetadataFilter analyzes a metadata filter and returns optimization info.
func (vs *VectorStore) AnalyzeMetadataFilter(filter map[string]string) map[string]any {
	if vs.metaIndex == nil {
		return nil
	}
	return vs.metaIndex.AnalyzeFilter(filter)
}

// Hybrid score combines cosine and BM25-like lexical score.
func (vs *VectorStore) hybridScore(hid uint64, qVec []float32, qTokens []string, alpha float64) float64 {
	vecScore := float64(0)
	if ix, ok := vs.idToIx[hid]; ok {
		dVec := vs.Data[ix*vs.Dim : (ix+1)*vs.Dim]
		vecScore = float64(DotProduct(qVec, dVec))
	}
	bm := vs.bm25(hid, qTokens)
	return alpha*vecScore + (1-alpha)*bm
}

func (vs *VectorStore) ingestLex(hid uint64, toks []string) {
	if len(toks) == 0 {
		return
	}
	tf := make(map[string]int)
	seen := make(map[string]bool)
	for _, t := range toks {
		if t == "" {
			continue
		}
		tf[t]++
		if !seen[t] {
			vs.df[t]++
			seen[t] = true
		}
	}
	prevLen := vs.docLen[hid]
	vs.sumDocL += len(toks) - prevLen
	vs.docLen[hid] = len(toks)
	vs.lexTF[hid] = tf
}

func (vs *VectorStore) ejectLex(hid uint64) {
	tf, ok := vs.lexTF[hid]
	if !ok {
		return
	}
	for term := range tf {
		if vs.df[term] > 0 {
			vs.df[term]--
		}
	}
	vs.sumDocL -= vs.docLen[hid]
	delete(vs.lexTF, hid)
	delete(vs.docLen, hid)
}

func (vs *VectorStore) bm25(hid uint64, qTokens []string) float64 {
	activeDocs := vs.Count - len(vs.Deleted)
	if activeDocs <= 0 {
		return 0
	}
	tf := vs.lexTF[hid]
	if tf == nil {
		return 0
	}
	avgdl := float64(vs.sumDocL)
	if avgdl == 0 {
		avgdl = 1
	} else {
		avgdl = avgdl / float64(activeDocs)
	}
	// BM25 parameters
	k1 := 1.2
	b := 0.75
	score := 0.0
	seen := make(map[string]bool)
	for _, qt := range qTokens {
		if seen[qt] {
			continue
		}
		seen[qt] = true
		df := vs.df[qt]
		if df == 0 {
			continue
		}
		idf := math.Log((float64(activeDocs)-float64(df)+0.5)/(float64(df)+0.5) + 1)
		tfDoc := float64(tf[qt])
		if tfDoc == 0 {
			continue
		}
		num := tfDoc * (k1 + 1)
		den := tfDoc + k1*(1-b+b*float64(vs.docLen[hid])/avgdl)
		score += idf * (num / den)
	}
	return score
}

func (vs *VectorStore) ingestMeta(hid uint64, meta map[string]string) {
	if meta == nil {
		return
	}
	nums := make(map[string]float64)
	times := make(map[string]time.Time)
	stringMeta := make(map[string]string)
	for k, v := range meta {
		if v == "" {
			continue
		}
		if t, err := time.Parse(time.RFC3339, v); err == nil {
			times[k] = t
			vs.timeIndex[k] = append(vs.timeIndex[k], timeEntry{ID: hid, T: t})
			continue
		}
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			nums[k] = f
			vs.numIndex[k] = append(vs.numIndex[k], numEntry{ID: hid, V: f})
			continue
		}
		// String metadata - add to bitmap index
		stringMeta[k] = v
	}
	if len(nums) > 0 {
		vs.NumMeta[hid] = nums
	}
	if len(times) > 0 {
		vs.TimeMeta[hid] = times
	}
	// Add to metadata bitmap index for fast pre-filtering
	if vs.metaIndex != nil && len(stringMeta) > 0 {
		vs.metaIndex.AddDocument(hid, stringMeta)
	}
}

func (vs *VectorStore) ejectMeta(hid uint64) {
	delete(vs.NumMeta, hid)
	delete(vs.TimeMeta, hid)
	for k, entries := range vs.numIndex {
		filtered := entries[:0]
		for _, e := range entries {
			if e.ID != hid {
				filtered = append(filtered, e)
			}
		}
		vs.numIndex[k] = filtered
	}
	for k, entries := range vs.timeIndex {
		filtered := entries[:0]
		for _, e := range entries {
			if e.ID != hid {
				filtered = append(filtered, e)
			}
		}
		vs.timeIndex[k] = filtered
	}
	// Remove from metadata bitmap index
	if vs.metaIndex != nil {
		vs.metaIndex.RemoveDocument(hid)
	}
}

// Persistence snapshot.
func (vs *VectorStore) Save(path string) error {
	vs.RLock()
	defer vs.RUnlock()

	tmp := path + ".tmp"
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	defer f.Close()

	// Export legacy HNSW (deprecated - for backward compatibility)
	var hBuf []byte
	if vs.hnsw != nil {
		var buf bytes.Buffer
		if err := vs.hnsw.Export(&buf); err == nil {
			hBuf = buf.Bytes()
		}
	}

	// Export indexes (NEW)
	indexData := make(map[string][]byte)
	for collName, idx := range vs.indexes {
		if idx != nil {
			data, err := idx.Export()
			if err == nil {
				indexData[collName] = data
			}
		}
	}

	payload := &storage.Payload{
		Dim:       vs.Dim,
		Data:      vs.Data,
		Docs:      vs.Docs,
		IDs:       vs.IDs,
		Meta:      vs.Meta,
		Deleted:   vs.Deleted,
		Coll:      vs.Coll,
		TenantID:  vs.TenantID,
		Next:      vs.next,
		Count:     vs.Count,
		HNSW:      hBuf,      // Legacy (deprecated)
		Indexes:   indexData, // NEW
		Checksum:  vs.checksum,
		LastSaved: time.Now(),
		LexTF:     vs.lexTF,
		DocLen:    vs.docLen,
		DF:        vs.df,
		SumDocL:   vs.sumDocL,
		NumMeta:   vs.NumMeta,
		TimeMeta:  vs.TimeMeta,
	}

	// Use configured storage format (default: gob for backward compatibility)
	format := vs.storageFormat
	if format == nil {
		format = storage.Default()
	}

	if err := format.Save(f, payload); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	vs.checksum = vs.computeChecksum()
	vs.lastSaved = payload.LastSaved
	if vs.walPath != "" {
		_ = os.Remove(vs.walPath)
	}
	return os.Rename(tmp, path)
}

// getStorageFormat returns the configured storage format from env var.
func getStorageFormat() storage.Format {
	if formatName := os.Getenv("STORAGE_FORMAT"); formatName != "" {
		if f := storage.Get(formatName); f != nil {
			return f
		}
	}
	return storage.Default()
}

// tryLoadPayload attempts to load a payload from path using multiple formats.
// Returns the payload and the format used, or nil if all formats fail.
func tryLoadPayload(path string) (*storage.Payload, storage.Format) {
	// Try gob first (most common, backward compatible)
	if payload := tryLoadWithFormat(path, storage.Get("gob")); payload != nil {
		return payload, storage.Get("gob")
	}

	// Try sjson formats
	for _, formatName := range []string{"sjson", "sjson-zstd"} {
		if f := storage.Get(formatName); f != nil {
			if payload := tryLoadWithFormat(path, f); payload != nil {
				return payload, f
			}
		}
	}

	return nil, nil
}

// tryLoadWithFormat attempts to load a payload using a specific format.
func tryLoadWithFormat(path string, format storage.Format) *storage.Payload {
	if format == nil {
		return nil
	}

	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()

	payload, err := format.Load(f)
	if err != nil {
		return nil
	}
	return payload
}

// Load snapshot or init new store.
func loadOrInitStore(path string, capacity int, dim int) (*VectorStore, bool) {
	if _, err := os.Stat(path); err == nil {
		// Try to load with configured format, fall back to gob for backward compatibility
		payload, loadedFormat := tryLoadPayload(path)
		if payload == nil {
			fmt.Printf("warning: failed to load index with any format, rebuilding\n")
			return NewVectorStore(capacity, dim), false
		}
		_ = loadedFormat // format used for loading (for logging if needed)
		// Initialize JWT manager if configured
		var jwtMgr *JWTManager
		if secret := os.Getenv("JWT_SECRET"); secret != "" {
			issuer := os.Getenv("JWT_ISSUER")
			if issuer == "" {
				issuer = "vectordb"
			}
			jwtMgr = NewJWTManager(secret, issuer)
		}

		vs := &VectorStore{
			Data:        payload.Data,
			Dim:         payload.Dim,
			Count:       payload.Count,
			Docs:        payload.Docs,
			IDs:         payload.IDs,
			next:        payload.Next,
			Meta:        payload.Meta,
			Deleted:     payload.Deleted,
			Coll:        payload.Coll,
			TenantID:    payload.TenantID,
			indexes:     make(map[string]index.Index), // NEW
			hnsw:        hnsw.NewGraph[uint64](),      // Legacy
			idToIx:      make(map[uint64]int),
			walPath:     path + ".wal",
			walMu:       sync.Mutex{},
			walMaxBytes: 0,
			walMaxOps:   0,
			apiToken:    os.Getenv("API_TOKEN"),
			requireAuth: os.Getenv("REQUIRE_AUTH") == "1" || jwtMgr != nil || os.Getenv("API_TOKEN") != "",
			checksum:    payload.Checksum,
			lastSaved:   payload.LastSaved,
			lexTF:       payload.LexTF,
			docLen:      payload.DocLen,
			df:          payload.DF,
			sumDocL:     payload.SumDocL,
			NumMeta:     payload.NumMeta,
			TimeMeta:    payload.TimeMeta,
			// Multi-tenancy support (TenantID already set from payload above)
			acl:      NewACL(),
			quotas:   NewTenantQuota(),
			tenantRL: newTenantRateLimiter(envInt("TENANT_RPS", 100), envInt("TENANT_BURST", 100), time.Minute),
			jwtMgr:   jwtMgr,
			// Storage format
			storageFormat: getStorageFormat(),
			// Metadata index (rebuilt below)
			metaIndex: NewMetadataIndex(),
		}
		if vs.checksum == "" {
			vs.checksum = fmt.Sprintf("%x", hashID(fmt.Sprintf("%d-%d", payload.Count, payload.Next)))
		}
		// Checksum is best-effort; warn but continue to allow recovery.
		if !vs.validateChecksum() {
			fmt.Printf("warning: checksum mismatch; continuing with loaded snapshot\n")
		}
		for i, idStr := range vs.IDs {
			vs.idToIx[hashID(idStr)] = i
		}
		if vs.Meta == nil {
			vs.Meta = make(map[uint64]map[string]string)
		}
		if vs.Deleted == nil {
			vs.Deleted = make(map[uint64]bool)
		}
		if vs.Coll == nil {
			vs.Coll = make(map[uint64]string)
		}
		if vs.TenantID == nil {
			vs.TenantID = make(map[uint64]string)
		}
		if vs.NumMeta == nil {
			vs.NumMeta = make(map[uint64]map[string]float64)
		}
		if vs.TimeMeta == nil {
			vs.TimeMeta = make(map[uint64]map[string]time.Time)
		}
		if vs.lexTF == nil {
			vs.lexTF = make(map[uint64]map[string]int)
		}
		if vs.docLen == nil {
			vs.docLen = make(map[uint64]int)
		}
		if vs.df == nil {
			vs.df = make(map[string]int)
		}
		// Import legacy HNSW graph (deprecated)
		if len(payload.HNSW) > 0 {
			if err := vs.hnsw.Import(bytes.NewReader(payload.HNSW)); err != nil {
				fmt.Printf("warning: failed to load hnsw graph, rebuilding: %v\n", err)
				vs.hnsw = hnsw.NewGraph[uint64]()
			}
		}

		// Import indexes (NEW)
		if len(payload.Indexes) > 0 {
			for collName, data := range payload.Indexes {
				// Create HNSW index for this collection
				idx, err := index.NewHNSWIndex(vs.Dim, nil)
				if err != nil {
					fmt.Printf("warning: failed to create index for collection %s: %v\n", collName, err)
					continue
				}
				if err := idx.Import(data); err != nil {
					fmt.Printf("warning: failed to import index for collection %s: %v\n", collName, err)
					continue
				}
				vs.indexes[collName] = idx
			}
		}

		// If no indexes were imported, create default index from legacy HNSW
		if len(vs.indexes) == 0 {
			cfg := loadHNSWConfig()
			defaultIdx, err := index.NewHNSWIndex(vs.Dim, map[string]interface{}{
				"m":         cfg.M,
				"ml":        cfg.Ml,
				"ef_search": cfg.EfSearch,
			})
			if err == nil {
				// Migrate legacy HNSW data to new index
				if len(payload.HNSW) > 0 {
					// Export from legacy and import to new index abstraction
					var buf bytes.Buffer
					if err := vs.hnsw.Export(&buf); err == nil {
						fmt.Println("Migrating legacy HNSW index to index abstraction...")
						// Note: Direct migration not possible due to different formats
						// Index will be rebuilt on first save
					}
				}
				vs.indexes["default"] = defaultIdx
			}
		}

		if vs.next == 0 {
			vs.next = int64(len(vs.IDs))
		}
		// Ensure tenant ownership defaults to "default" when missing
		for idKey := range vs.idToIx {
			if vs.TenantID[idKey] == "" {
				vs.TenantID[idKey] = "default"
			}
		}
		// Rebuild metadata bitmap index from persisted Meta
		if vs.metaIndex != nil && len(vs.Meta) > 0 {
			vs.metaIndex.RebuildFromMeta(vs.Meta, vs.Deleted)
			fmt.Printf("Rebuilt metadata index: %d documents indexed\n", vs.metaIndex.GetDocumentCount())
		}
		if err := replayWAL(vs); err != nil {
			fmt.Printf("warning: WAL replay failed: %v\n", err)
		}
		return vs, true
	}
	vs := NewVectorStore(capacity, dim)
	vs.walPath = path + ".wal"
	return vs, false
}

type hnswConfig struct {
	M        int
	Ml       float64
	EfSearch int
}

func loadHNSWConfig() hnswConfig {
	cfg := hnswConfig{M: 16, Ml: 0.25, EfSearch: 64}
	if v := os.Getenv("HNSW_M"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.M = n
		}
	}
	if v := os.Getenv("HNSW_ML"); v != "" {
		if n, err := strconv.ParseFloat(v, 64); err == nil && n > 0 {
			cfg.Ml = n
		}
	}
	if v := os.Getenv("HNSW_EFSEARCH"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.EfSearch = n
		}
	}
	return cfg
}

// ======================================================================================
// Tools and Agents
// ======================================================================================

type RetrievalTool struct {
	Store    *VectorStore
	Embedder Embedder
}

func (t *RetrievalTool) Name() string { return "flat_buffer_retrieval" }

type RetrievalInput struct {
	Query string `json:"query"`
}

type RetrievalResult struct {
	IDs   []string `json:"ids"`
	Docs  []string `json:"docs"`
	Stats string   `json:"stats"`
}

func (t *RetrievalTool) Execute(ctx *core.ToolContext) *core.ToolExecResult {
	var queryText string
	if ctx.Request != nil && ctx.Request.ToolReq != nil {
		if input, ok := ctx.Request.ToolReq.Input.(*RetrievalInput); ok && input != nil {
			queryText = input.Query
		}
	}
	if queryText == "" {
		queryText = "default"
	}

	qVec, err := t.Embedder.Embed(queryText)
	if err != nil {
		return &core.ToolExecResult{
			Status: core.ToolFailed,
			Error:  fmt.Sprintf("embedder error: %v", err),
		}
	}

	start := time.Now()
	ids := t.Store.SearchANN(qVec, 3)
	elapsed := time.Since(start)

	output := fmt.Sprintf("Search completed in %s. Found IDs: %v.", elapsed, ids)
	docs := make([]string, 0, len(ids))
	idStrings := make([]string, 0, len(ids))
	for _, id := range ids {
		docs = append(docs, t.Store.GetDoc(id))
		idStrings = append(idStrings, t.Store.GetID(id))
	}

	return &core.ToolExecResult{
		Status: core.ToolComplete,
		Output: &RetrievalResult{
			IDs:   idStrings,
			Docs:  docs,
			Stats: output,
		},
		Duration: elapsed,
	}
}

type RerankInput struct {
	Query string   `json:"query"`
	Docs  []string `json:"docs"`
	TopK  int      `json:"top_k"`
}

type RerankResult struct {
	Docs   []string  `json:"docs"`
	Scores []float32 `json:"scores"`
	Stats  string    `json:"stats"`
	IDs    []int     `json:"ids,omitempty"`
}

type RerankTool struct {
	Reranker Reranker
}

func (t *RerankTool) Name() string { return "reranker" }

func (t *RerankTool) Execute(ctx *core.ToolContext) *core.ToolExecResult {
	var input RerankInput
	if ctx.Request != nil && ctx.Request.ToolReq != nil {
		if in, ok := ctx.Request.ToolReq.Input.(*RerankInput); ok && in != nil {
			input = *in
		}
	}
	if input.TopK <= 0 || input.TopK > len(input.Docs) {
		input.TopK = len(input.Docs)
	}

	docs, scores, stats, err := t.Reranker.Rerank(input.Query, input.Docs, input.TopK)
	if err != nil {
		return &core.ToolExecResult{Status: core.ToolFailed, Error: fmt.Sprintf("reranker error: %v", err)}
	}

	return &core.ToolExecResult{
		Status: core.ToolComplete,
		Output: &RerankResult{
			Docs:   docs,
			Scores: scores,
			Stats:  stats,
		},
	}
}

type RAGState struct {
	Step    string
	Query   string
	Context string
}

// RangeFilter supports numeric or RFC3339 time comparisons.
type RangeFilter struct {
	Key     string   `json:"key"`
	Min     *float64 `json:"min,omitempty"`
	Max     *float64 `json:"max,omitempty"`
	TimeMin string   `json:"time_min,omitempty"` // RFC3339
	TimeMax string   `json:"time_max,omitempty"` // RFC3339
	LexOnly bool     `json:"lex_only,omitempty"` // force lexical mode on this field
}

func NewRAGAgent(llmToolID core.ToolID, retrievalToolID core.ToolID, rerankToolID core.ToolID) *core.FuncAgent {
	return core.NewFuncAgent("rag_orchestrator", func(ctx *core.AgentContext, msg *core.Message) {
		state, _ := ctx.UserState.(*RAGState)
		if state == nil {
			state = &RAGState{Step: "start"}
		}

		switch msg.Type {
		case core.MsgUserInput:
			state.Query = msg.Text
			state.Step = "retrieving"
			ctx.Scheduler.RequestTool(ctx, &core.ToolRequestPayload{
				ToolID: retrievalToolID,
				Input:  &RetrievalInput{Query: msg.Text},
			})

		case core.MsgToolResult:
			switch msg.ToolRes.ToolID {
			case retrievalToolID:
				state.Step = "reranking"
				var docs []string
				switch out := msg.ToolRes.Output.(type) {
				case *RetrievalResult:
					docs = out.Docs
				case RetrievalResult:
					docs = out.Docs
				case map[string]any:
					if rawDocs, ok := out["docs"].([]string); ok {
						docs = rawDocs
					}
				}
				if len(docs) == 0 {
					docs = []string{"no docs retrieved"}
				}
				ctx.Scheduler.RequestTool(ctx, &core.ToolRequestPayload{
					ToolID: rerankToolID,
					Input: &RerankInput{
						Query: state.Query,
						Docs:  docs,
						TopK:  3,
					},
				})

			case rerankToolID:
				state.Step = "generating"
				var docs []string
				switch out := msg.ToolRes.Output.(type) {
				case *RerankResult:
					docs = out.Docs
				case RerankResult:
					docs = out.Docs
				case map[string]any:
					if rawDocs, ok := out["docs"].([]string); ok {
						docs = rawDocs
					}
				}
				state.Context = strings.Join(docs, "\n---\n")

				systemPrompt := "You are a high-performance technical assistant."
				userPrompt := fmt.Sprintf("Question: %s\n\nContext:\n%s", state.Query, state.Context)

				ctx.Scheduler.RequestTool(ctx, &core.ToolRequestPayload{
					ToolID: llmToolID,
					Input: &tools.LLMRequest{
						System: systemPrompt,
						Messages: []tools.LLMMessage{
							{Role: "user", Content: userPrompt},
						},
					},
				})

			case llmToolID:
				state.Step = "done"
				ctx.Scheduler.ConvMgr().MarkComplete(ctx.ConvID)
			}
		}
		ctx.UserState = state
	})
}

// ======================================================================================
// Embedder/Reranker interfaces and hash fallback
// ======================================================================================

type Embedder interface {
	Embed(text string) ([]float32, error)
	Dim() int
}

type Reranker interface {
	Rerank(query string, docs []string, topK int) ([]string, []float32, string, error)
}

type HashEmbedder struct {
	dim int
}

func NewHashEmbedder(dim int) *HashEmbedder {
	return &HashEmbedder{dim: dim}
}

func (e *HashEmbedder) Dim() int { return e.dim }

func (e *HashEmbedder) Embed(text string) ([]float32, error) {
	if text == "" {
		text = "empty"
	}
	h := fnv.New64a()
	_, _ = h.Write([]byte(text))
	seed := int64(h.Sum64())
	vec := make([]float32, e.dim)
	r := rand.New(rand.NewSource(seed))
	var norm float64
	for i := 0; i < e.dim; i++ {
		val := r.Float64()*2 - 1
		vec[i] = float32(val)
		norm += val * val
	}
	norm = math.Sqrt(norm)
	if norm == 0 {
		return vec, nil
	}
	for i := range vec {
		vec[i] /= float32(norm)
	}
	return vec, nil
}

// OpenAIEmbedder uses OpenAI's text-embedding-3-small model
type OpenAIEmbedder struct {
	apiKey string
	model  string
	dim    int
	client *http.Client
}

func NewOpenAIEmbedder(apiKey string) *OpenAIEmbedder {
	return &OpenAIEmbedder{
		apiKey: apiKey,
		model:  "text-embedding-3-small",
		dim:    1536,
		client: &http.Client{Timeout: 30 * time.Second},
	}
}

func (e *OpenAIEmbedder) Dim() int { return e.dim }

func (e *OpenAIEmbedder) Embed(text string) ([]float32, error) {
	if text == "" {
		text = "empty"
	}

	reqBody := map[string]interface{}{
		"input": text,
		"model": e.model,
	}
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal error: %w", err)
	}

	req, err := http.NewRequest("POST", "https://api.openai.com/v1/embeddings", bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("request error: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.apiKey)

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("API error: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		var errResp struct {
			Error struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		json.NewDecoder(resp.Body).Decode(&errResp)
		return nil, fmt.Errorf("OpenAI API error %d: %s", resp.StatusCode, errResp.Error.Message)
	}

	var result struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode error: %w", err)
	}

	if len(result.Data) == 0 || len(result.Data[0].Embedding) == 0 {
		return nil, fmt.Errorf("no embedding returned")
	}

	// Convert float64 to float32
	vec := make([]float32, len(result.Data[0].Embedding))
	for i, v := range result.Data[0].Embedding {
		vec[i] = float32(v)
	}
	return vec, nil
}

// OllamaEmbedder uses Ollama's local embedding models
type OllamaEmbedder struct {
	baseURL string
	model   string
	dim     int
	client  *http.Client
}

func NewOllamaEmbedder(baseURL, model string) *OllamaEmbedder {
	// nomic-embed-text produces 768-dim vectors
	dim := 768
	if model == "granite-embedding" {
		dim = 384
	}
	return &OllamaEmbedder{
		baseURL: baseURL,
		model:   model,
		dim:     dim,
		client:  &http.Client{Timeout: 60 * time.Second},
	}
}

func (e *OllamaEmbedder) Dim() int { return e.dim }

func (e *OllamaEmbedder) Embed(text string) ([]float32, error) {
	if text == "" {
		text = "empty"
	}

	reqBody := map[string]interface{}{
		"model":  e.model,
		"prompt": text,
	}
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal error: %w", err)
	}

	req, err := http.NewRequest("POST", e.baseURL+"/api/embeddings", bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("request error: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("Ollama API error: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Ollama API error %d", resp.StatusCode)
	}

	var result struct {
		Embedding []float64 `json:"embedding"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode error: %w", err)
	}

	if len(result.Embedding) == 0 {
		return nil, fmt.Errorf("no embedding returned")
	}

	// Convert float64 to float32 and L2 normalize
	vec := make([]float32, len(result.Embedding))
	var norm float64
	for i, v := range result.Embedding {
		vec[i] = float32(v)
		norm += v * v
	}
	norm = math.Sqrt(norm)
	if norm > 0 {
		for i := range vec {
			vec[i] /= float32(norm)
		}
	}
	e.dim = len(vec) // Update dim based on actual response
	return vec, nil
}

type SimpleReranker struct {
	Embedder Embedder
}

func (r *SimpleReranker) Rerank(query string, docs []string, topK int) ([]string, []float32, string, error) {
	qVec, err := r.Embedder.Embed(query)
	if err != nil {
		return nil, nil, "", err
	}
	if topK <= 0 || topK > len(docs) {
		topK = len(docs)
	}
	bestDocs := make([]string, 0, topK)
	bestScores := make([]float32, 0, topK)

	for _, doc := range docs {
		dVec, err := r.Embedder.Embed(doc)
		if err != nil {
			continue
		}
		score := DotProduct(qVec, dVec)
		if len(bestDocs) < topK {
			bestDocs = append(bestDocs, doc)
			bestScores = append(bestScores, score)
			continue
		}
		minIdx := 0
		for i := 1; i < len(bestScores); i++ {
			if bestScores[i] < bestScores[minIdx] {
				minIdx = i
			}
		}
		if score > bestScores[minIdx] {
			bestScores[minIdx] = score
			bestDocs[minIdx] = doc
		}
	}
	return bestDocs, bestScores, "Simple rerank", nil
}

// ======================================================================================
// Utility
// ======================================================================================

func DotProduct(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	var sum float32
	for i := 0; i < len(a); i++ {
		sum += a[i] * b[i]
	}
	return sum
}

func hashID(s string) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(s))
	return h.Sum64()
}

func matchesMeta(meta map[string]string, filt map[string]string) bool {
	if len(filt) == 0 {
		return true
	}
	for k, v := range filt {
		if mv, ok := meta[k]; !ok || mv != v {
			return false
		}
	}
	return true
}

func matchesAny(meta map[string]string, any []map[string]string) bool {
	for _, f := range any {
		if matchesMeta(meta, f) {
			return true
		}
	}
	return false
}

func matchesRanges(meta map[string]string, num map[string]float64, times map[string]time.Time, ranges []RangeFilter) bool {
	if len(ranges) == 0 {
		return true
	}
	for _, rf := range ranges {
		val, ok := meta[rf.Key]
		if !ok {
			return false
		}
		// Try time bounds if provided.
		if rf.TimeMin != "" || rf.TimeMax != "" {
			mv, okT := times[rf.Key]
			if !okT {
				var err error
				mv, err = time.Parse(time.RFC3339, val)
				if err != nil {
					return false
				}
			}
			if rf.TimeMin != "" {
				minT, err := time.Parse(time.RFC3339, rf.TimeMin)
				if err != nil || mv.Before(minT) {
					return false
				}
			}
			if rf.TimeMax != "" {
				maxT, err := time.Parse(time.RFC3339, rf.TimeMax)
				if err != nil || mv.After(maxT) {
					return false
				}
			}
			continue
		}
		// Numeric bounds if set.
		if rf.Min != nil || rf.Max != nil {
			fv, okN := num[rf.Key]
			if !okN {
				var err error
				fv, err = strconv.ParseFloat(val, 64)
				if err != nil {
					return false
				}
			}
			if rf.Min != nil && fv < *rf.Min {
				return false
			}
			if rf.Max != nil && fv > *rf.Max {
				return false
			}
		}
	}
	return true
}

func (vs *VectorStore) candidateIDsForRange(ranges []RangeFilter) map[uint64]struct{} {
	if len(ranges) == 0 {
		return nil
	}
	candidates := make(map[uint64]struct{})
	for idx, rf := range ranges {
		local := make(map[uint64]struct{})
		if rf.Min != nil || rf.Max != nil {
			entries := vs.numIndex[rf.Key]
			for _, e := range entries {
				if rf.Min != nil && e.V < *rf.Min {
					continue
				}
				if rf.Max != nil && e.V > *rf.Max {
					continue
				}
				local[e.ID] = struct{}{}
			}
		} else if rf.TimeMin != "" || rf.TimeMax != "" {
			entries := vs.timeIndex[rf.Key]
			var minT, maxT time.Time
			if rf.TimeMin != "" {
				minT, _ = time.Parse(time.RFC3339, rf.TimeMin)
			}
			if rf.TimeMax != "" {
				maxT, _ = time.Parse(time.RFC3339, rf.TimeMax)
			}
			for _, e := range entries {
				if !minT.IsZero() && e.T.Before(minT) {
					continue
				}
				if !maxT.IsZero() && e.T.After(maxT) {
					continue
				}
				local[e.ID] = struct{}{}
			}
		}
		if idx == 0 {
			candidates = local
		} else {
			// intersect
			for id := range candidates {
				if _, ok := local[id]; !ok {
					delete(candidates, id)
				}
			}
		}
	}
	return candidates
}

// tokenize is a simple Unicode-aware tokenizer used for metadata/text analysis.
func tokenize(text string) []string {
	stop := loadStopwords()
	tokens := make([]string, 0, len(text)/4+1)
	var buf strings.Builder

	flush := func() {
		if buf.Len() == 0 {
			return
		}
		tok := strings.ToLower(strings.TrimFunc(buf.String(), func(r rune) bool {
			return unicode.IsPunct(r) || unicode.IsSymbol(r)
		}))
		buf.Reset()
		if tok != "" && (!stopwordEnabled || !stop[tok]) {
			tokens = append(tokens, tok)
		}
	}

	for _, r := range text {
		switch {
		case unicode.IsSpace(r):
			flush()
		case unicode.IsPunct(r) || unicode.IsSymbol(r):
			flush()
		default:
			buf.WriteRune(r)
		}
	}
	flush()
	return tokens
}

var stopwords map[string]bool
var stopOnce sync.Once
var stopwordEnabled = true

func loadStopwords() map[string]bool {
	stopOnce.Do(func() {
		stopwords = map[string]bool{
			"the": true, "a": true, "an": true, "in": true, "on": true, "for": true,
			"and": true, "or": true, "but": true, "of": true, "to": true, "is": true,
		}
		if os.Getenv("DISABLE_STOPWORDS") == "1" {
			stopwordEnabled = false
		}
		if extra := os.Getenv("STOPWORDS_EXTRA"); extra != "" {
			for _, tok := range strings.Split(extra, ",") {
				tok = strings.TrimSpace(strings.ToLower(tok))
				if tok != "" {
					stopwords[tok] = true
				}
			}
		}
	})
	return stopwords
}

type walEntry struct {
	Seq    uint64 // Monotonic sequence number (for WAL streaming)
	Op     string
	ID     string
	Doc    string
	Meta   map[string]string
	Vec    []float32
	Coll   string
	Tenant string
	Time   int64 // Unix timestamp (for WAL streaming)
}

type numEntry struct {
	ID uint64
	V  float64
}

type timeEntry struct {
	ID uint64
	T  time.Time
}

// SetWALHook registers a callback that receives every WAL entry after it is persisted.
// This is used by the shard server to stream writes to replicas.
func (vs *VectorStore) SetWALHook(h func(walEntry)) {
	vs.walMu.Lock()
	defer vs.walMu.Unlock()
	vs.walHook = h
}

func (vs *VectorStore) appendWAL(op, id, doc string, meta map[string]string, vec []float32, tenant string) error {
	// Normalize tenant
	if tenant == "" && id != "" {
		tenant = vs.TenantID[hashID(id)]
	}
	if tenant == "" {
		tenant = "default"
	}

	entry := walEntry{
		Op:   op,
		ID:   id,
		Doc:  doc,
		Meta: meta,
		Vec:  vec,
		Coll: func() string {
			if id == "" {
				return ""
			}
			return vs.Coll[hashID(id)]
		}(),
		Tenant: tenant,
		Time:   time.Now().Unix(),
	}

	if vs.walPath != "" {
		vs.walMu.Lock()
		f, err := os.OpenFile(vs.walPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			vs.walMu.Unlock()
			return fmt.Errorf("failed to open WAL: %w", err)
		}

		if err := json.NewEncoder(f).Encode(&entry); err != nil {
			f.Close()
			vs.walMu.Unlock()
			return fmt.Errorf("failed to encode WAL entry: %w", err)
		}

		// Fsync to ensure durability
		if err := f.Sync(); err != nil {
			f.Close()
			vs.walMu.Unlock()
			return fmt.Errorf("failed to sync WAL: %w", err)
		}
		vs.walOps++

		var doSnapshot bool
		if vs.walMaxOps > 0 && vs.walOps >= vs.walMaxOps {
			doSnapshot = true
			vs.walOps = 0
		}
		if !doSnapshot && vs.walMaxBytes > 0 {
			if info, err := f.Stat(); err == nil && info.Size() >= vs.walMaxBytes {
				doSnapshot = true
				vs.walOps = 0
			}
		}
		_ = f.Close()
		vs.walMu.Unlock()

		if doSnapshot {
			snapPath := strings.TrimSuffix(vs.walPath, ".wal")
			go func() {
				if err := vs.Save(snapPath); err == nil {
					_ = os.Truncate(vs.walPath, 0)
				}
			}()
		}
	}

	// Forward to external WAL hooks (e.g., replication stream)
	if vs.walHook != nil {
		vs.walHook(entry)
	}

	return nil
}

func replayWAL(vs *VectorStore) error {
	if vs.walPath == "" {
		return nil
	}
	f, err := os.Open(vs.walPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // No WAL to replay
		}
		return fmt.Errorf("failed to open WAL for replay: %w", err)
	}
	defer f.Close()

	// Disable WAL writes during replay to avoid re-appending the same ops.
	path := vs.walPath
	vs.walPath = ""
	hook := vs.walHook
	vs.walHook = nil
	defer func() {
		vs.walPath = path
		vs.walHook = hook
	}()

	dec := json.NewDecoder(f)
	var replayErrors []error
	for {
		var e walEntry
		if err := dec.Decode(&e); err != nil {
			if err.Error() != "EOF" {
				replayErrors = append(replayErrors, fmt.Errorf("WAL decode error: %w", err))
			}
			break
		}
		if e.Tenant == "" {
			e.Tenant = "default"
		}
		switch e.Op {
		case "insert":
			if _, err := vs.Add(e.Vec, e.Doc, e.ID, e.Meta, e.Coll, e.Tenant); err != nil {
				replayErrors = append(replayErrors, fmt.Errorf("replay insert failed: %w", err))
			}
		case "upsert":
			if _, err := vs.Upsert(e.Vec, e.Doc, e.ID, e.Meta, e.Coll, e.Tenant); err != nil {
				replayErrors = append(replayErrors, fmt.Errorf("replay upsert failed: %w", err))
			}
		case "delete":
			if err := vs.Delete(e.ID); err != nil {
				replayErrors = append(replayErrors, fmt.Errorf("replay delete failed: %w", err))
			}
		}
	}

	if len(replayErrors) > 0 {
		// Log errors but don't fail - partial replay is better than none
		fmt.Printf("WAL replay completed with %d errors\n", len(replayErrors))
		for _, e := range replayErrors {
			fmt.Printf("  - %v\n", e)
		}
	}

	_ = os.Remove(path)
	return nil
}

func (vs *VectorStore) computeChecksum() string {
	return fmt.Sprintf("%x", hashID(fmt.Sprintf("%d-%d-%d", vs.Count, vs.next, len(vs.Docs))))
}

func (vs *VectorStore) validateChecksum() bool {
	if vs.checksum == "" {
		return true
	}
	return vs.checksum == vs.computeChecksum()
}

// ======================================================================================
// Embedder selection (Hash by default; ONNX under build tag)
// ======================================================================================

func initEmbedder(defaultDim int) Embedder {
	// Priority 1: OpenAI embeddings (highest quality, requires API key)
	if apiKey := os.Getenv("OPENAI_API_KEY"); apiKey != "" {
		fmt.Println(">>> Using OpenAI embedder (text-embedding-3-small)")
		return NewOpenAIEmbedder(apiKey)
	}

	// Priority 2: Ollama embeddings (local, good quality)
	ollamaURL := os.Getenv("OLLAMA_URL")
	if ollamaURL == "" {
		ollamaURL = "http://localhost:11434"
	}
	ollamaModel := os.Getenv("OLLAMA_EMBED_MODEL")
	if ollamaModel == "" {
		ollamaModel = "nomic-embed-text" // Default to nomic-embed-text
	}
	// Test if Ollama is available
	client := &http.Client{Timeout: 5 * time.Second}
	if resp, err := client.Get(ollamaURL + "/api/tags"); err == nil {
		resp.Body.Close()
		if resp.StatusCode == http.StatusOK {
			fmt.Printf(">>> Using Ollama embedder (%s)\n", ollamaModel)
			return NewOllamaEmbedder(ollamaURL, ollamaModel)
		}
	}

	// Priority 3: ONNX embeddings (local, good quality, requires onnxruntime)
	defaultModel := "vectordb/models/bge-small-en-v1.5/model.onnx"
	defaultTok := "vectordb/models/bge-small-en-v1.5/tokenizer.json"

	modelPath := os.Getenv("ONNX_EMBED_MODEL")
	tokPath := os.Getenv("ONNX_EMBED_TOKENIZER")
	if modelPath == "" {
		if _, err := os.Stat(defaultModel); err == nil {
			modelPath = defaultModel
		}
	}
	if tokPath == "" {
		if _, err := os.Stat(defaultTok); err == nil {
			tokPath = defaultTok
		}
	}
	maxLen := 512
	if env := os.Getenv("ONNX_EMBED_MAX_LEN"); env != "" {
		if v, err := strconv.Atoi(env); err == nil && v >= 0 {
			maxLen = v
		}
	}
	if modelPath != "" && tokPath != "" {
		if emb, err := NewOnnxEmbedder(modelPath, tokPath, defaultDim, maxLen); err == nil {
			fmt.Println(">>> Using ONNX embedder:", modelPath)
			return emb
		}
		fmt.Println(">>> ONNX init failed")
	}

	// Priority 4: Hash embedder (fallback, low quality)
	fmt.Println(">>> Using hash embedder (set OPENAI_API_KEY or start Ollama for better quality)")
	return NewHashEmbedder(defaultDim)
}

func initReranker(embedder Embedder) Reranker {
	modelPath := os.Getenv("ONNX_RERANK_MODEL")
	tokPath := os.Getenv("ONNX_RERANK_TOKENIZER")
	maxLen := 512
	if env := os.Getenv("ONNX_RERANK_MAX_LEN"); env != "" {
		if v, err := strconv.Atoi(env); err == nil && v >= 0 {
			maxLen = v
		}
	}
	if modelPath != "" && tokPath != "" {
		if rr, err := NewOnnxCrossEncoderReranker(modelPath, tokPath, maxLen); err == nil {
			fmt.Println("Using ONNX reranker:", modelPath)
			return rr
		}
		fmt.Println("Falling back to simple reranker (ONNX init failed)")
	}
	return &SimpleReranker{Embedder: embedder}
}

// warmupModels optionally runs a dummy embed/rerank to catch missing models early.
func warmupModels(embedder Embedder, reranker Reranker) {
	if os.Getenv("DISABLE_WARMUP") == "1" {
		return
	}
	if embedder != nil {
		if _, err := embedder.Embed("warmup"); err != nil {
			fmt.Printf("embedder warmup warning: %v\n", err)
		}
	}
	if reranker != nil {
		if _, _, _, err := reranker.Rerank("warmup", []string{"warmup"}, 1); err != nil {
			fmt.Printf("reranker warmup warning: %v\n", err)
		}
	}
}

// ======================================================================================
// Coordinator Mode: runs as distributed coordinator with quorum
// ======================================================================================

func runCoordinatorMode() {
	fmt.Println(">>> Starting Distributed Coordinator Mode...")

	// Read coordinator configuration from environment
	coordinatorID := os.Getenv("COORDINATOR_ID")
	if coordinatorID == "" {
		coordinatorID = fmt.Sprintf("coordinator-%d", time.Now().Unix())
	}

	// Parse peer coordinators
	peerCoordinators := []string{}
	if peers := os.Getenv("PEER_COORDINATORS"); peers != "" {
		peerCoordinators = strings.Split(peers, ",")
		for i := range peerCoordinators {
			peerCoordinators[i] = strings.TrimSpace(peerCoordinators[i])
		}
	}

	// Get port
	port := envInt("PORT", 8080)
	listenAddr := fmt.Sprintf(":%d", port)

	// Get number of shards and replication factor
	numShards := envInt("NUM_SHARDS", 2)
	replicationFactor := envInt("REPLICATION_FACTOR", 2)

	// Determine read strategy
	readStrategyStr := os.Getenv("READ_STRATEGY")
	var readStrategy ReadStrategy
	switch readStrategyStr {
	case "primary-only":
		readStrategy = ReadPrimaryOnly
	case "replica-prefer":
		readStrategy = ReadReplicaPrefer
	case "balanced":
		readStrategy = ReadBalanced
	default:
		readStrategy = ReadReplicaPrefer
	}

	// Initialize JWT manager if configured
	var jwtMgr *JWTManager
	if secret := os.Getenv("JWT_SECRET"); secret != "" {
		issuer := os.Getenv("JWT_ISSUER")
		if issuer == "" {
			issuer = "vectordb"
		}
		jwtMgr = NewJWTManager(secret, issuer)
	}

	// Create coordinator config
	cfg := CoordinatorServerConfig{
		ListenAddr:        listenAddr,
		NumShards:         numShards,
		ReplicationFactor: replicationFactor,
		ReadStrategy:      readStrategy,
		CoordinatorID:     coordinatorID,
		PeerCoordinators:  peerCoordinators,
		EnableFailover:    os.Getenv("ENABLE_FAILOVER") == "1",
		FailoverConfig: FailoverConfig{
			UnhealthyThreshold: time.Duration(envInt("FAILOVER_THRESHOLD_SEC", 30)) * time.Second,
			CheckInterval:      time.Duration(envInt("FAILOVER_CHECK_SEC", 5)) * time.Second,
			EnableAutoFailover: os.Getenv("ENABLE_AUTO_FAILOVER") == "1",
		},
		EnableMetrics: os.Getenv("ENABLE_METRICS") != "0", // Enabled by default
		EnableAuth:    jwtMgr != nil,
		JWTMgr:        jwtMgr,
	}

	fmt.Printf(">>> Coordinator Configuration:\n")
	fmt.Printf("    ID: %s\n", coordinatorID)
	fmt.Printf("    Listen: %s\n", listenAddr)
	fmt.Printf("    Shards: %d\n", numShards)
	fmt.Printf("    Replication Factor: %d\n", replicationFactor)
	fmt.Printf("    Read Strategy: %s\n", readStrategy)
	if len(peerCoordinators) > 0 {
		fmt.Printf("    Peer Coordinators: %v\n", peerCoordinators)
	} else {
		fmt.Printf("    Peer Coordinators: none (single-coordinator mode)\n")
	}
	fmt.Printf("    Failover: enabled=%v, auto=%v\n", cfg.EnableFailover, cfg.FailoverConfig.EnableAutoFailover)
	fmt.Println()

	// Run coordinator
	RunCoordinator(cfg)
}

// ======================================================================================
// Shard Mode: runs as a shard server (primary or replica)
// ======================================================================================

func runShardMode() {
	fmt.Println(">>> Starting Shard Server Mode...")

	// Read shard configuration from environment
	nodeID := os.Getenv("NODE_ID")
	if nodeID == "" {
		fmt.Println("ERROR: NODE_ID environment variable required for shard mode")
		os.Exit(1)
	}

	shardID := envInt("SHARD_ID", -1)
	if shardID < 0 {
		fmt.Println("ERROR: SHARD_ID environment variable required for shard mode")
		os.Exit(1)
	}

	roleStr := os.Getenv("ROLE")
	if roleStr == "" {
		fmt.Println("ERROR: ROLE environment variable required (primary or replica)")
		os.Exit(1)
	}
	var role ReplicaRole
	switch roleStr {
	case "primary":
		role = RolePrimary
	case "replica":
		role = RoleReplica
	default:
		fmt.Printf("ERROR: Invalid ROLE '%s' (must be 'primary' or 'replica')\n", roleStr)
		os.Exit(1)
	}

	httpAddr := os.Getenv("HTTP_ADDR")
	if httpAddr == "" {
		fmt.Println("ERROR: HTTP_ADDR environment variable required (e.g., ':9000')")
		os.Exit(1)
	}

	coordinatorAddr := os.Getenv("COORDINATOR_ADDR")
	if coordinatorAddr == "" {
		fmt.Println("WARNING: COORDINATOR_ADDR not set, will not register with coordinator")
	}

	// Optional: primary address for replicas
	primaryAddr := os.Getenv("PRIMARY_ADDR")
	if role == RoleReplica && primaryAddr == "" {
		fmt.Println("WARNING: REPLICA without PRIMARY_ADDR set")
	}

	// Optional: replica addresses for primary
	var replicas []string
	if replicasStr := os.Getenv("REPLICAS"); replicasStr != "" {
		replicas = strings.Split(replicasStr, ",")
		for i := range replicas {
			replicas[i] = strings.TrimSpace(replicas[i])
		}
	}

	// Vector store config
	capacity := envInt("CAPACITY", 1000) // Reduced from 100000 for low-memory deployment
	dimension := envInt("DIMENSION", 384)
	indexPath := os.Getenv("INDEX_PATH")
	if indexPath == "" {
		indexPath = fmt.Sprintf("vectordb/shard-%d-%s.gob", shardID, nodeID)
	}

	// Initialize embedder
	embedder := initEmbedder(dimension)

	// Create shard server config
	cfg := ShardServerConfig{
		NodeID:          nodeID,
		ShardID:         shardID,
		Role:            role,
		HTTPAddr:        httpAddr,
		PrimaryAddr:     primaryAddr,
		Replicas:        replicas,
		CoordinatorAddr: coordinatorAddr,
		Capacity:        capacity,
		Dimension:       dimension,
		IndexPath:       indexPath,
		Embedder:        embedder,
	}

	fmt.Printf(">>> Shard Configuration:\n")
	fmt.Printf("    Node ID: %s\n", nodeID)
	fmt.Printf("    Shard ID: %d\n", shardID)
	fmt.Printf("    Role: %s\n", role)
	fmt.Printf("    HTTP Address: %s\n", httpAddr)
	if role == RoleReplica && primaryAddr != "" {
		fmt.Printf("    Primary: %s\n", primaryAddr)
	}
	if role == RolePrimary && len(replicas) > 0 {
		fmt.Printf("    Replicas: %v\n", replicas)
	}
	if coordinatorAddr != "" {
		fmt.Printf("    Coordinator: %s\n", coordinatorAddr)
	}
	fmt.Printf("    Index Path: %s\n", indexPath)
	fmt.Println()

	// Create shard server
	shard, err := NewShardServer(cfg)
	if err != nil {
		fmt.Printf("ERROR: Failed to create shard server: %v\n", err)
		os.Exit(1)
	}

	// Start shard server
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if err := shard.Start(ctx); err != nil {
		fmt.Printf("ERROR: Failed to start shard server: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("Shard server started. Press Ctrl+C to shutdown.")

	// Wait for shutdown signal
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh

	fmt.Println("\nShutdown signal received...")
	cancel()

	// Graceful shutdown
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()
	if err := shard.Shutdown(shutdownCtx); err != nil {
		fmt.Printf("ERROR: Shutdown failed: %v\n", err)
		os.Exit(1)
	}
}

// ======================================================================================
// Main: bootstrap, HTTP API
// ======================================================================================

func main() {
	// Check if running in shard mode
	if os.Getenv("SHARD_MODE") == "1" {
		runShardMode()
		return
	}

	// Check if running in coordinator mode
	if os.Getenv("COORDINATOR_MODE") == "1" {
		runCoordinatorMode()
		return
	}

	// Check if running in multi-protocol transport mode
	if os.Getenv("TRANSPORT_MODE") == "1" {
		runWithTransport()
		return
	}

	// Initialize structured logging
	logConfig := logging.DefaultConfig()
	if os.Getenv("LOG_FORMAT") == "text" {
		logConfig.Format = "text"
	}
	if os.Getenv("LOG_LEVEL") == "debug" {
		logConfig.Level = logging.LevelDebug
	}
	logger := logging.Init(logConfig)
	logger.Info("initializing vector engine")

	const indexPath = "vectordb/index.gob"
	defaultDim := 384
	if envDim := os.Getenv("EMBED_DIM"); envDim != "" {
		if v, err := strconv.Atoi(envDim); err == nil && v >= 0 {
			defaultDim = v
		}
	}

	initMetrics()

	// Initialize OpenTelemetry tracing
	// Configured via environment variables:
	//   OTEL_SERVICE_NAME - service name (default: "vectordb")
	//   OTEL_EXPORTER_OTLP_ENDPOINT - OTLP endpoint (optional)
	//   OTEL_TRACE_SAMPLE_RATE - sampling rate (default: 1.0)
	//   OTEL_ENABLE_CONSOLE - enable console exporter (default: false)
	if err := telemetry.SetupSimple(); err != nil {
		logger.Warn("telemetry setup failed", "error", err)
	} else {
		logger.Info("opentelemetry tracing initialized")
		defer func() {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			if err := telemetry.Shutdown(ctx); err != nil {
				logger.Warn("telemetry shutdown failed", "error", err)
			}
		}()
	}

	// Check if we should force hash embedder (low-memory mode)
	var embedder Embedder
	if os.Getenv("USE_HASH_EMBEDDER") == "1" {
		logger.Info("using hash embedder (low-memory mode)")
		embedder = NewHashEmbedder(defaultDim)
	} else {
		embedder = initEmbedder(defaultDim)
	}

	// Make initial capacity configurable for low-memory deployments
	initialCapacity := 1000 // Reduced from 100000 for low-memory deployment
	if envCap := os.Getenv("VECTOR_CAPACITY"); envCap != "" {
		if v, err := strconv.Atoi(envCap); err == nil && v >= 0 {
			initialCapacity = v
		}
	}
	store, loaded := loadOrInitStore(indexPath, initialCapacity, embedder.Dim())
	store.walMaxBytes = envInt64("WAL_MAX_BYTES", 5*1024*1024)
	store.walMaxOps = envInt("WAL_MAX_OPS", 1000)

	// Setup replication if enabled
	if os.Getenv("ENABLE_REPLICATION") == "1" {
		logger.Info("replication enabled")

		// Parse replicas
		var replicas []*ReplicaNode
		if replicasStr := os.Getenv("REPLICAS"); replicasStr != "" {
			for i, addr := range strings.Split(replicasStr, ",") {
				addr = strings.TrimSpace(addr)
				if addr != "" {
					replicas = append(replicas, &ReplicaNode{
						NodeID:   fmt.Sprintf("replica-%d", i),
						BaseURL:  addr,
						Healthy:  true,
						Priority: i,
					})
				}
			}
		}

		if len(replicas) > 0 {
			logger.Info("configured replicas", "count", len(replicas))
			store.replicas = replicas

			// Create replication manager
			replConfig := DefaultReplicationConfig()
			if mode := os.Getenv("REPLICATION_MODE"); mode != "" {
				switch mode {
				case "sync":
					replConfig.Mode = SyncReplication
				case "strong":
					replConfig.Mode = StrongConsistency
				default:
					replConfig.Mode = AsyncReplication
				}
			}

			// Initialize metrics collector (nil for now, could be added later)
			store.replicationMgr = NewReplicationManager(replConfig, nil)
			logger.Info("replication manager initialized", "mode", replConfig.Mode.String())

			// Setup replication hook
			store.SetWALHook(func(e walEntry) {
				// Convert to replication WALEntry
				replEntry := &WALEntry{
					Op:     e.Op,
					ID:     e.ID,
					Doc:    e.Doc,
					Meta:   e.Meta,
					Vec:    e.Vec,
					Coll:   e.Coll,
					Tenant: e.Tenant,
					Time:   e.Time,
				}

				// Replicate (use shardID=0 for standalone)
				if err := store.replicationMgr.ReplicateEntry(replEntry, store.replicas, 0); err != nil {
					logger.Error("replication failed", "error", err, "op", e.Op, "id", e.ID)
				}
			})
		} else {
			logger.Warn("replication enabled but no replicas configured")
		}
	}

	if !loaded {
		// Make hydration count configurable for low-memory deployments
		hydrationCount := 100 // Reduced from 100000 for low-memory deployment
		if envHydrate := os.Getenv("HYDRATION_COUNT"); envHydrate != "" {
			if v, err := strconv.Atoi(envHydrate); err == nil && v >= 0 {
				hydrationCount = v
			}
		}

		logger.Info("hydrating index", "count", hydrationCount)
		for i := 0; i < hydrationCount; i++ {
			vec, _ := embedder.Embed(fmt.Sprintf("doc-%d", i))
			if _, err := store.Add(vec, fmt.Sprintf("doc-%d content about memory locality and vector search.", i), "", nil, "default", ""); err != nil {
				logger.Warn("failed to add vector", "index", i, "error", err)
			}
		}
		if err := store.Save(indexPath); err != nil {
			logger.Warn("failed to save index", "error", err)
		}
	}
	logger.Info("index ready", "vectors", store.Count, "ram_contiguous", true)

	sched := core.NewScheduler(core.DefaultSchedulerConfig)
	defer sched.Shutdown()
	go sched.Run()

	retrievalTool := &RetrievalTool{Store: store, Embedder: embedder}
	retrievalID := sched.Tools().Register(retrievalTool, core.ToolPolicy{
		MaxRetries:     0,
		DefaultTimeout: 200 * time.Millisecond,
	}, nil)

	reranker := initReranker(embedder)
	rerankTool := &RerankTool{Reranker: reranker}
	rerankID := sched.Tools().Register(rerankTool, core.ToolPolicy{
		MaxRetries:     0,
		DefaultTimeout: 200 * time.Millisecond,
	}, nil)

	// Model warmup (best-effort)
	warmupModels(embedder, reranker)

	router := tools.NewModelRouter()
	llmCfg, err := router.FastestModel()
	if err != nil {
		logger.Warn("no api key found, llm calls may fail")
	}
	llmID := sched.Tools().Register(tools.NewLLMTool(llmCfg), core.DefaultToolPolicy, nil)

	agent := NewRAGAgent(llmID, retrievalID, rerankID)
	agentID := sched.Agents().Register(agent, nil)

	// HTTP API with graceful shutdown
	handler := newHTTPHandler(store, embedder, reranker, indexPath)
	addr := ":8080"
	srv := &http.Server{
		Addr:    addr,
		Handler: handler,
	}

	go func() {
		logger.Info("http api listening", "addr", addr, "endpoints", "POST /insert, POST /batch_insert, POST /query, POST /delete, GET /health, GET /metrics")
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Error("http server error", "error", err)
		}
	}()

	// Background compaction/GC with auto-triggering based on tombstone ratio
	go func() {
		interval := time.Duration(envInt("COMPACT_INTERVAL_MIN", 60)) * time.Minute
		tombstoneThreshold := float64(envInt("COMPACT_TOMBSTONE_THRESHOLD", 10)) / 100.0 // Default 10%

		t := time.NewTicker(interval)
		defer t.Stop()
		for range t.C {
			store.RLock()
			total := store.Count
			deleted := len(store.Deleted)
			store.RUnlock()

			if total == 0 {
				continue
			}

			deletedRatio := float64(deleted) / float64(total)
			if deletedRatio >= tombstoneThreshold {
				logger.Info("auto-compaction triggered", "deleted_ratio", deletedRatio, "deleted", deleted, "total", total)
				if err := store.Compact(indexPath); err != nil {
					logger.Error("compact error", "error", err)
				} else {
					logger.Info("compact completed", "purged", deleted)
				}
			}
		}
	}()

	// Optional snapshot export scheduler
	go func() {
		exportPath := os.Getenv("SNAPSHOT_EXPORT_PATH")
		if exportPath == "" {
			return
		}
		interval := time.Duration(envInt("EXPORT_INTERVAL_MIN", 0)) * time.Minute
		if interval <= 0 {
			return
		}
		t := time.NewTicker(interval)
		defer t.Stop()
		for range t.C {
			if err := store.Save(exportPath); err != nil {
				logger.Error("snapshot export error", "error", err)
			} else {
				logger.Info("snapshot exported", "path", exportPath)
			}
		}
	}()

	// Setup graceful shutdown
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGTERM, syscall.SIGINT)

	// Check if server should run indefinitely (no demo)
	serverOnly := os.Getenv("SERVER_ONLY") == "1"

	if !serverOnly {
		// Start optional RAG conversation (demo mode)
		fmt.Println(">>> Starting RAG Conversation...")
		convID := sched.StartConversation(agentID, "Why is memory locality important for vector search?")

		ticker := time.NewTicker(500 * time.Millisecond)
		timeout := time.After(10 * time.Second)
		conversationDone := false

		for !conversationDone {
			select {
			case sig := <-sigCh:
				fmt.Printf("\n>>> Received signal %v, initiating graceful shutdown...\n", sig)
				conversationDone = true
			case <-timeout:
				fmt.Println("Timeout reached (Demo end)")
				conversationDone = true
			case <-ticker.C:
				if conv, ok := sched.ConvMgr().Get(convID); ok && conv.State == core.ConvComplete {
					fmt.Println(">>> Conversation Marked Complete by Agent.")
					conversationDone = true
				}
			}
		}
	} else {
		// Server-only mode: wait for signal indefinitely
		fmt.Println(">>> Server running in HTTP-only mode. Press Ctrl+C to stop.")
		sig := <-sigCh
		fmt.Printf("\n>>> Received signal %v, initiating graceful shutdown...\n", sig)
	}

	// Graceful shutdown sequence
	fmt.Println(">>> Shutting down HTTP server...")
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		fmt.Printf("HTTP server shutdown error: %v\n", err)
	}

	fmt.Println(">>> Saving final snapshot...")
	if err := store.Save(indexPath); err != nil {
		logger.Error("failed to save final snapshot", "error", err)
	} else {
		logger.Info("final snapshot saved successfully")
	}

	logger.Info("shutdown complete")
}

func envInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func envInt64(key string, def int64) int64 {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil && n > 0 {
			return n
		}
	}
	return def
}

func fileSize(path string) int64 {
	if path == "" {
		return 0
	}
	info, err := os.Stat(path)
	if err != nil {
		return 0
	}
	return info.Size()
}
