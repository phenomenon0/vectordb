package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"hash/fnv"
	"io"
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
	"sync/atomic"
	"syscall"
	"time"
	"unicode"

	"github.com/phenomenon0/vectordb/internal/index"
	"github.com/phenomenon0/vectordb/internal/logging"
	"github.com/phenomenon0/vectordb/internal/obsidian"
	"github.com/phenomenon0/vectordb/internal/security"
	"github.com/phenomenon0/vectordb/internal/storage"
	"github.com/phenomenon0/vectordb/internal/telemetry"

	"golang.org/x/net/http2"
	"golang.org/x/net/http2/h2c"

	deepdatav1 "github.com/phenomenon0/vectordb/api/gen/deepdata/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"

	"net"
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
	// Index abstraction - the single source of truth for vector search
	indexes     map[string]index.Index // Collection -> Index mapping
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
	TenantID    map[uint64]string     // vector hash -> tenant ID
	acl         *security.ACL         // access control lists
	quotas      *security.TenantQuota // storage quotas per tenant
	tenantRL    *tenantRateLimiter    // per-tenant rate limiting
	jwtMgr      *security.JWTManager  // JWT token manager
	requireAuth bool                  // Require JWT authentication
	// Storage format (gob, cowrie, cowrie-zstd)
	storageFormat storage.Format
	// Metadata bitmap index for fast pre-filtering
	// Metadata bitmap index for fast pre-filtering
	metaIndex *MetadataIndex

	// Recovery state
	walReplayError error // Non-nil if WAL replay failed during load

	// Background goroutine lifecycle
	bgWg             sync.WaitGroup // Tracks in-flight background snapshot goroutines
	snapshotRunning  atomic.Bool    // Guards against concurrent background snapshots racing on .tmp file

	// Limits (configurable via env vars)
	maxCollections int // MAX_COLLECTIONS (default 10,000)
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
		logging.Default().Error("failed to create default HNSW index", "error", err)
		os.Exit(1)
	}

	// Initialize JWT manager if configured
	var jwtMgr *security.JWTManager
	if secret := os.Getenv("JWT_SECRET"); secret != "" {
		issuer := os.Getenv("JWT_ISSUER")
		if issuer == "" {
			issuer = "vectordb"
		}
		jwtMgr = security.NewJWTManager(secret, issuer)
	}

	// Select storage format (default: gob for backward compatibility)
	// Options: "gob", "cowrie", "cowrie-zstd"
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
		acl:      security.NewACL(),
		quotas:   security.NewTenantQuota(),
		tenantRL: newTenantRateLimiter(envInt("TENANT_RPS", 100), envInt("TENANT_BURST", 100), envInt("MAX_TENANTS", 100_000), time.Minute),
		jwtMgr:   jwtMgr,
		// Storage
		storageFormat: storageFormat,
		// Metadata index for fast pre-filtering
		metaIndex: NewMetadataIndex(),
		// Limits
		maxCollections: envInt("MAX_COLLECTIONS", 10_000),
	}
}

func (vs *VectorStore) resolveCollectionIndexLocked(collection string) (string, index.Index, bool, error) {
	if collection == "" {
		collection = "default"
	}

	if idx, ok := vs.indexes[collection]; ok && idx != nil {
		return collection, idx, false, nil
	}

	if collection == "default" {
		idx := vs.indexes["default"]
		if idx == nil {
			return "", nil, false, fmt.Errorf("default index not initialized")
		}
		return collection, idx, false, nil
	}

	if vs.maxCollections > 0 && len(vs.indexes) >= vs.maxCollections {
		return "", nil, false, fmt.Errorf("collection limit exceeded: maximum %d collections allowed (set MAX_COLLECTIONS to increase)", vs.maxCollections)
	}

	cfg := loadHNSWConfig()
	newIdx, err := index.NewHNSWIndex(vs.Dim, map[string]interface{}{
		"m":         cfg.M,
		"ml":        cfg.Ml,
		"ef_search": cfg.EfSearch,
	})
	if err != nil {
		return "", nil, false, fmt.Errorf("failed to create collection index: %w", err)
	}

	vs.indexes[collection] = newIdx
	return collection, newIdx, true, nil
}

func (vs *VectorStore) commitNewVectorLocked(v []float32, doc, id string, meta map[string]string, collection, tenantID string, autoGenerated bool) {
	vs.Data = append(vs.Data, v...)
	vs.Docs = append(vs.Docs, doc)
	vs.IDs = append(vs.IDs, id)
	vs.Seqs = append(vs.Seqs, uint64(vs.Count))
	vs.Count++
	if autoGenerated {
		vs.next++
	}

	hid := hashID(id)
	vs.ingestLex(hid, tokenize(doc))
	vs.idToIx[hid] = vs.Count - 1
	if meta != nil {
		vs.Meta[hid] = meta
		vs.ingestMeta(hid, meta)
	}
	vs.Coll[hid] = collection
	vs.TenantID[hid] = tenantID
	delete(vs.Deleted, hid)
}

func (vs *VectorStore) rollbackNewVectorLocked(idx index.Index, collection string, createdIndex bool, hid uint64, tenantID string, totalBytes int64) {
	if idx != nil {
		if err := idx.Delete(context.Background(), hid); err != nil && !isNotFoundError(err) {
			logging.Default().Warn("failed to rollback index insert", "hid", hid, "error", err)
		}
	}
	if createdIndex {
		delete(vs.indexes, collection)
	}
	vs.quotas.RemoveUsage(tenantID, totalBytes, 1)
}

func (vs *VectorStore) addNewLocked(v []float32, doc, id string, meta map[string]string, collection string, tenantID string, autoGenerated bool) (string, error) {
	vectorBytes := int64(len(v) * 4)
	docBytes := int64(len(doc))
	totalBytes := vectorBytes + docBytes
	if err := vs.quotas.AddUsage(tenantID, totalBytes, 1); err != nil {
		return "", fmt.Errorf("quota check failed: %w", err)
	}

	collection, idx, createdIndex, err := vs.resolveCollectionIndexLocked(collection)
	if err != nil {
		vs.quotas.RemoveUsage(tenantID, totalBytes, 1)
		return "", err
	}

	hid := hashID(id)
	if err := idx.Add(context.Background(), hid, v); err != nil {
		vs.rollbackNewVectorLocked(idx, collection, createdIndex, hid, tenantID, totalBytes)
		return "", fmt.Errorf("failed to add vector to index: %w", err)
	}

	if err := vs.appendWAL("insert", id, doc, meta, v, collection, tenantID); err != nil {
		vs.rollbackNewVectorLocked(idx, collection, createdIndex, hid, tenantID, totalBytes)
		return "", fmt.Errorf("failed to append to WAL: %w", err)
	}

	vs.commitNewVectorLocked(v, doc, id, meta, collection, tenantID, autoGenerated)
	return id, nil
}

// Add appends a vector/doc/meta with tenant ownership.
func (vs *VectorStore) Add(v []float32, doc string, id string, meta map[string]string, collection string, tenantID string) (string, error) {
	vs.Lock()
	defer vs.Unlock()
	if len(v) != vs.Dim {
		return "", fmt.Errorf("dimension mismatch: expected %d, got %d", vs.Dim, len(v))
	}
	autoGenerated := false
	if id == "" {
		id = fmt.Sprintf("doc-%d", vs.next)
		autoGenerated = true
	}
	if tenantID == "" {
		tenantID = "default" // Default tenant for backward compatibility
	}

	return vs.addNewLocked(v, doc, id, meta, collection, tenantID, autoGenerated)
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

		updateCollection := vs.Coll[hid]
		if updateCollection == "" {
			updateCollection = "default"
		}
		if collection != "" {
			updateCollection = collection
		}

		targetCollection, targetIdx, createdIndex, err := vs.resolveCollectionIndexLocked(updateCollection)
		if err != nil {
			return "", err
		}

		currentCollection := vs.Coll[hid]
		if currentCollection == "" {
			currentCollection = "default"
		}
		currentIdx, ok := vs.indexes[currentCollection]
		if !ok || currentIdx == nil {
			currentIdx = vs.indexes["default"]
			currentCollection = "default"
		}
		if currentIdx == nil {
			return "", fmt.Errorf("current index not initialized")
		}

		oldVec := append([]float32(nil), vs.Data[ix*vs.Dim:(ix+1)*vs.Dim]...)
		deleteErr := currentIdx.Delete(context.Background(), hid)
		if deleteErr != nil && !isNotFoundError(deleteErr) {
			if createdIndex {
				delete(vs.indexes, targetCollection)
			}
			return "", fmt.Errorf("failed to delete vector from current index: %w", deleteErr)
		}

		restoreCurrent := func() {
			if err := currentIdx.Add(context.Background(), hid, oldVec); err != nil {
				logging.Default().Warn("failed to restore index entry", "id", id, "error", err)
			}
		}

		if err := targetIdx.Add(context.Background(), hid, v); err != nil {
			if createdIndex {
				delete(vs.indexes, targetCollection)
			}
			if targetIdx != currentIdx {
				restoreCurrent()
			} else if deleteErr == nil || isNotFoundError(deleteErr) {
				restoreCurrent()
			}
			return "", fmt.Errorf("failed to update vector in index: %w", err)
		}

		if err := vs.appendWAL("upsert", id, doc, meta, v, targetCollection, tenantID); err != nil {
			if errDel := targetIdx.Delete(context.Background(), hid); errDel != nil && !isNotFoundError(errDel) {
				logging.Default().Warn("failed to rollback updated index entry", "id", id, "error", errDel)
			}
			if createdIndex {
				delete(vs.indexes, targetCollection)
			}
			restoreCurrent()
			return "", fmt.Errorf("failed to append to WAL: %w", err)
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
		}
		vs.Coll[hid] = targetCollection
		vs.TenantID[hid] = tenantID
		delete(vs.Deleted, hid)
		return id, nil
	}

	return vs.addNewLocked(v, doc, id, meta, collection, tenantID, false)
}

// isNotFoundError checks if an error is a "not found" type error that can be safely ignored
func isNotFoundError(err error) bool {
	if err == nil {
		return false
	}
	errStr := strings.ToLower(err.Error())
	return strings.Contains(errStr, "not found") ||
		strings.Contains(errStr, "not exist") ||
		strings.Contains(errStr, "does not exist")
}

func (vs *VectorStore) Delete(id string) error {
	vs.Lock()
	defer vs.Unlock()
	hid := hashID(id)
	tenant := vs.TenantID[hid]
	if tenant == "" {
		tenant = "default"
	}

	// Delete from index abstraction
	delCollection := vs.Coll[hid]
	if delCollection == "" {
		delCollection = "default"
	}
	if idx, ok := vs.indexes[delCollection]; ok && idx != nil {
		if err := idx.Delete(context.Background(), hid); err != nil && !isNotFoundError(err) {
			logging.Default().Warn("failed to delete from index", "id", id, "collection", delCollection, "error", err)
		}
	} else if idx := vs.indexes["default"]; idx != nil {
		if err := idx.Delete(context.Background(), hid); err != nil && !isNotFoundError(err) {
			logging.Default().Warn("failed to delete from default index", "id", id, "error", err)
		}
	}

	vs.Deleted[hid] = true
	vs.ejectLex(hid)
	vs.ejectMeta(hid)
	delete(vs.Meta, hid)
	delete(vs.Coll, hid)
	if err := vs.appendWAL("delete", id, "", nil, nil, delCollection, tenant); err != nil {
		return fmt.Errorf("failed to append to WAL: %w", err)
	}
	return nil
}

func (vs *VectorStore) Get(index int) []float32 {
	if index < 0 || index >= vs.Count {
		return nil
	}
	offset := index * vs.Dim
	end := offset + vs.Dim
	if end > len(vs.Data) {
		return nil
	}
	return vs.Data[offset:end]
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
	return vs.SearchScan(query, k, "")
}

// SearchScan performs brute-force scan, optionally filtering by collection.
func (vs *VectorStore) SearchScan(query []float32, k int, collection string) []int {
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
		// Collection pre-filter during scan to avoid post-filter dropping results
		if collection != "" && vs.Coll[hid] != collection {
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

// SearchANN performs ANN search via the index abstraction.
// For backward compatibility, searches the "default" collection with default params.
func (vs *VectorStore) SearchANN(query []float32, k int) []int {
	return vs.SearchANNWithParams(query, k, "", 0)
}

// SearchANNWithParams performs ANN search with collection and efSearch parameters.
// If collection is empty, searches the "default" collection.
// If efSearch is 0, uses the index's default ef_search value.
func (vs *VectorStore) SearchANNWithParams(query []float32, k int, collection string, efSearch int) []int {
	vs.RLock()
	defer vs.RUnlock()

	if collection == "" {
		collection = "default"
	}

	// Find the appropriate index
	idx, ok := vs.indexes[collection]
	if !ok {
		idx = vs.indexes["default"]
	}
	if idx == nil {
		return nil
	}

	// Build search params
	var params index.SearchParams = index.DefaultSearchParams{}
	if efSearch > 0 {
		params = index.HNSWSearchParams{EfSearch: efSearch}
	}

	results, err := idx.Search(context.Background(), query, k, params)
	if err != nil {
		return nil
	}

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

	tmp := path + ".tmp"
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		vs.RUnlock()
		return err
	}
	f, err := os.Create(tmp)
	if err != nil {
		vs.RUnlock()
		return err
	}

	// Export indexes
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
		HNSW:      nil,       // Legacy field - no longer written
		Indexes:   indexData, // Primary index storage
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
		_ = f.Close()
		_ = os.Remove(tmp)
		vs.RUnlock()
		return err
	}
	// Sync to durable storage before closing — without this, os.Rename could
	// succeed but the file contents may not be persisted on power loss, leaving
	// a corrupt or zero-length snapshot after the WAL has already been removed.
	if err := f.Sync(); err != nil {
		_ = f.Close()
		_ = os.Remove(tmp)
		vs.RUnlock()
		return err
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(tmp)
		vs.RUnlock()
		return err
	}

	// Capture snapshot metadata while under read lock, then update mutable fields under write lock.
	newChecksum := vs.computeChecksum()
	lastSaved := payload.LastSaved
	vs.RUnlock()

	vs.Lock()
	vs.checksum = newChecksum
	vs.lastSaved = lastSaved
	vs.Unlock()

	// Rename atomically commits the new snapshot.
	// WAL cleanup is the caller's responsibility — Save() must not delete the
	// WAL because background snapshot goroutines race with concurrent appendWAL
	// writers: entries written after the RLock was released but before this point
	// would be lost if we deleted the WAL here.
	if err := os.Rename(tmp, path); err != nil {
		return err
	}
	return nil
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

	// Try cowrie formats
	for _, formatName := range []string{"cowrie", "cowrie-zstd"} {
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
			logging.Default().Warn("failed to load index with any format, rebuilding")
			return NewVectorStore(capacity, dim), false
		}
		_ = loadedFormat // format used for loading (for logging if needed)
		// Initialize JWT manager if configured
		var jwtMgr *security.JWTManager
		if secret := os.Getenv("JWT_SECRET"); secret != "" {
			issuer := os.Getenv("JWT_ISSUER")
			if issuer == "" {
				issuer = "vectordb"
			}
			jwtMgr = security.NewJWTManager(secret, issuer)
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
			indexes:     make(map[string]index.Index),
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
			// Numeric/time index maps for range queries (must be initialized!)
			numIndex:  make(map[string][]numEntry),
			timeIndex: make(map[string][]timeEntry),
			// Multi-tenancy support (TenantID already set from payload above)
			acl:      security.NewACL(),
			quotas:   security.NewTenantQuota(),
			tenantRL: newTenantRateLimiter(envInt("TENANT_RPS", 100), envInt("TENANT_BURST", 100), envInt("MAX_TENANTS", 100_000), time.Minute),
			jwtMgr:   jwtMgr,
			// Storage format
			storageFormat: getStorageFormat(),
			// Metadata index (rebuilt below)
			metaIndex: NewMetadataIndex(),
		}
		if vs.checksum == "" {
			vs.checksum = vs.computeChecksum()
		}
		// Checksum migration: if stored checksum was computed with an older formula, recompute.
		if !vs.validateChecksum() {
			oldFormula := fmt.Sprintf("%x", hashID(fmt.Sprintf("%d-%d", payload.Count, payload.Next)))
			if vs.checksum == oldFormula {
				logging.Default().Info("migrating checksum from old formula")
				vs.checksum = vs.computeChecksum()
			} else {
				logging.Default().Warn("checksum mismatch; continuing with loaded snapshot")
			}
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
		// Import indexes from snapshot
		if len(payload.Indexes) > 0 {
			for collName, data := range payload.Indexes {
				// Create HNSW index for this collection
				idx, err := index.NewHNSWIndex(vs.Dim, nil)
				if err != nil {
					logging.Default().Warn("failed to create index for collection", "collection", collName, "error", err)
					continue
				}
				if err := idx.Import(data); err != nil {
					logging.Default().Warn("failed to import index for collection", "collection", collName, "error", err)
					continue
				}
				vs.indexes[collName] = idx
			}
		}

		// If no indexes were imported, create default index and populate from Data
		// This handles migration from legacy snapshots that only had HNSW data
		if len(vs.indexes) == 0 {
			cfg := loadHNSWConfig()
			defaultIdx, err := index.NewHNSWIndex(vs.Dim, map[string]interface{}{
				"m":         cfg.M,
				"ml":        cfg.Ml,
				"ef_search": cfg.EfSearch,
			})
			if err != nil {
				return nil, false
			}

			// Migrate all non-deleted vectors to new index
			migrated := 0
			for i, idStr := range vs.IDs {
				hid := hashID(idStr)
				if vs.Deleted[hid] {
					continue
				}
				vec := vs.Data[i*vs.Dim : (i+1)*vs.Dim]
				if err := defaultIdx.Add(context.Background(), hid, vec); err != nil {
					logging.Default().Warn("failed to migrate vector", "id", idStr, "error", err)
					continue
				}
				migrated++
			}
			vs.indexes["default"] = defaultIdx
			if migrated > 0 {
				logging.Default().Info("migrated vectors to index abstraction", "count", migrated)
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
			logging.Default().Info("rebuilt metadata index", "documents_indexed", vs.metaIndex.GetDocumentCount())
		}
		// Rebuild numeric/time range indexes from persisted NumMeta/TimeMeta.
		// Without this, range-filter queries return empty results after restart.
		for hid, nums := range vs.NumMeta {
			for k, v := range nums {
				vs.numIndex[k] = append(vs.numIndex[k], numEntry{ID: hid, V: v})
			}
		}
		for hid, times := range vs.TimeMeta {
			for k, t := range times {
				vs.timeIndex[k] = append(vs.timeIndex[k], timeEntry{ID: hid, T: t})
			}
		}
		if len(vs.NumMeta) > 0 || len(vs.TimeMeta) > 0 {
			logging.Default().Info("rebuilt range indexes", "numeric_docs", len(vs.NumMeta), "time_docs", len(vs.TimeMeta))
		}
		// Recover any frozen WAL left by a crash during background snapshot.
		// The frozen WAL contains entries from before the rotation — replay it
		// first since those entries are older than any entries in the current WAL.
		frozenPath := vs.walPath + ".frozen"
		if _, ferr := os.Stat(frozenPath); ferr == nil {
			logging.Default().Warn("found frozen WAL from interrupted snapshot, replaying", "path", frozenPath)
			savedPath := vs.walPath
			vs.walPath = frozenPath
			if err := replayWAL(vs); err != nil {
				logging.Default().Error("frozen WAL replay failed", "error", err)
				vs.walReplayError = err
			}
			vs.walPath = savedPath
		}
		if err := replayWAL(vs); err != nil {
			logging.Default().Error("WAL replay failed — some data may be lost", "error", err)
			vs.walReplayError = err // Store error for inspection
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
// Embedder/Reranker interfaces and hash fallback
// ======================================================================================

type Embedder interface {
	Embed(text string) ([]float32, error)      // encode as document (for indexing)
	EmbedQuery(text string) ([]float32, error)  // encode as query (for searching)
	Dim() int
}

// SwappableEmbedder wraps an Embedder and allows hot-swapping at runtime.
type SwappableEmbedder struct {
	mu    sync.RWMutex
	inner Embedder
}

func NewSwappableEmbedder(e Embedder) *SwappableEmbedder {
	return &SwappableEmbedder{inner: e}
}

func (s *SwappableEmbedder) Embed(text string) ([]float32, error) {
	s.mu.RLock()
	e := s.inner
	s.mu.RUnlock()
	return e.Embed(text)
}

func (s *SwappableEmbedder) EmbedQuery(text string) ([]float32, error) {
	s.mu.RLock()
	e := s.inner
	s.mu.RUnlock()
	return e.EmbedQuery(text)
}

func (s *SwappableEmbedder) Dim() int {
	s.mu.RLock()
	e := s.inner
	s.mu.RUnlock()
	return e.Dim()
}

func (s *SwappableEmbedder) Swap(e Embedder) {
	s.mu.Lock()
	s.inner = e
	s.mu.Unlock()
}

func (s *SwappableEmbedder) Inner() Embedder {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.inner
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

func (e *HashEmbedder) EmbedQuery(text string) ([]float32, error) { return e.Embed(text) }

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

func (e *OpenAIEmbedder) EmbedQuery(text string) ([]float32, error) { return e.Embed(text) }

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

// OllamaEmbedder uses Ollama's local embedding models.
// Supports prefix-based asymmetry for nomic-embed-text models.
type OllamaEmbedder struct {
	baseURL string
	model   string
	dim     int
	isNomic bool // nomic models use "search_query: " / "search_document: " prefixes
	client  *http.Client
}

func NewOllamaEmbedder(baseURL, model string) *OllamaEmbedder {
	// nomic-embed-text produces 768-dim vectors
	dim := 768
	if model == "granite-embedding" {
		dim = 384
	}
	isNomic := strings.Contains(model, "nomic")
	return &OllamaEmbedder{
		baseURL: baseURL,
		model:   model,
		dim:     dim,
		isNomic: isNomic,
		client:  &http.Client{Timeout: 60 * time.Second},
	}
}

func (e *OllamaEmbedder) Dim() int { return e.dim }

// MaxChunkChars is the max characters per chunk (~2000 tokens for safety)
const MaxChunkChars = 6000

func (e *OllamaEmbedder) Embed(text string) ([]float32, error) {
	if e.isNomic {
		return e.embedWithPrefix(text, "search_document: ")
	}
	return e.embedWithPrefix(text, "")
}

func (e *OllamaEmbedder) EmbedQuery(text string) ([]float32, error) {
	if e.isNomic {
		return e.embedWithPrefix(text, "search_query: ")
	}
	return e.embedWithPrefix(text, "")
}

func (e *OllamaEmbedder) embedWithPrefix(text, prefix string) ([]float32, error) {
	if text == "" {
		text = "empty"
	}

	// If text is short enough, embed directly
	if len(text) <= MaxChunkChars {
		return e.embedSingle(prefix + text)
	}

	// Long text: chunk and average embeddings (late chunking / pooling)
	chunks := smartChunk(text, MaxChunkChars, 200) // 200 char overlap
	if len(chunks) == 0 {
		return e.embedSingle(prefix + text[:MaxChunkChars])
	}

	// Embed each chunk and compute weighted average
	var allVecs [][]float32
	var weights []float32
	for _, chunk := range chunks {
		vec, err := e.embedSingle(prefix + chunk)
		if err != nil {
			continue // Skip failed chunks
		}
		allVecs = append(allVecs, vec)
		// Weight by chunk length (longer chunks = more important)
		weights = append(weights, float32(len(chunk)))
	}

	if len(allVecs) == 0 {
		return nil, fmt.Errorf("all chunks failed to embed")
	}

	// Weighted average pooling
	return weightedAverageVecs(allVecs, weights), nil
}

// embedSingle embeds a single chunk of text
func (e *OllamaEmbedder) embedSingle(text string) ([]float32, error) {
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
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Ollama API error %d: %s", resp.StatusCode, string(body))
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
	if e.dim == 0 {
		e.dim = len(vec)
	}
	return vec, nil
}

// smartChunk splits text into semantic chunks with overlap
// Uses sentence boundaries for cleaner splits
func smartChunk(text string, maxChars, overlap int) []string {
	if len(text) <= maxChars {
		return []string{text}
	}

	var chunks []string

	// Split into sentences first (approximate)
	sentences := splitSentences(text)

	var currentChunk strings.Builder
	var currentLen int

	for _, sentence := range sentences {
		sentLen := len(sentence)

		// If single sentence exceeds max, split it by words
		if sentLen > maxChars {
			// Flush current chunk
			if currentLen > 0 {
				chunks = append(chunks, strings.TrimSpace(currentChunk.String()))
				currentChunk.Reset()
				currentLen = 0
			}
			// Split long sentence by words
			wordChunks := splitByWords(sentence, maxChars, overlap)
			chunks = append(chunks, wordChunks...)
			continue
		}

		// Check if adding this sentence exceeds limit
		if currentLen+sentLen > maxChars && currentLen > 0 {
			// Save current chunk
			chunkText := strings.TrimSpace(currentChunk.String())
			chunks = append(chunks, chunkText)

			// Start new chunk with overlap from end of previous
			currentChunk.Reset()
			if overlap > 0 && len(chunkText) > overlap {
				// Get last N chars for overlap
				overlapText := chunkText[len(chunkText)-overlap:]
				// Try to start at word boundary
				if idx := strings.LastIndex(overlapText, " "); idx > 0 {
					overlapText = overlapText[idx+1:]
				}
				currentChunk.WriteString(overlapText)
				currentChunk.WriteString(" ")
				currentLen = len(overlapText) + 1
			} else {
				currentLen = 0
			}
		}

		currentChunk.WriteString(sentence)
		currentChunk.WriteString(" ")
		currentLen += sentLen + 1
	}

	// Don't forget the last chunk
	if currentLen > 0 {
		chunks = append(chunks, strings.TrimSpace(currentChunk.String()))
	}

	return chunks
}

// splitSentences splits text into sentences using common delimiters
func splitSentences(text string) []string {
	var sentences []string
	var current strings.Builder

	runes := []rune(text)
	for i, r := range runes {
		current.WriteRune(r)

		// Check for sentence end
		if r == '.' || r == '!' || r == '?' || r == '\n' {
			// Look ahead to confirm (avoid splitting on abbreviations like "Dr.")
			isEnd := true
			if r == '.' && i+1 < len(runes) {
				next := runes[i+1]
				// Not a sentence end if followed by lowercase or digit
				if (next >= 'a' && next <= 'z') || (next >= '0' && next <= '9') {
					isEnd = false
				}
			}
			if isEnd {
				s := strings.TrimSpace(current.String())
				if len(s) > 0 {
					sentences = append(sentences, s)
				}
				current.Reset()
			}
		}
	}

	// Remaining text
	if current.Len() > 0 {
		s := strings.TrimSpace(current.String())
		if len(s) > 0 {
			sentences = append(sentences, s)
		}
	}

	return sentences
}

// splitByWords splits text by words when sentences are too long
func splitByWords(text string, maxChars, overlap int) []string {
	words := strings.Fields(text)
	if len(words) == 0 {
		return nil
	}

	var chunks []string
	var current strings.Builder
	currentLen := 0

	for _, word := range words {
		wordLen := len(word)
		if currentLen+wordLen+1 > maxChars && currentLen > 0 {
			chunks = append(chunks, strings.TrimSpace(current.String()))
			current.Reset()
			currentLen = 0
		}
		if currentLen > 0 {
			current.WriteString(" ")
			currentLen++
		}
		current.WriteString(word)
		currentLen += wordLen
	}

	if currentLen > 0 {
		chunks = append(chunks, strings.TrimSpace(current.String()))
	}

	return chunks
}

// weightedAverageVecs computes weighted average of vectors and normalizes
func weightedAverageVecs(vecs [][]float32, weights []float32) []float32 {
	if len(vecs) == 0 {
		return nil
	}
	if len(vecs) == 1 {
		return vecs[0]
	}

	dim := len(vecs[0])
	result := make([]float32, dim)

	// Normalize weights
	var totalWeight float32
	for _, w := range weights {
		totalWeight += w
	}
	if totalWeight == 0 {
		totalWeight = 1
	}

	// Weighted sum
	for i, vec := range vecs {
		w := weights[i] / totalWeight
		for j, v := range vec {
			result[j] += v * w
		}
	}

	// L2 normalize the result
	var norm float32
	for _, v := range result {
		norm += v * v
	}
	norm = float32(math.Sqrt(float64(norm)))
	if norm > 0 {
		for i := range result {
			result[i] /= norm
		}
	}

	return result
}

type SimpleReranker struct {
	Embedder Embedder
}

func (r *SimpleReranker) Rerank(query string, docs []string, topK int) ([]string, []float32, string, error) {
	qVec, err := r.Embedder.EmbedQuery(query)
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

	order := make([]int, len(bestScores))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(i, j int) bool {
		return bestScores[order[i]] > bestScores[order[j]]
	})

	sortedDocs := make([]string, 0, len(order))
	sortedScores := make([]float32, 0, len(order))
	for _, idx := range order {
		sortedDocs = append(sortedDocs, bestDocs[idx])
		sortedScores = append(sortedScores, bestScores[idx])
	}

	return sortedDocs, sortedScores, "Simple rerank", nil
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

type RangeFilter struct {
	Key     string   `json:"key"`
	Min     *float64 `json:"min,omitempty"`
	Max     *float64 `json:"max,omitempty"`
	TimeMin string   `json:"time_min,omitempty"`
	TimeMax string   `json:"time_max,omitempty"`
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

func (vs *VectorStore) appendWAL(op, id, doc string, meta map[string]string, vec []float32, collection string, tenant string) error {
	// Normalize tenant
	if tenant == "" && id != "" {
		tenant = vs.TenantID[hashID(id)]
	}
	if tenant == "" {
		tenant = "default"
	}
	if collection == "" && id != "" {
		collection = vs.Coll[hashID(id)]
	}
	if collection == "" {
		collection = "default"
	}

	entry := walEntry{
		Op:     op,
		ID:     id,
		Doc:    doc,
		Meta:   meta,
		Vec:    vec,
		Coll:   collection,
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
		if err := f.Close(); err != nil {
			logging.Default().Warn("failed to close WAL file", "error", err)
		}
		vs.walMu.Unlock()

		if doSnapshot && vs.snapshotRunning.CompareAndSwap(false, true) {
			snapPath := strings.TrimSuffix(vs.walPath, ".wal")
			// Rotate the WAL under walMu: rename the current WAL to a frozen
			// path so the snapshot goroutine can safely delete it after Save()
			// without racing with concurrent appendWAL writers — new writes
			// will create a fresh WAL file via O_CREATE|O_APPEND.
			frozenWAL := vs.walPath + ".frozen"
			vs.walMu.Lock()
			rotateErr := os.Rename(vs.walPath, frozenWAL)
			vs.walOps = 0
			vs.walMu.Unlock()
			if rotateErr != nil {
				logging.Default().Warn("failed to rotate WAL for snapshot", "error", rotateErr)
				vs.snapshotRunning.Store(false)
			} else {
				vs.bgWg.Add(1)
				go func() {
					defer vs.bgWg.Done()
					defer vs.snapshotRunning.Store(false)
					if err := vs.Save(snapPath); err == nil {
						_ = os.Remove(frozenWAL)
					} else {
						// Save failed — restore the frozen WAL so entries aren't lost.
						// If a new WAL was created concurrently, the rename will fail
						// and the frozen WAL stays for the next snapshot attempt.
						if renameErr := os.Rename(frozenWAL, vs.walPath); renameErr != nil {
							logging.Default().Warn("failed to restore frozen WAL", "error", renameErr, "frozen_path", frozenWAL)
						}
					}
				}()
			}
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
			if err != io.EOF {
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
		logging.Default().Error("WAL replay completed with errors — WAL preserved for manual inspection",
			"error_count", len(replayErrors), "wal_path", path)
		for _, e := range replayErrors {
			logging.Default().Warn("WAL replay error", "error", e)
		}
		// Return an error so walReplayError is set, /health and /readyz report
		// unhealthy, and the WAL is preserved for operator inspection/recovery.
		return fmt.Errorf("WAL replay had %d errors", len(replayErrors))
	}

	// All entries replayed successfully — safe to remove the WAL.
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
		logging.Default().Info("using OpenAI embedder", "model", "text-embedding-3-small")
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
			logging.Default().Info("using Ollama embedder", "model", ollamaModel)
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
			logging.Default().Info("using ONNX embedder", "model_path", modelPath)
			return emb
		}
		logging.Default().Warn("ONNX embedder init failed")
	}

	// Priority 4: Hash embedder (fallback, low quality)
	logging.Default().Info("using hash embedder (set OPENAI_API_KEY or start Ollama for better quality)")
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
			logging.Default().Info("using ONNX reranker", "model_path", modelPath)
			return rr
		}
		logging.Default().Warn("falling back to simple reranker (ONNX init failed)")
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
			logging.Default().Warn("embedder warmup failed", "error", err)
		}
	}
	if reranker != nil {
		if _, _, _, err := reranker.Rerank("warmup", []string{"warmup"}, 1); err != nil {
			logging.Default().Warn("reranker warmup failed", "error", err)
		}
	}
}

// ======================================================================================
// Main: bootstrap, HTTP API
// ======================================================================================

func main() {
	// CLI flag parsing — strip "serve" subcommand if present
	args := os.Args[1:]
	if len(args) > 0 && args[0] == "serve" {
		args = args[1:]
	}
	fs := flag.NewFlagSet("vectordb", flag.ExitOnError)
	flagPort := fs.String("port", "", "HTTP port (env: PORT)")
	flagMode := fs.String("mode", "", "Engine mode: local or pro (env: VECTORDB_MODE)")
	flagDataDir := fs.String("data-dir", "", "Data directory (env: VECTORDB_DATA_DIR)")
	flagDim := fs.String("dimension", "", "Embedding dimension (env: EMBED_DIM)")
	flagEmbedder := fs.String("embedder", "", "Embedder type: ollama, openai, hash (env: EMBEDDER_TYPE)")
	flagEmbModel := fs.String("embedder-model", "", "Embedder model name (env: OLLAMA_EMBED_MODEL)")
	flagEmbURL := fs.String("embedder-url", "", "Embedder URL (env: OLLAMA_URL)")

	fs.Parse(args)
	// Flags set env vars so downstream code works unchanged
	if *flagPort != "" {
		os.Setenv("PORT", *flagPort)
	}
	if *flagMode != "" {
		os.Setenv("VECTORDB_MODE", *flagMode)
	}
	if *flagDataDir != "" {
		os.Setenv("VECTORDB_DATA_DIR", *flagDataDir)
	}
	if *flagDim != "" {
		os.Setenv("EMBED_DIM", *flagDim)
	}
	if *flagEmbedder != "" {
		os.Setenv("EMBEDDER_TYPE", *flagEmbedder)
	}
	if *flagEmbModel != "" {
		os.Setenv("OLLAMA_EMBED_MODEL", *flagEmbModel)
	}
	if *flagEmbURL != "" {
		os.Setenv("OLLAMA_URL", *flagEmbURL)
	}

	// Initialize structured logging (JSON by default, LOG_FORMAT=text for dev)
	logConfig := logging.DefaultConfig()
	if os.Getenv("LOG_FORMAT") == "text" {
		logConfig.Format = "text"
	}
	switch os.Getenv("LOG_LEVEL") {
	case "debug":
		logConfig.Level = logging.LevelDebug
	case "warn":
		logConfig.Level = logging.LevelWarn
	case "error":
		logConfig.Level = logging.LevelError
	default:
		// "info" or unset → LevelInfo (already the default)
	}
	logger := logging.Init(logConfig)
	logger.Info("initializing vector engine")

	// ==========================================================================
	// Startup Config Validation — fail fast on invalid env var values
	// ==========================================================================
	if configErrs := validateEnvConfig(logger); len(configErrs) > 0 {
		for _, e := range configErrs {
			logger.Error("invalid configuration", "detail", e)
		}
		fmt.Fprintf(os.Stderr, "FATAL: %d configuration error(s) — fix the environment variables above and restart\n", len(configErrs))
		os.Exit(1)
	}

	// ==========================================================================
	// Mode System Initialization (LOCAL or PRO)
	// ==========================================================================
	modeConfig, err := LoadModeFromEnv()
	if err != nil {
		logger.Error("failed to load mode configuration", "error", err)
		os.Exit(1)
	}

	// Ensure data directory exists
	dataDir, err := EnsureDataDirectory(modeConfig.Mode)
	if err != nil {
		logger.Error("failed to create data directory", "error", err)
		os.Exit(1)
	}
	logger.Info("data directory ready", "path", dataDir)

	// Initialize cost tracker (only for PRO mode)
	costTracker, err := NewCostTracker(modeConfig.Mode)
	if err != nil {
		logger.Warn("failed to initialize cost tracker", "error", err)
	}
	if costTracker != nil {
		defer costTracker.Close()
		logger.Info("cost tracking enabled", "db", GetCostDBPath(modeConfig.Mode))
	}

	// Use mode-specific index path
	indexPath := GetIndexPath(modeConfig.Mode)

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

	// Initialize embedder based on mode (LOCAL: ONNX, PRO: OpenAI)
	var embedder Embedder
	if os.Getenv("USE_HASH_EMBEDDER") == "1" {
		logger.Info("using hash embedder (low-memory mode)")
		embedder = NewHashEmbedder(modeConfig.Dimension)
	} else {
		// Use mode-aware embedder factory
		embedder, err = InitEmbedderForMode(modeConfig, costTracker)
		if err != nil {
			logger.Error("failed to initialize embedder", "error", err)
			// Fall back to hash embedder — use modeConfig.Dimension which
			// InitEmbedderForMode may have updated before failing
			logger.Warn("falling back to hash embedder")
			embedder = NewHashEmbedder(modeConfig.Dimension)
		}
	}

	// Print mode banner (after embedder init so it reflects actual config)
	PrintModeBanner(modeConfig)

	// Make initial capacity configurable for low-memory deployments
	initialCapacity := 1000 // Reduced from 100000 for low-memory deployment
	if envCap := os.Getenv("VECTOR_CAPACITY"); envCap != "" {
		if v, err := strconv.Atoi(envCap); err == nil && v >= 0 {
			initialCapacity = v
		}
	}
	// Wrap in SwappableEmbedder so we can hot-swap at runtime via API
	swappableEmbedder := NewSwappableEmbedder(embedder)
	store, loaded := loadOrInitStore(indexPath, initialCapacity, swappableEmbedder.Dim())
	store.walMaxBytes = envInt64("WAL_MAX_BYTES", 5*1024*1024)
	store.walMaxOps = envInt("WAL_MAX_OPS", 1000)

	if !loaded {
		logger.Info("fresh index initialized", "capacity", initialCapacity, "dimension", swappableEmbedder.Dim())
	}
	logger.Info("index ready", "vectors", store.Count, "ram_contiguous", true)

	reranker := initReranker(swappableEmbedder)
	warmupModels(swappableEmbedder, reranker)

	// HTTP API with graceful shutdown
	handler, collectionHTTP := newHTTPHandler(store, swappableEmbedder, reranker, indexPath)
	addr := fmt.Sprintf(":%d", envInt("PORT", 8080))

	// Wrap handler with h2c (HTTP/2 cleartext) for connection multiplexing
	// without TLS. HTTP/1.1 clients continue to work transparently.
	// Set HTTP_H2C=0 to disable.
	var finalHandler http.Handler = handler
	if os.Getenv("HTTP_H2C") != "0" {
		finalHandler = h2c.NewHandler(handler, &http2.Server{})
	}

	srv := &http.Server{
		Addr:              addr,
		Handler:           finalHandler,
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       time.Duration(envInt("HTTP_READ_TIMEOUT_SEC", 60)) * time.Second,
		WriteTimeout:      time.Duration(envInt("HTTP_WRITE_TIMEOUT_SEC", 300)) * time.Second,
		IdleTimeout:       120 * time.Second,
		MaxHeaderBytes:    1 << 20,
	}

	go func() {
		logger.Info("http api listening", "addr", addr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Error("http server error", "error", err)
		}
	}()

	// gRPC server (GRPC_PORT=0 to disable, default 50051)
	var grpcSrv *grpc.Server
	grpcPort := envInt("GRPC_PORT", 50051)
	if grpcPort > 0 {
		lis, err := net.Listen("tcp", fmt.Sprintf(":%d", grpcPort))
		if err != nil {
			logger.Error("failed to listen for gRPC", "error", err)
		} else {
			grpcSrv = grpc.NewServer(
				grpc.MaxRecvMsgSize(64*1024*1024),
				grpc.MaxSendMsgSize(64*1024*1024),
				grpc.UnaryInterceptor(grpcAuthInterceptor(store.jwtMgr, store.apiToken, store.requireAuth, logger)),
			)
			deepdatav1.RegisterDeepDataServer(grpcSrv, &CollectionGRPCServer{
				manager: collectionHTTP.Manager(),
			})
			go func() {
				logger.Info("grpc api listening", "addr", lis.Addr())
				if err := grpcSrv.Serve(lis); err != nil {
					logger.Error("grpc server error", "error", err)
				}
			}()
		}
	}

	// Background compaction
	compactDone := make(chan struct{})
	compactStop := make(chan struct{})
	go func() {
		defer close(compactDone)
		interval := time.Duration(envInt("COMPACT_INTERVAL_MIN", 60)) * time.Minute
		tombstoneThreshold := float64(envInt("COMPACT_TOMBSTONE_THRESHOLD", 10)) / 100.0
		t := time.NewTicker(interval)
		defer t.Stop()
		for {
			select {
			case <-compactStop:
				return
			case <-t.C:
				store.RLock()
				total := store.Count
				deleted := len(store.Deleted)
				store.RUnlock()
				if total == 0 {
					continue
				}
				if float64(deleted)/float64(total) >= tombstoneThreshold {
					logger.Info("auto-compaction triggered", "deleted", deleted, "total", total)
					if err := store.Compact(indexPath); err != nil {
						logger.Error("compact error", "error", err)
					}
				}
			}
		}
	}()

	// Background Obsidian vault sync
	var obsidianSyncCancel context.CancelFunc
	obsidianDone := make(chan struct{})
	{
		cfg := obsidian.LoadOrDetectConfig(dataDir)
		obsidian.ApplyEnvOverrides(&cfg)

		if cfg.Enabled && cfg.VaultPath != "" {
			cfg.StateFile = filepath.Join(dataDir, ".obsidian-sync-state")
			syncCtx, cancel := context.WithCancel(context.Background())
			obsidianSyncCancel = cancel

			// Wire store methods into callbacks to avoid import cycles
			embedFn := func(text string) ([]float32, error) {
				return swappableEmbedder.Embed(text)
			}
			upsertFn := func(vec []float32, doc, id string, meta map[string]string, collection string) error {
				_, err := store.Upsert(vec, doc, id, meta, collection, "default")
				return err
			}
			deleteFn := func(id string) error {
				return store.Delete(id)
			}
			iterFn := func(collection string, fn func(id string, meta map[string]string) bool) {
				type snapshotEntry struct {
					id   string
					meta map[string]string
				}

				store.RLock()
				entries := make([]snapshotEntry, 0, store.Count)
				for i := 0; i < store.Count; i++ {
					docID := store.GetID(i)
					hid := hashID(docID)
					if store.Deleted[hid] {
						continue
					}
					if collection != "" && store.Coll[hid] != collection {
						continue
					}
					metaCopy := make(map[string]string, len(store.Meta[hid]))
					for k, v := range store.Meta[hid] {
						metaCopy[k] = v
					}
					entries = append(entries, snapshotEntry{id: docID, meta: metaCopy})
				}
				store.RUnlock()

				for _, entry := range entries {
					if !fn(entry.id, entry.meta) {
						break
					}
				}
			}

			go func() {
				defer close(obsidianDone)
				obsidian.SyncLoop(syncCtx, cfg, logger, embedFn, upsertFn, deleteFn, iterFn)
			}()
			logger.Info("obsidian auto-sync started", "vault", cfg.VaultPath, "interval", cfg.Interval)
		} else {
			close(obsidianDone) // Not started — unblock shutdown wait
			if vaults := obsidian.DetectVaults(); len(vaults) > 0 {
				logger.Info("obsidian vault detected (not syncing — set OBSIDIAN_VAULT to enable)",
					"vault", vaults[0])
			}
		}
	}

	// Setup graceful shutdown
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGTERM, syscall.SIGINT)

	// Wait for shutdown signal
	logging.Default().Info("server running, press Ctrl+C to stop")
	sig := <-sigCh
	logging.Default().Info("received signal, initiating graceful shutdown", "signal", sig)

	// Stop background compaction
	close(compactStop)

	// Stop obsidian sync
	if obsidianSyncCancel != nil {
		obsidianSyncCancel()
	}

	// Graceful shutdown sequence
	if grpcSrv != nil {
		logging.Default().Info("shutting down gRPC server")
		grpcDone := make(chan struct{})
		go func() {
			grpcSrv.GracefulStop()
			close(grpcDone)
		}()
		select {
		case <-grpcDone:
		case <-time.After(30 * time.Second):
			logging.Default().Warn("gRPC graceful shutdown timed out, forcing stop")
			grpcSrv.Stop()
		}
	}
	logging.Default().Info("shutting down HTTP server")
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		logging.Default().Error("HTTP server shutdown error", "error", err)
	}

	// Wait for background goroutines to finish before final save
	logger.Info("waiting for background compaction to finish...")
	<-compactDone
	logger.Info("waiting for obsidian sync to finish...")
	<-obsidianDone
	logger.Info("waiting for in-flight WAL snapshots to finish...")
	store.bgWg.Wait()

	logging.Default().Info("saving final snapshot")
	if err := store.Save(indexPath); err != nil {
		logger.Error("failed to save final snapshot", "error", err)
	} else {
		logger.Info("final snapshot saved successfully")
		// Clean up WAL after successful save — safe here because shutdown has
		// drained all writers (bgWg.Wait completed above, HTTP/gRPC stopped).
		if store.walPath != "" {
			_ = os.Remove(store.walPath)
		}
	}
	if err := collectionHTTP.Save(indexPath + ".collections"); err != nil {
		logger.Error("failed to save collection state", "error", err)
	} else {
		logger.Info("collection state saved successfully")
	}

	logger.Info("shutdown complete")
}

func envInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil {
			logging.Default().Warn("invalid integer env var, using default", "key", key, "value", v, "default", def)
			return def
		}
		return n
	}
	return def
}

func envInt64(key string, def int64) int64 {
	if v := os.Getenv(key); v != "" {
		n, err := strconv.ParseInt(v, 10, 64)
		if err != nil || n <= 0 {
			logging.Default().Warn("invalid positive integer env var, using default", "key", key, "value", v, "default", def)
			return def
		}
		return n
	}
	return def
}

// validateEnvConfig checks all known environment variables for valid values at
// startup. If any env var is set to an unparseable or out-of-range value, this
// returns a list of errors. The caller should log them and exit — fail-fast
// prevents silent misconfiguration in production.
func validateEnvConfig(logger *logging.Logger) []string {
	var errs []string

	// Helper: check that an env var, if set, parses as a positive integer
	checkPosInt := func(key string) {
		if v := os.Getenv(key); v != "" {
			n, err := strconv.Atoi(v)
			if err != nil {
				errs = append(errs, fmt.Sprintf("%s=%q is not a valid integer", key, v))
			} else if n <= 0 {
				errs = append(errs, fmt.Sprintf("%s=%d must be positive", key, n))
			}
		}
	}

	// Helper: check that an env var, if set, parses as a non-negative integer
	checkNonNegInt := func(key string) {
		if v := os.Getenv(key); v != "" {
			n, err := strconv.Atoi(v)
			if err != nil {
				errs = append(errs, fmt.Sprintf("%s=%q is not a valid integer", key, v))
			} else if n < 0 {
				errs = append(errs, fmt.Sprintf("%s=%d must be non-negative", key, n))
			}
		}
	}

	// Helper: check positive int64
	checkPosInt64 := func(key string) {
		if v := os.Getenv(key); v != "" {
			n, err := strconv.ParseInt(v, 10, 64)
			if err != nil {
				errs = append(errs, fmt.Sprintf("%s=%q is not a valid integer", key, v))
			} else if n <= 0 {
				errs = append(errs, fmt.Sprintf("%s=%d must be positive", key, n))
			}
		}
	}

	// Helper: check positive float
	checkPosFloat := func(key string) {
		if v := os.Getenv(key); v != "" {
			n, err := strconv.ParseFloat(v, 64)
			if err != nil {
				errs = append(errs, fmt.Sprintf("%s=%q is not a valid float", key, v))
			} else if n <= 0 {
				errs = append(errs, fmt.Sprintf("%s=%f must be positive", key, n))
			}
		}
	}

	// STORAGE_FORMAT: must be a registered format name
	if v := os.Getenv("STORAGE_FORMAT"); v != "" {
		if storage.Get(v) == nil {
			errs = append(errs, fmt.Sprintf("STORAGE_FORMAT=%q is not a registered format (available: %v)", v, storage.List()))
		}
	}

	// LOG_LEVEL: only "debug" or "" (info) are meaningful
	if v := os.Getenv("LOG_LEVEL"); v != "" {
		switch strings.ToLower(v) {
		case "debug", "info", "warn", "error":
			// valid
		default:
			errs = append(errs, fmt.Sprintf("LOG_LEVEL=%q is not valid (use: debug, info, warn, error)", v))
		}
	}

	// Integer config vars
	checkPosInt("PORT")
	checkNonNegInt("VECTOR_CAPACITY")
	checkPosInt64("WAL_MAX_BYTES")
	checkPosInt("WAL_MAX_OPS")
	checkPosInt("HNSW_M")
	checkPosFloat("HNSW_ML")
	checkPosInt("HNSW_EFSEARCH")
	checkPosInt("TENANT_RPS")
	checkPosInt("TENANT_BURST")
	checkPosInt("MAX_TENANTS")
	checkPosInt("MAX_COLLECTIONS")
	checkPosInt("HTTP_READ_TIMEOUT_SEC")
	checkPosInt("HTTP_WRITE_TIMEOUT_SEC")
	checkPosInt("HTTP_REQUEST_TIMEOUT_SEC")

	// EMBED_DIM: positive integer if set
	checkPosInt("EMBED_DIM")

	// ONNX_EMBED_MAX_LEN / ONNX_RERANK_MAX_LEN: positive integer if set
	checkPosInt("ONNX_EMBED_MAX_LEN")
	checkPosInt("ONNX_RERANK_MAX_LEN")

	return errs
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

// grpcAuthInterceptor returns a gRPC unary interceptor that mirrors the HTTP
// guard middleware: JWT validation, legacy API-token checking, and requireAuth
// enforcement. On success it injects a *security.TenantContext into the
// context so downstream handlers can inspect tenant identity and permissions.
func grpcAuthInterceptor(jwtMgr *security.JWTManager, apiToken string, requireAuth bool, logger *logging.Logger) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (resp any, err error) {
		// Panic recovery — same as before, prevents crashes from taking down the process
		defer func() {
			if r := recover(); r != nil {
				logger.Error("panic recovered in gRPC handler", "error", r, "method", info.FullMethod)
				err = status.Errorf(codes.Internal, "internal error")
			}
		}()

		// Extract auth token from gRPC metadata (mirrors HTTP Authorization header)
		token := ""
		if md, ok := metadata.FromIncomingContext(ctx); ok {
			if vals := md.Get("authorization"); len(vals) > 0 {
				token = strings.TrimPrefix(vals[0], "Bearer ")
			}
		}

		var tenantCtx *security.TenantContext

		if jwtMgr != nil {
			if token == "" {
				if requireAuth {
					return nil, status.Error(codes.Unauthenticated, "missing authentication token")
				}
				tenantCtx = &security.TenantContext{
					TenantID:    "default",
					Permissions: map[string]bool{"read": true, "write": true},
					Collections: make(map[string]bool),
				}
			} else {
				var valErr error
				tenantCtx, valErr = jwtMgr.ValidateTenantToken(token)
				if valErr != nil {
					return nil, status.Errorf(codes.Unauthenticated, "invalid token: %v", valErr)
				}
			}
		} else {
			authenticated := false
			if apiToken != "" {
				if token == apiToken {
					authenticated = true
				} else if token != "" {
					return nil, status.Error(codes.Unauthenticated, "unauthorized")
				}
			}
			if requireAuth && !authenticated {
				return nil, status.Error(codes.Unauthenticated, "unauthorized")
			}
			tenantCtx = &security.TenantContext{
				TenantID:    "default",
				Permissions: map[string]bool{"read": true, "write": true},
				Collections: make(map[string]bool),
				IsAdmin:     jwtMgr == nil && apiToken == "",
			}
		}

		ctx = context.WithValue(ctx, security.TenantContextKey, tenantCtx)
		return handler(ctx, req)
	}
}
