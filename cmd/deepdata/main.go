package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
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

	"github.com/phenomenon0/vectordb/internal/apierror"
	"github.com/phenomenon0/vectordb/internal/index"
	"github.com/phenomenon0/vectordb/internal/logging"
	"github.com/phenomenon0/vectordb/internal/security"
	"github.com/phenomenon0/vectordb/internal/storage"
	"github.com/phenomenon0/vectordb/internal/telemetry"

	"golang.org/x/net/http2"
	"golang.org/x/net/http2/h2c"

	deepdatav3 "github.com/phenomenon0/vectordb/api/gen/deepdata/v3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"

	"net"
)

// ======================================================================================
// Vector Store (HNSW + metadata + persistence)
// ======================================================================================

type VectorStore struct {
	sync.RWMutex
	Data    []float32
	Dim     int
	Count   int
	Docs    []string
	IDs     []string
	Seqs    []uint64
	next    int64
	nextSeq uint64
	// Index abstraction - the single source of truth for vector search
	indexes       map[string]index.Index // Collection -> Index mapping
	idToIx        map[uint64]int
	Meta          map[uint64]map[string]string
	Deleted       map[uint64]bool
	Coll          map[uint64]string
	NumMeta       map[uint64]map[string]float64
	TimeMeta      map[uint64]map[string]time.Time
	numIndex      map[string][]numEntry
	timeIndex     map[string][]timeEntry
	walPath       string
	walMu         sync.Mutex
	walMaxBytes   int64
	walMaxOps     int
	walOps        int
	walRotate     int64
	walHook       func(walEntry) // Optional hook to forward WAL events (e.g., to replication stream)
	walFault      error          // Latched after an append may have partially reached durable storage
	nextWALSeq    uint64         // Next monotonic WAL sequence to allocate (starts at 1)
	appliedWALSeq uint64         // Highest WAL sequence represented in logical state
	// Authentication and limit state, shared with the V3 surface (runtime.go).
	*serverRuntime
	checksum           string
	lastSaved          time.Time
	lastSnapshotWALSeq uint64 // WAL high-water in the last successfully renamed snapshot
	// Lexical stats for hybrid/BM25
	lexTF   map[uint64]map[string]int
	docLen  map[uint64]int
	df      map[string]int
	sumDocL int
	// Multi-tenancy support
	TenantID map[uint64]string  // vector hash -> tenant ID
	tenantRL *tenantRateLimiter // per-tenant rate limiting
	// Storage format (gob, cowrie, cowrie-zstd)
	storageFormat storage.Format
	// Metadata bitmap index for fast pre-filtering
	// Metadata bitmap index for fast pre-filtering
	metaIndex *MetadataIndex

	// Background goroutine lifecycle
	bgWg            sync.WaitGroup // Tracks in-flight background snapshot goroutines
	snapshotRunning atomic.Bool    // Deduplicates background checkpoint scheduling
	snapshotMu      sync.Mutex     // Serializes every snapshot commit, including explicit saves

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

	// Select storage format (default: gob for backward compatibility)
	// Options: "gob", "cowrie", "cowrie-zstd"
	storageFormat := storage.Default()
	if formatName := os.Getenv("STORAGE_FORMAT"); formatName != "" {
		if f := storage.Get(formatName); f != nil {
			storageFormat = f
		}
	}

	return &VectorStore{
		// Authentication and limits, read from the same environment as before.
		serverRuntime: newServerRuntime(),
		Data:          make([]float32, 0, capacity*dim),
		Dim:           dim,
		Count:         0,
		indexes:       map[string]index.Index{"default": defaultIdx},
		idToIx:        make(map[uint64]int),
		Meta:          make(map[uint64]map[string]string),
		Deleted:       make(map[uint64]bool),
		Coll:          make(map[uint64]string),
		Seqs:          make([]uint64, 0, capacity),
		NumMeta:       make(map[uint64]map[string]float64),
		TimeMeta:      make(map[uint64]map[string]time.Time),
		numIndex:      make(map[string][]numEntry),
		timeIndex:     make(map[string][]timeEntry),
		walMaxBytes:   0,
		walMaxOps:     0,
		nextWALSeq:    1,
		lexTF:         make(map[uint64]map[string]int),
		docLen:        make(map[uint64]int),
		df:            make(map[string]int),
		sumDocL:       0,
		// Multi-tenancy
		TenantID: make(map[uint64]string),
		tenantRL: newTenantRateLimiter(envInt("TENANT_RPS", 100), envInt("TENANT_BURST", 100), envInt("MAX_TENANTS", 100_000), time.Minute),
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
	vs.Seqs = append(vs.Seqs, vs.nextSeq)
	vs.nextSeq++
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
		// A rejected Add (most commonly a duplicate ID) did not install this
		// vector. Deleting hid here would remove the pre-existing index entry.
		if createdIndex {
			delete(vs.indexes, collection)
		}
		vs.quotas.RemoveUsage(tenantID, totalBytes, 1)
		return "", fmt.Errorf("failed to add vector to index: %w", err)
	}

	nextID := vs.next
	if autoGenerated {
		nextID++
	}
	walRecord, err := vs.appendWAL("insert", id, doc, meta, v, collection, tenantID, nextID)
	if err != nil {
		vs.rollbackNewVectorLocked(idx, collection, createdIndex, hid, tenantID, totalBytes)
		return "", fmt.Errorf("failed to append to WAL: %w", err)
	}

	vs.commitNewVectorLocked(v, doc, id, meta, collection, tenantID, autoGenerated)
	vs.commitWALMutation(walRecord)
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
	originalNext := vs.next
	if id == "" {
		autoGenerated = true
		for {
			if vs.next == math.MaxInt64 {
				return "", fmt.Errorf("automatic document ID space exhausted")
			}
			id = fmt.Sprintf("doc-%d", vs.next)
			if _, exists := vs.idToIx[hashID(id)]; !exists {
				break
			}
			vs.next++
		}
	}
	if tenantID == "" {
		tenantID = "default" // Default tenant for backward compatibility
	}

	result, err := vs.addNewLocked(v, doc, id, meta, collection, tenantID, autoGenerated)
	if err != nil && autoGenerated {
		vs.next = originalNext
	}
	return result, err
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

		walRecord, err := vs.appendWAL("upsert", id, doc, meta, v, targetCollection, tenantID, vs.next)
		if err != nil {
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
		vs.commitWALMutation(walRecord)
		return id, nil
	}

	return vs.addNewLocked(v, doc, id, meta, collection, tenantID, false)
}

// isNotFoundError checks if an error is a "not found" type error that can be safely ignored.
// Prefers the sentinel index.ErrNotFound via errors.Is, with a string fallback for
// legacy backends that haven't adopted the sentinel yet.
func isNotFoundError(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, index.ErrNotFound) {
		return true
	}
	// Fallback for index backends that don't yet wrap index.ErrNotFound
	errStr := strings.ToLower(err.Error())
	return strings.Contains(errStr, "not found") ||
		strings.Contains(errStr, "not exist") ||
		strings.Contains(errStr, "does not exist")
}

func (vs *VectorStore) Delete(id string) error {
	vs.Lock()
	defer vs.Unlock()
	hid := hashID(id)
	if _, exists := vs.idToIx[hid]; !exists {
		return nil
	}
	if vs.Deleted[hid] {
		return nil
	}
	tenant := vs.TenantID[hid]
	if tenant == "" {
		tenant = "default"
	}

	// Delete from index abstraction
	delCollection := vs.Coll[hid]
	if delCollection == "" {
		delCollection = "default"
	}
	// The delete record must be durable before logical state changes. If the
	// process dies after fsync but before mutation, startup replay completes it.
	walRecord, err := vs.appendWAL("delete", id, "", nil, nil, delCollection, tenant, vs.next)
	if err != nil {
		return fmt.Errorf("failed to append to WAL: %w", err)
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
	vs.commitWALMutation(walRecord)
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
	vs.snapshotMu.Lock()
	defer vs.snapshotMu.Unlock()

	vs.RLock()

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		vs.RUnlock()
		return err
	}
	f, err := os.CreateTemp(filepath.Dir(path), "."+filepath.Base(path)+".tmp-*")
	if err != nil {
		vs.RUnlock()
		return err
	}
	tmp := f.Name()

	// Export indexes
	indexData := make(map[string][]byte)
	indexTypes := make(map[string]string)
	indexDims := make(map[string]int)
	indexChecksums := make(map[string]string)
	activeByCollection := make(map[string]int)
	for _, id := range vs.IDs {
		hid := hashID(id)
		if vs.Deleted[hid] {
			continue
		}
		collection := vs.Coll[hid]
		if collection == "" {
			collection = "default"
		}
		activeByCollection[collection]++
	}
	for collection := range activeByCollection {
		if vs.indexes[collection] == nil {
			_ = f.Close()
			_ = os.Remove(tmp)
			vs.RUnlock()
			return fmt.Errorf("active collection %q has no index", collection)
		}
	}
	for collName, idx := range vs.indexes {
		if idx != nil {
			data, err := idx.Export()
			if err != nil {
				_ = f.Close()
				_ = os.Remove(tmp)
				vs.RUnlock()
				return fmt.Errorf("export index %q: %w", collName, err)
			}
			indexType, err := persistedIndexType(idx.Name())
			if err != nil {
				_ = f.Close()
				_ = os.Remove(tmp)
				vs.RUnlock()
				return fmt.Errorf("describe index %q: %w", collName, err)
			}
			stats := idx.Stats()
			if stats.Dim != vs.Dim {
				_ = f.Close()
				_ = os.Remove(tmp)
				vs.RUnlock()
				return fmt.Errorf("index %q dimension mismatch: got %d, want %d", collName, stats.Dim, vs.Dim)
			}
			if stats.Active != activeByCollection[collName] {
				_ = f.Close()
				_ = os.Remove(tmp)
				vs.RUnlock()
				return fmt.Errorf("index %q active count mismatch: got %d, want %d", collName, stats.Active, activeByCollection[collName])
			}
			indexData[collName] = data
			indexTypes[collName] = indexType
			indexDims[collName] = stats.Dim
			indexChecksums[collName] = indexBlobChecksum(data)
		}
	}
	newChecksum := vs.computeChecksum()
	lastSaved := time.Now()

	payload := &storage.Payload{
		FormatVersion:  storage.CurrentFormatVersion,
		Dim:            vs.Dim,
		Data:           vs.Data,
		Docs:           vs.Docs,
		IDs:            vs.IDs,
		Seqs:           vs.Seqs,
		Meta:           vs.Meta,
		Deleted:        vs.Deleted,
		Coll:           vs.Coll,
		TenantID:       vs.TenantID,
		Next:           vs.next,
		NextSeq:        vs.nextSeq,
		WALHighWater:   vs.appliedWALSeq,
		Count:          vs.Count,
		HNSW:           nil,       // Legacy field - no longer written
		Indexes:        indexData, // Primary index storage
		IndexTypes:     indexTypes,
		IndexDims:      indexDims,
		IndexChecksums: indexChecksums,
		Checksum:       newChecksum,
		LastSaved:      lastSaved,
		LexTF:          vs.lexTF,
		DocLen:         vs.docLen,
		DF:             vs.df,
		SumDocL:        vs.sumDocL,
		NumMeta:        vs.NumMeta,
		TimeMeta:       vs.TimeMeta,
	}

	// Use configured storage format (default: gob for backward compatibility)
	format := vs.storageFormat
	if format == nil {
		format = storage.Default()
	}
	if err := validateStoragePayload(payload); err != nil {
		_ = f.Close()
		_ = os.Remove(tmp)
		vs.RUnlock()
		return fmt.Errorf("validate snapshot before save: %w", err)
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

	vs.RUnlock()

	// Rename atomically commits the new snapshot.
	// WAL cleanup is the caller's responsibility — Save() must not delete the
	// WAL because background snapshot goroutines race with concurrent appendWAL
	// writers: entries written after the RLock was released but before this point
	// would be lost if we deleted the WAL here.
	if err := renameSnapshotFile(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	// Persist the directory entry as well as the file contents. Without this,
	// a power loss can lose the rename even though the snapshot itself was synced.
	dir, err := os.Open(filepath.Dir(path))
	if err != nil {
		return fmt.Errorf("open snapshot directory: %w", err)
	}
	if err := dir.Sync(); err != nil {
		_ = dir.Close()
		return fmt.Errorf("sync snapshot directory: %w", err)
	}
	if err := dir.Close(); err != nil {
		return fmt.Errorf("close snapshot directory: %w", err)
	}

	vs.Lock()
	vs.checksum = newChecksum
	vs.lastSaved = lastSaved
	vs.lastSnapshotWALSeq = payload.WALHighWater
	vs.Unlock()
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

func persistedIndexType(name string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "hnsw", "ivf", "flat", "diskann", "sparse", "binary", "ivf_binary", "ivf-binary":
		return strings.ReplaceAll(strings.ToLower(strings.TrimSpace(name)), "-", "_"), nil
	case "pq-adc":
		return "pq", nil
	case "ivf-pq-adc":
		return "ivf_pq", nil
	case "pq4-adc":
		return "pq4", nil
	default:
		return "", fmt.Errorf("unsupported index type %q", name)
	}
}

func indexBlobChecksum(data []byte) string {
	sum := sha256.Sum256(data)
	return fmt.Sprintf("sha256:%x", sum[:])
}

// tryLoadPayload attempts every supported snapshot codec while preserving the
// decoder errors. An existing file that no codec can verify is corrupt state,
// not permission to initialize an empty database.
func tryLoadPayload(path string) (*storage.Payload, storage.Format, error) {
	formatNames := []string{"gob", "cowrie", "cowrie-zstd", "cowrie-delta-zstd"}
	loadErrors := make([]error, 0, len(formatNames))
	for _, formatName := range formatNames {
		format := storage.Get(formatName)
		if format == nil {
			continue
		}
		payload, err := tryLoadWithFormat(path, format)
		if err != nil {
			loadErrors = append(loadErrors, fmt.Errorf("%s: %w", formatName, err))
			continue
		}
		if err := validateStoragePayload(payload); err != nil {
			loadErrors = append(loadErrors, fmt.Errorf("%s validation: %w", formatName, err))
			continue
		}
		return payload, format, nil
	}

	if len(loadErrors) == 0 {
		return nil, nil, fmt.Errorf("no snapshot formats are registered")
	}
	return nil, nil, fmt.Errorf("load snapshot %q: %w", path, errors.Join(loadErrors...))
}

// tryLoadWithFormat attempts to load a payload using a specific format.
func tryLoadWithFormat(path string, format storage.Format) (*storage.Payload, error) {
	if format == nil {
		return nil, fmt.Errorf("storage format is nil")
	}

	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	payload, err := format.Load(f)
	if err != nil {
		return nil, err
	}
	return payload, nil
}

func validateStoragePayload(payload *storage.Payload) error {
	if payload == nil {
		return fmt.Errorf("payload is nil")
	}
	if payload.FormatVersion < 0 || payload.FormatVersion > storage.CurrentFormatVersion {
		return fmt.Errorf("unsupported format version %d", payload.FormatVersion)
	}
	if payload.FormatVersion >= storage.CanonicalFormatVersion && payload.Checksum == "" {
		return fmt.Errorf("canonical snapshot format %d is missing a checksum", payload.FormatVersion)
	}
	if payload.FormatVersion < storage.CurrentFormatVersion && payload.WALHighWater != 0 {
		return fmt.Errorf("snapshot format %d cannot contain a WAL high-water mark", payload.FormatVersion)
	}
	if payload.Dim <= 0 {
		return fmt.Errorf("invalid dimension %d", payload.Dim)
	}
	if payload.Count < 0 || payload.Next < 0 {
		return fmt.Errorf("invalid counters: count=%d next=%d", payload.Count, payload.Next)
	}
	if len(payload.Docs) != len(payload.IDs) {
		return fmt.Errorf("document/id length mismatch: %d != %d", len(payload.Docs), len(payload.IDs))
	}
	if payload.Count != len(payload.IDs) {
		return fmt.Errorf("count/id length mismatch: %d != %d", payload.Count, len(payload.IDs))
	}
	if payload.FormatVersion >= storage.CanonicalFormatVersion {
		if payload.VectorType != 0 || len(payload.VectorData) != 0 {
			return fmt.Errorf("current server does not support persisted VectorData payloads")
		}
		knownIDs := make(map[uint64]struct{}, len(payload.IDs))
		seenIDs := make(map[string]struct{}, len(payload.IDs))
		for _, id := range payload.IDs {
			if id == "" {
				return fmt.Errorf("document ID is empty")
			}
			if _, duplicate := seenIDs[id]; duplicate {
				return fmt.Errorf("duplicate document ID %q", id)
			}
			seenIDs[id] = struct{}{}
			hashed := hashID(id)
			if _, collision := knownIDs[hashed]; collision {
				return fmt.Errorf("document ID hash collision for %q", id)
			}
			knownIDs[hashed] = struct{}{}
			if !payload.Deleted[hashed] {
				if payload.Coll[hashed] == "" {
					return fmt.Errorf("active document %q is missing a collection", id)
				}
				if payload.TenantID[hashed] == "" {
					return fmt.Errorf("active document %q is missing a tenant", id)
				}
			}
		}
		validateKnownKeys := func(name string, keys []uint64) error {
			for _, id := range keys {
				if _, ok := knownIDs[id]; !ok {
					return fmt.Errorf("%s references unknown document hash %d", name, id)
				}
			}
			return nil
		}
		for name, keys := range map[string][]uint64{
			"metadata":         sortedUint64MapKeys(payload.Meta),
			"deletions":        sortedUint64MapKeys(payload.Deleted),
			"collections":      sortedUint64MapKeys(payload.Coll),
			"tenants":          sortedUint64MapKeys(payload.TenantID),
			"lexical terms":    sortedUint64MapKeys(payload.LexTF),
			"document lengths": sortedUint64MapKeys(payload.DocLen),
			"numeric metadata": sortedUint64MapKeys(payload.NumMeta),
			"time metadata":    sortedUint64MapKeys(payload.TimeMeta),
		} {
			if err := validateKnownKeys(name, keys); err != nil {
				return err
			}
		}
		if len(payload.Seqs) != len(payload.IDs) {
			return fmt.Errorf("sequence/id length mismatch: %d != %d", len(payload.Seqs), len(payload.IDs))
		}
		for i := 1; i < len(payload.Seqs); i++ {
			if payload.Seqs[i] <= payload.Seqs[i-1] {
				return fmt.Errorf("sequences are not strictly increasing at position %d", i)
			}
		}
		if len(payload.Seqs) > 0 && payload.NextSeq <= payload.Seqs[len(payload.Seqs)-1] {
			return fmt.Errorf("next sequence %d does not exceed high-water mark %d", payload.NextSeq, payload.Seqs[len(payload.Seqs)-1])
		}
		if len(payload.Indexes) == 0 {
			return fmt.Errorf("current-format snapshot has no indexes")
		}
		if _, ok := payload.Indexes["default"]; !ok {
			return fmt.Errorf("current-format snapshot has no default index")
		}
		if len(payload.IndexTypes) != len(payload.Indexes) || len(payload.IndexDims) != len(payload.Indexes) || len(payload.IndexChecksums) != len(payload.Indexes) {
			return fmt.Errorf("index descriptor count mismatch: blobs=%d types=%d dims=%d checksums=%d", len(payload.Indexes), len(payload.IndexTypes), len(payload.IndexDims), len(payload.IndexChecksums))
		}
		for collection, data := range payload.Indexes {
			if collection == "" {
				return fmt.Errorf("index collection name is empty")
			}
			if _, err := persistedIndexType(payload.IndexTypes[collection]); err != nil {
				return fmt.Errorf("index %q: %w", collection, err)
			}
			if payload.IndexDims[collection] != payload.Dim {
				return fmt.Errorf("index %q dimension mismatch: got %d, want %d", collection, payload.IndexDims[collection], payload.Dim)
			}
			if got, want := payload.IndexChecksums[collection], indexBlobChecksum(data); got != want {
				return fmt.Errorf("index %q checksum mismatch", collection)
			}
		}
		for id, collection := range payload.Coll {
			if payload.Deleted[id] {
				continue
			}
			if collection == "" {
				collection = "default"
			}
			if _, ok := payload.Indexes[collection]; !ok {
				return fmt.Errorf("document %d references missing collection index %q", id, collection)
			}
		}
	}
	if len(payload.Data) != len(payload.IDs)*payload.Dim {
		return fmt.Errorf("vector data length mismatch: got %d, want %d", len(payload.Data), len(payload.IDs)*payload.Dim)
	}
	return nil
}

func recognizedLegacyChecksum(payload *storage.Payload) bool {
	if payload.Checksum == "" {
		// The historical Save path wrote the previous in-memory checksum into
		// the first snapshot, which was empty for a newly initialized store.
		return true
	}
	weakStateChecksum := fmt.Sprintf("%x", hashID(fmt.Sprintf("%d-%d-%d", payload.Count, payload.Next, len(payload.Docs))))
	olderCounterChecksum := fmt.Sprintf("%x", hashID(fmt.Sprintf("%d-%d", payload.Count, payload.Next)))
	return payload.Checksum == weakStateChecksum || payload.Checksum == olderCounterChecksum
}

// Load snapshot or initialize a store only when no prior snapshot exists.
// Existing state that cannot be verified is returned as an error so callers
// can fail before binding listeners and preserve the original recovery data.
func loadOrInitStore(path string, capacity int, dim int) (*VectorStore, bool, error) {
	if _, err := os.Stat(path); err == nil {
		// Try to load with configured format, fall back to gob for backward compatibility
		payload, loadedFormat, err := tryLoadPayload(path)
		if err != nil {
			return nil, false, err
		}
		logging.Default().Info("loaded snapshot", "format", loadedFormat.Name(), "path", path)
		// Initialize JWT manager if configured
		vs := &VectorStore{
			// Authentication and limits, read from the same environment as before.
			serverRuntime:      newServerRuntime(),
			Data:               payload.Data,
			Dim:                payload.Dim,
			Count:              payload.Count,
			Docs:               payload.Docs,
			IDs:                payload.IDs,
			Seqs:               payload.Seqs,
			next:               payload.Next,
			nextSeq:            payload.NextSeq,
			nextWALSeq:         payload.WALHighWater + 1,
			appliedWALSeq:      payload.WALHighWater,
			lastSnapshotWALSeq: payload.WALHighWater,
			Meta:               payload.Meta,
			Deleted:            payload.Deleted,
			Coll:               payload.Coll,
			TenantID:           payload.TenantID,
			indexes:            make(map[string]index.Index),
			idToIx:             make(map[uint64]int),
			walPath:            path + ".wal",
			walMu:              sync.Mutex{},
			walMaxBytes:        0,
			walMaxOps:          0,
			checksum:           payload.Checksum,
			lastSaved:          payload.LastSaved,
			lexTF:              payload.LexTF,
			docLen:             payload.DocLen,
			df:                 payload.DF,
			sumDocL:            payload.SumDocL,
			NumMeta:            payload.NumMeta,
			TimeMeta:           payload.TimeMeta,
			// Numeric/time index maps for range queries (must be initialized!)
			numIndex:  make(map[string][]numEntry),
			timeIndex: make(map[string][]timeEntry),
			// Multi-tenancy support (TenantID already set from payload above)
			tenantRL: newTenantRateLimiter(envInt("TENANT_RPS", 100), envInt("TENANT_BURST", 100), envInt("MAX_TENANTS", 100_000), time.Minute),
			// Storage format
			storageFormat: getStorageFormat(),
			// Metadata index (rebuilt below)
			metaIndex: NewMetadataIndex(),
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
				indexType := "hnsw"
				indexDim := vs.Dim
				if payload.FormatVersion >= storage.CanonicalFormatVersion {
					indexType = payload.IndexTypes[collName]
					indexDim = payload.IndexDims[collName]
				}
				idx, err := index.Create(indexType, indexDim, nil)
				if err != nil {
					return nil, false, fmt.Errorf("create %s index for collection %q: %w", indexType, collName, err)
				}
				if err := idx.Import(data); err != nil {
					return nil, false, fmt.Errorf("import index for collection %q: %w", collName, err)
				}
				stats := idx.Stats()
				if stats.Dim != vs.Dim {
					return nil, false, fmt.Errorf("imported index %q dimension mismatch: got %d, want %d", collName, stats.Dim, vs.Dim)
				}
				if payload.FormatVersion >= storage.CanonicalFormatVersion {
					expectedActive := 0
					for _, id := range vs.IDs {
						hid := hashID(id)
						collection := vs.Coll[hid]
						if collection == "" {
							collection = "default"
						}
						if collection == collName && !vs.Deleted[hid] {
							expectedActive++
						}
					}
					if stats.Active != expectedActive {
						return nil, false, fmt.Errorf("imported index %q active count mismatch: got %d, want %d", collName, stats.Active, expectedActive)
					}
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
				return nil, false, fmt.Errorf("create default index: %w", err)
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
					return nil, false, fmt.Errorf("rebuild default index for document %q: %w", idStr, err)
				}
				migrated++
			}
			vs.indexes["default"] = defaultIdx
			if migrated > 0 {
				logging.Default().Info("migrated vectors to index abstraction", "count", migrated)
			}
		}

		if payload.FormatVersion < storage.CanonicalFormatVersion && len(vs.Seqs) == 0 {
			vs.Seqs = make([]uint64, len(vs.IDs))
			for i := range vs.Seqs {
				vs.Seqs[i] = uint64(i)
			}
		}
		if len(vs.Seqs) != len(vs.IDs) {
			return nil, false, fmt.Errorf("sequence/id length mismatch after migration: %d != %d", len(vs.Seqs), len(vs.IDs))
		}
		if payload.FormatVersion < storage.CanonicalFormatVersion {
			if len(vs.Seqs) > 0 {
				vs.nextSeq = vs.Seqs[len(vs.Seqs)-1] + 1
			} else {
				vs.nextSeq = 0
			}
		}
		if payload.FormatVersion < storage.CanonicalFormatVersion && vs.next == 0 {
			vs.next = int64(len(vs.IDs))
		}
		// Normalize fields omitted by historical snapshots before migration.
		if payload.FormatVersion < storage.CanonicalFormatVersion {
			for idKey := range vs.idToIx {
				if vs.TenantID[idKey] == "" {
					vs.TenantID[idKey] = "default"
				}
				if !vs.Deleted[idKey] && vs.Coll[idKey] == "" {
					vs.Coll[idKey] = "default"
				}
			}
		}
		// Legacy snapshots omitted fields now covered by the state checksum, so
		// normalize them before migrating the checksum. Current snapshots must
		// match exactly after the same codec-stable normalization.
		if payload.FormatVersion < storage.CurrentFormatVersion {
			if payload.FormatVersion == storage.CanonicalFormatVersion {
				if payload.Checksum != vs.computeV3Checksum() {
					return nil, false, fmt.Errorf("canonical version 3 snapshot checksum mismatch")
				}
				logging.Default().Warn("migrating canonical version 3 snapshot checksum")
			} else if recognizedLegacyChecksum(payload) {
				logging.Default().Warn("migrating recognized legacy snapshot checksum", "format_version", payload.FormatVersion)
			} else {
				return nil, false, fmt.Errorf("unrecognized legacy snapshot checksum")
			}
			vs.checksum = vs.computeChecksum()
		} else if !vs.validateChecksum() {
			return nil, false, fmt.Errorf("snapshot checksum mismatch")
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
		if recovered, err := checkpointWALRecovery(vs, path); err != nil {
			return nil, false, fmt.Errorf("recover WAL artifacts: %w", err)
		} else if recovered {
			logging.Default().Info("WAL recovery checkpoint committed", "path", path, "wal_high_water", vs.appliedWALSeq)
		}
		return vs, true, nil
	} else if !os.IsNotExist(err) {
		return nil, false, fmt.Errorf("stat snapshot %q: %w", path, err)
	}
	vs := NewVectorStore(capacity, dim)
	vs.walPath = path + ".wal"
	if recovered, err := checkpointWALRecovery(vs, path); err != nil {
		return nil, false, fmt.Errorf("recover WAL without snapshot: %w", err)
	} else if recovered {
		logging.Default().Info("recovered WAL without prior snapshot", "path", path, "vectors", vs.Count)
		return vs, true, nil
	}
	return vs, false, nil
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
	EmbedQuery(text string) ([]float32, error) // encode as query (for searching)
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

const currentWALVersion = 1

type walEntry struct {
	Version  int
	Seq      uint64 // Monotonic durability sequence, distinct from pagination Seqs
	Op       string
	ID       string
	Doc      string
	Meta     map[string]string
	Vec      []float32
	Coll     string
	Tenant   string
	NextID   int64  // Auto-generated document ID high-water mark after this mutation
	Time     int64  // Unix timestamp (for WAL streaming)
	Checksum string // SHA-256 of this record with Checksum cleared
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

// commitWALMutation advances the in-memory checkpoint only after the logical
// mutation is complete. Callers hold vs.Lock, so a snapshot cannot observe an
// LSN whose state has not been installed yet.
func (vs *VectorStore) commitWALMutation(entry walEntry) {
	if entry.Seq > vs.appliedWALSeq {
		vs.appliedWALSeq = entry.Seq
	}
	vs.walMu.Lock()
	hook := vs.walHook
	vs.walMu.Unlock()
	if hook != nil {
		hook(entry)
	}
}

func walEntryChecksum(entry walEntry) (string, error) {
	entry.Checksum = ""
	encoded, err := json.Marshal(entry)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(encoded)
	return fmt.Sprintf("sha256:%x", sum[:]), nil
}

func syncParentDirectory(path string) error {
	dir, err := os.Open(filepath.Dir(path))
	if err != nil {
		return err
	}
	if err := dir.Sync(); err != nil {
		_ = dir.Close()
		return err
	}
	return dir.Close()
}

func removeDurableFile(path string) error {
	if err := os.Remove(path); err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	if err := syncParentDirectory(path); err != nil {
		return fmt.Errorf("sync directory after removing %q: %w", path, err)
	}
	return nil
}

// removeWALArtifact is a narrow test seam for cleanup-failure recovery tests.
var removeWALArtifact = removeDurableFile

// syncWALFile is a narrow test seam for indeterminate append failures.
var syncWALFile = func(f *os.File) error { return f.Sync() }

// syncWALDirectory is a narrow test seam for WAL create/rotation durability.
var syncWALDirectory = syncParentDirectory

// renameSnapshotFile is a narrow test seam for snapshot commit ordering.
var renameSnapshotFile = os.Rename

func (vs *VectorStore) appendWAL(op, id, doc string, meta map[string]string, vec []float32, collection string, tenant string, nextID int64) (walEntry, error) {
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
		Version: currentWALVersion,
		Op:      op,
		ID:      id,
		Doc:     doc,
		Meta:    meta,
		Vec:     vec,
		Coll:    collection,
		Tenant:  tenant,
		NextID:  nextID,
		Time:    time.Now().Unix(),
	}

	vs.walMu.Lock()
	if vs.walFault != nil {
		fault := vs.walFault
		vs.walMu.Unlock()
		return entry, fmt.Errorf("WAL writes are disabled after an indeterminate append: %w", fault)
	}
	if vs.walPath == "" {
		vs.walMu.Unlock()
		return entry, nil
	}
	if vs.nextWALSeq == 0 {
		vs.nextWALSeq = 1
	}
	entry.Seq = vs.nextWALSeq
	checksum, err := walEntryChecksum(entry)
	if err != nil {
		vs.walMu.Unlock()
		return entry, fmt.Errorf("checksum WAL entry: %w", err)
	}
	entry.Checksum = checksum

	_, statErr := os.Stat(vs.walPath)
	newFile := os.IsNotExist(statErr)
	if statErr != nil && !newFile {
		vs.walMu.Unlock()
		return entry, fmt.Errorf("inspect WAL: %w", statErr)
	}
	f, err := os.OpenFile(vs.walPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		vs.walMu.Unlock()
		return entry, fmt.Errorf("failed to open WAL: %w", err)
	}
	latchFault := func(stage string, cause error) (walEntry, error) {
		if vs.walFault == nil {
			vs.walFault = fmt.Errorf("WAL append became indeterminate during %s: %w", stage, cause)
		}
		fault := vs.walFault
		vs.walMu.Unlock()
		return entry, fault
	}

	if err := json.NewEncoder(f).Encode(&entry); err != nil {
		_ = f.Close()
		return latchFault("encode", err)
	}
	if err := syncWALFile(f); err != nil {
		_ = f.Close()
		return latchFault("fsync", err)
	}
	if err := f.Close(); err != nil {
		return latchFault("close", err)
	}
	if newFile {
		if err := syncWALDirectory(vs.walPath); err != nil {
			return latchFault("directory fsync", err)
		}
	}
	vs.nextWALSeq = entry.Seq + 1
	vs.walOps++

	var doSnapshot bool
	if vs.walMaxOps > 0 && vs.walOps >= vs.walMaxOps {
		doSnapshot = true
		vs.walOps = 0
	}
	if !doSnapshot && vs.walMaxBytes > 0 {
		if info, err := os.Stat(vs.walPath); err == nil && info.Size() >= vs.walMaxBytes {
			doSnapshot = true
			vs.walOps = 0
		}
	}
	vs.walMu.Unlock()

	if doSnapshot && vs.snapshotRunning.CompareAndSwap(false, true) {
		snapPath := strings.TrimSuffix(vs.walPath, ".wal")
		frozenWAL := vs.walPath + ".frozen"
		vs.walMu.Lock()
		_, frozenErr := os.Stat(frozenWAL)
		frozenExists := frozenErr == nil
		if frozenErr != nil && !os.IsNotExist(frozenErr) {
			vs.walMu.Unlock()
			logging.Default().Warn("failed to inspect frozen WAL", "error", frozenErr)
			vs.snapshotRunning.Store(false)
		} else {
			var rotateErr, rotateSyncErr error
			if !frozenExists {
				rotateErr = os.Rename(vs.walPath, frozenWAL)
				if rotateErr == nil {
					rotateSyncErr = syncWALDirectory(vs.walPath)
				}
			}
			vs.walOps = 0
			vs.walMu.Unlock()
			if rotateSyncErr != nil {
				logging.Default().Error("WAL rotation directory sync was indeterminate", "error", rotateSyncErr)
				vs.snapshotRunning.Store(false)
				vs.walMu.Lock()
				return latchFault("WAL rotation directory fsync", rotateSyncErr)
			} else if rotateErr != nil {
				logging.Default().Warn("failed to rotate WAL for snapshot", "error", rotateErr)
				vs.snapshotRunning.Store(false)
				if _, statErr := os.Stat(vs.walPath); statErr != nil {
					vs.walMu.Lock()
					return latchFault("WAL rotation", errors.Join(rotateErr, statErr))
				}
			} else {
				vs.bgWg.Add(1)
				go func() {
					defer vs.bgWg.Done()
					defer vs.snapshotRunning.Store(false)
					if err := vs.Save(snapPath); err == nil {
						covered, verifyErr := vs.frozenWALCoveredByLastSnapshot(frozenWAL)
						if verifyErr != nil || !covered {
							if verifyErr == nil {
								verifyErr = fmt.Errorf("frozen WAL extends beyond committed snapshot")
							}
							vs.Lock()
							if vs.walFault == nil {
								vs.walFault = fmt.Errorf("cannot verify checkpointed frozen WAL: %w", verifyErr)
							}
							vs.Unlock()
							logging.Default().Error("retaining unverified frozen WAL and disabling writes", "error", verifyErr, "path", frozenWAL)
						} else if err := removeWALArtifact(frozenWAL); err != nil {
							logging.Default().Warn("failed to remove checkpointed frozen WAL", "error", err, "path", frozenWAL)
						}
					} else {
						// Keep frozen and current WALs separate. A later checkpoint or
						// restart validates both before removing either artifact.
						logging.Default().Warn("background snapshot failed; WAL artifacts retained", "error", err)
					}
				}()
			}
		}
	}

	return entry, nil
}

func readWALFile(path string, dim int) ([]walEntry, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	dec := json.NewDecoder(f)
	dec.DisallowUnknownFields()
	entries := make([]walEntry, 0)
	for {
		var entry walEntry
		if err := dec.Decode(&entry); err != nil {
			if err == io.EOF {
				break
			}
			return nil, fmt.Errorf("decode WAL %q: %w", path, err)
		}
		if entry.Op != "insert" && entry.Op != "upsert" && entry.Op != "delete" {
			return nil, fmt.Errorf("unknown WAL operation %q in %q", entry.Op, path)
		}
		if entry.ID == "" {
			return nil, fmt.Errorf("WAL operation %q has empty ID in %q", entry.Op, path)
		}
		if entry.Op != "delete" && len(entry.Vec) != dim {
			return nil, fmt.Errorf("WAL operation %q for %q has dimension %d, want %d", entry.Op, entry.ID, len(entry.Vec), dim)
		}
		legacy := entry.Version == 0 && entry.Seq == 0 && entry.Checksum == ""
		if !legacy {
			if entry.Version != currentWALVersion || entry.Seq == 0 || entry.Checksum == "" {
				return nil, fmt.Errorf("invalid WAL envelope in %q", path)
			}
			want, err := walEntryChecksum(entry)
			if err != nil {
				return nil, fmt.Errorf("checksum WAL entry in %q: %w", path, err)
			}
			if entry.Checksum != want {
				return nil, fmt.Errorf("WAL checksum mismatch at sequence %d in %q", entry.Seq, path)
			}
		}
		entries = append(entries, entry)
	}
	return entries, nil
}

func (vs *VectorStore) frozenWALCoveredByLastSnapshot(path string) (bool, error) {
	entries, err := readWALFile(path, vs.Dim)
	if err != nil {
		return false, err
	}
	var maxSeq uint64
	for _, entry := range entries {
		if entry.Version == 0 {
			return false, fmt.Errorf("background checkpoint encountered a legacy WAL record")
		}
		if entry.Seq > maxSeq {
			maxSeq = entry.Seq
		}
	}
	vs.RLock()
	savedSeq := vs.lastSnapshotWALSeq
	vs.RUnlock()
	return maxSeq <= savedSeq, nil
}

func equalStringMap(left, right map[string]string) bool {
	if len(left) != len(right) {
		return false
	}
	for key, value := range left {
		if right[key] != value {
			return false
		}
	}
	return true
}

func equalFloat32Slice(left, right []float32) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if math.Float32bits(left[i]) != math.Float32bits(right[i]) {
			return false
		}
	}
	return true
}

func insertMatchesState(vs *VectorStore, entry walEntry) bool {
	hid := hashID(entry.ID)
	ix, ok := vs.idToIx[hid]
	if !ok || ix < 0 || ix >= len(vs.IDs) || vs.Deleted[hid] {
		return false
	}
	collection := entry.Coll
	if collection == "" {
		collection = "default"
	}
	tenant := entry.Tenant
	if tenant == "" {
		tenant = "default"
	}
	return vs.IDs[ix] == entry.ID &&
		vs.Docs[ix] == entry.Doc &&
		equalFloat32Slice(vs.Data[ix*vs.Dim:(ix+1)*vs.Dim], entry.Vec) &&
		equalStringMap(vs.Meta[hid], entry.Meta) &&
		vs.Coll[hid] == collection &&
		vs.TenantID[hid] == tenant
}

func applyWALEntry(vs *VectorStore, entry walEntry) (bool, error) {
	if entry.Tenant == "" {
		entry.Tenant = "default"
	}
	if entry.Coll == "" {
		entry.Coll = "default"
	}
	changed := true
	switch entry.Op {
	case "insert":
		if _, exists := vs.idToIx[hashID(entry.ID)]; exists {
			if !insertMatchesState(vs, entry) {
				return false, fmt.Errorf("conflicting replayed insert for %q", entry.ID)
			}
			changed = false
		} else if _, err := vs.Add(entry.Vec, entry.Doc, entry.ID, entry.Meta, entry.Coll, entry.Tenant); err != nil {
			return false, fmt.Errorf("replay insert %q: %w", entry.ID, err)
		}
	case "upsert":
		if _, err := vs.Upsert(entry.Vec, entry.Doc, entry.ID, entry.Meta, entry.Coll, entry.Tenant); err != nil {
			return false, fmt.Errorf("replay upsert %q: %w", entry.ID, err)
		}
	case "delete":
		hid := hashID(entry.ID)
		if _, exists := vs.idToIx[hid]; !exists || vs.Deleted[hid] {
			changed = false
		} else if err := vs.Delete(entry.ID); err != nil {
			return false, fmt.Errorf("replay delete %q: %w", entry.ID, err)
		}
	}
	if entry.NextID > vs.next {
		vs.next = entry.NextID
	} else if entry.Version == 0 && entry.NextID == 0 {
		// Historical WAL records did not persist the auto-ID counter. Recover a
		// conservative high-water from IDs produced by the old doc-N allocator so
		// WAL-only migration cannot reuse an acknowledged identifier.
		if suffix, ok := strings.CutPrefix(entry.ID, "doc-"); ok {
			if value, err := strconv.ParseInt(suffix, 10, 64); err == nil && value >= 0 && value < math.MaxInt64 && value+1 > vs.next {
				vs.next = value + 1
			}
		}
	}
	return changed, nil
}

func replayWALEntries(vs *VectorStore, batches [][]walEntry) (bool, error) {
	var sawLegacy, sawCurrent bool
	for _, entries := range batches {
		for _, entry := range entries {
			if entry.Version == 0 {
				sawLegacy = true
			} else {
				sawCurrent = true
			}
		}
	}
	if sawLegacy && sawCurrent {
		return false, fmt.Errorf("mixed legacy and sequenced WAL records require explicit recovery")
	}

	savedPath := vs.walPath
	savedHook := vs.walHook
	vs.walPath = ""
	vs.walHook = nil
	defer func() {
		vs.walPath = savedPath
		vs.walHook = savedHook
	}()

	lastSeen := uint64(0)
	expected := vs.appliedWALSeq + 1
	changed := false
	for _, entries := range batches {
		for _, entry := range entries {
			legacy := entry.Version == 0
			if !legacy {
				if lastSeen != 0 && entry.Seq <= lastSeen {
					return false, fmt.Errorf("WAL sequence is not strictly increasing: %d after %d", entry.Seq, lastSeen)
				}
				lastSeen = entry.Seq
				if entry.Seq <= vs.appliedWALSeq {
					continue
				}
				if entry.Seq != expected {
					return false, fmt.Errorf("WAL sequence gap: got %d, want %d", entry.Seq, expected)
				}
			}
			entryChanged, err := applyWALEntry(vs, entry)
			if err != nil {
				return false, err
			}
			changed = changed || entryChanged
			if legacy {
				vs.appliedWALSeq++
			} else {
				vs.appliedWALSeq = entry.Seq
				expected++
			}
			vs.nextWALSeq = vs.appliedWALSeq + 1
		}
	}
	return changed, nil
}

func walArtifactPaths(snapshotPath string) ([]string, error) {
	paths := make([]string, 0, 2)
	for _, path := range []string{snapshotPath + ".wal.frozen", snapshotPath + ".wal"} {
		if _, err := os.Stat(path); err == nil {
			paths = append(paths, path)
		} else if !os.IsNotExist(err) {
			return nil, fmt.Errorf("inspect WAL artifact %q: %w", path, err)
		}
	}
	return paths, nil
}

func existingLegacyRootArtifacts(indexPath string) ([]string, error) {
	if indexPath == "" {
		return nil, nil
	}
	paths := make([]string, 0, 3)
	for _, path := range []string{indexPath, indexPath + ".wal.frozen", indexPath + ".wal"} {
		if _, err := os.Lstat(path); err == nil {
			paths = append(paths, path)
		} else if !os.IsNotExist(err) {
			return nil, fmt.Errorf("inspect legacy root artifact %q: %w", path, err)
		}
	}
	return paths, nil
}

// checkpointWALRecovery validates every recovery artifact before applying any
// record, then commits the recovered state before removing either log.
func checkpointWALRecovery(vs *VectorStore, snapshotPath string) (bool, error) {
	paths, err := walArtifactPaths(snapshotPath)
	if err != nil || len(paths) == 0 {
		return false, err
	}
	batches := make([][]walEntry, 0, len(paths))
	for _, path := range paths {
		entries, err := readWALFile(path, vs.Dim)
		if err != nil {
			return false, err
		}
		batches = append(batches, entries)
	}
	if _, err := replayWALEntries(vs, batches); err != nil {
		return false, err
	}
	if err := vs.Save(snapshotPath); err != nil {
		return false, fmt.Errorf("checkpoint recovered WAL state: %w", err)
	}
	for _, path := range paths {
		if err := removeWALArtifact(path); err != nil {
			return false, fmt.Errorf("remove checkpointed WAL artifact: %w", err)
		}
	}
	return true, nil
}

// replayWAL is retained for focused tests and manual recovery tooling. It
// deliberately does not remove the WAL; production startup uses
// checkpointWALRecovery so cleanup happens only after a durable snapshot.
func replayWAL(vs *VectorStore) error {
	if vs.walPath == "" {
		return nil
	}
	entries, err := readWALFile(vs.walPath, vs.Dim)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	_, err = replayWALEntries(vs, [][]walEntry{entries})
	return err
}

func sortedUint64MapKeys[V any](values map[uint64]V) []uint64 {
	keys := make([]uint64, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	return keys
}

func sortedStringMapKeys[V any](values map[string]V) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

// computeChecksum returns a codec-independent digest of every persisted field
// that defines logical query state. Derived indexes and LastSaved are excluded:
// index blobs have their own format validation and timestamps are metadata, not
// database contents. Length-prefixing and sorted map keys make the hash stable
// across Go map iteration order and across Gob/Cowrie round-trips.
func (vs *VectorStore) computeChecksumForFormat(formatVersion int, includeWALHighWater bool) string {
	digest := sha256.New()
	var scalar [8]byte
	writeUint64 := func(value uint64) {
		binary.LittleEndian.PutUint64(scalar[:], value)
		_, _ = digest.Write(scalar[:])
	}
	writeInt64 := func(value int64) { writeUint64(uint64(value)) }
	writeBytes := func(value []byte) {
		writeUint64(uint64(len(value)))
		_, _ = digest.Write(value)
	}
	writeString := func(value string) { writeBytes([]byte(value)) }
	writeBool := func(value bool) {
		if value {
			writeUint64(1)
		} else {
			writeUint64(0)
		}
	}

	writeString("deepdata-logical-state-v1")
	writeString("snapshot_format_version")
	writeInt64(int64(formatVersion))
	writeString("dim")
	writeInt64(int64(vs.Dim))
	writeString("count")
	writeInt64(int64(vs.Count))
	writeString("next")
	writeInt64(vs.next)
	writeString("next_seq")
	writeUint64(vs.nextSeq)
	if includeWALHighWater {
		writeString("wal_high_water")
		writeUint64(vs.appliedWALSeq)
	}

	writeString("data")
	writeUint64(uint64(len(vs.Data)))
	for _, value := range vs.Data {
		writeUint64(uint64(math.Float32bits(value)))
	}
	writeString("docs")
	writeUint64(uint64(len(vs.Docs)))
	for _, value := range vs.Docs {
		writeString(value)
	}
	writeString("ids")
	writeUint64(uint64(len(vs.IDs)))
	for _, value := range vs.IDs {
		writeString(value)
	}
	writeString("seqs")
	writeUint64(uint64(len(vs.Seqs)))
	for _, value := range vs.Seqs {
		writeUint64(value)
	}

	writeString("meta")
	metaKeys := sortedUint64MapKeys(vs.Meta)
	writeUint64(uint64(len(metaKeys)))
	for _, id := range metaKeys {
		writeUint64(id)
		fields := sortedStringMapKeys(vs.Meta[id])
		writeUint64(uint64(len(fields)))
		for _, field := range fields {
			writeString(field)
			writeString(vs.Meta[id][field])
		}
	}
	writeString("deleted")
	deletedKeys := sortedUint64MapKeys(vs.Deleted)
	writeUint64(uint64(len(deletedKeys)))
	for _, id := range deletedKeys {
		writeUint64(id)
		writeBool(vs.Deleted[id])
	}
	writeString("collections")
	collectionKeys := sortedUint64MapKeys(vs.Coll)
	writeUint64(uint64(len(collectionKeys)))
	for _, id := range collectionKeys {
		writeUint64(id)
		writeString(vs.Coll[id])
	}
	writeString("tenants")
	tenantKeys := sortedUint64MapKeys(vs.TenantID)
	writeUint64(uint64(len(tenantKeys)))
	for _, id := range tenantKeys {
		writeUint64(id)
		writeString(vs.TenantID[id])
	}

	writeString("lex_tf")
	lexKeys := sortedUint64MapKeys(vs.lexTF)
	writeUint64(uint64(len(lexKeys)))
	for _, id := range lexKeys {
		writeUint64(id)
		terms := sortedStringMapKeys(vs.lexTF[id])
		writeUint64(uint64(len(terms)))
		for _, term := range terms {
			writeString(term)
			writeInt64(int64(vs.lexTF[id][term]))
		}
	}
	writeString("doc_len")
	docLenKeys := sortedUint64MapKeys(vs.docLen)
	writeUint64(uint64(len(docLenKeys)))
	for _, id := range docLenKeys {
		writeUint64(id)
		writeInt64(int64(vs.docLen[id]))
	}
	writeString("df")
	dfKeys := sortedStringMapKeys(vs.df)
	writeUint64(uint64(len(dfKeys)))
	for _, term := range dfKeys {
		writeString(term)
		writeInt64(int64(vs.df[term]))
	}
	writeString("sum_doc_l")
	writeInt64(int64(vs.sumDocL))

	writeString("num_meta")
	numKeys := sortedUint64MapKeys(vs.NumMeta)
	writeUint64(uint64(len(numKeys)))
	for _, id := range numKeys {
		writeUint64(id)
		fields := sortedStringMapKeys(vs.NumMeta[id])
		writeUint64(uint64(len(fields)))
		for _, field := range fields {
			writeString(field)
			writeUint64(math.Float64bits(vs.NumMeta[id][field]))
		}
	}
	writeString("time_meta")
	timeKeys := sortedUint64MapKeys(vs.TimeMeta)
	writeUint64(uint64(len(timeKeys)))
	for _, id := range timeKeys {
		writeUint64(id)
		fields := sortedStringMapKeys(vs.TimeMeta[id])
		writeUint64(uint64(len(fields)))
		for _, field := range fields {
			writeString(field)
			writeInt64(vs.TimeMeta[id][field].UTC().UnixNano())
		}
	}

	return fmt.Sprintf("sha256:%x", digest.Sum(nil))
}

func (vs *VectorStore) computeChecksum() string {
	return vs.computeChecksumForFormat(storage.CurrentFormatVersion, true)
}

// Version 3 was the first canonical snapshot checksum. It predates the WAL
// checkpoint high-water field, so its digest must be verified with the exact
// historical field set before migrating to the current format.
func (vs *VectorStore) computeV3Checksum() string {
	return vs.computeChecksumForFormat(3, false)
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
	// `routes` prints the contract's HTTP surface and exits. It reads only
	// the embedded operations list, so it needs no data directory, no
	// environment and no server; the docs linter runs it to generate the
	// route table in internal/collection/API.md (DOC-03).
	if len(args) > 0 && args[0] == "routes" {
		if err := printRoutes(os.Stdout); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
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

	// The production server exposes only the caller-supplied-vector V3/gRPC
	// collection engine. Historical handlers remain in source for offline
	// migration tests, but no runtime environment switch may re-enable them in
	// the RC binary.
	const canonicalOnly = true
	if canonicalOnly {
		if err := validateCanonicalAuthEnvironment(); err != nil {
			logger.Error("canonical authentication configuration rejected", "error", err)
			os.Exit(1)
		}
		configuredMode := strings.ToLower(strings.TrimSpace(os.Getenv("VECTORDB_MODE")))
		if configuredMode == "" {
			if err := os.Setenv("VECTORDB_MODE", string(ModeLocal)); err != nil {
				logger.Error("failed to select canonical local data path", "error", err)
				os.Exit(1)
			}
		} else if configuredMode != string(ModeLocal) {
			logger.Error("canonical RC accepts caller-supplied vectors and supports only the local persistence path", "VECTORDB_MODE", configuredMode)
			os.Exit(1)
		}
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

	// One text embedder per process, named by DEEPDATA_EMBEDDER (default none:
	// callers send vectors). A configured-but-unreachable embedder refuses to
	// start, like unreadable persistence below.
	var embedder Embedder
	if canonicalOnly {
		serverEmb, embErr := newServerEmbedderFromEnv()
		if embErr != nil {
			logger.Error("refusing to start with an unusable text embedder", "error", embErr)
			os.Exit(1)
		}
		if serverEmb != nil {
			embedder = serverEmb
			logger.Info("text embedder ready", "embedder", serverEmb.Label(), "dim", serverEmb.Dim())
		} else {
			logger.Info("no text embedder configured (DEEPDATA_EMBEDDER=none); clients must provide vectors")
		}
	}

	if canonicalOnly {
		legacyArtifacts, inspectErr := existingLegacyRootArtifacts(indexPath)
		if inspectErr != nil {
			logger.Error("failed to inspect unsupported legacy persistence", "error", inspectErr)
			os.Exit(1)
		}
		if len(legacyArtifacts) > 0 {
			logger.Error("legacy root persistence requires an explicit offline migration before canonical RC startup", "artifacts", legacyArtifacts)
			os.Exit(1)
		}
	}

	// The V3 surface keeps its authentication and limit state here; the legacy
	// engine is never constructed.
	rt := newServerRuntime()

	// HTTP API with graceful shutdown
	handler, collectionHTTP := newCanonicalHTTPHandler(rt, embedder, nil, indexPath)
	if err := collectionHTTP.PersistenceError(); err != nil {
		logger.Error("refusing to start with unreadable collection persistence state", "path", indexPath+".collections", "error", err)
		os.Exit(1)
	}
	if canonicalOnly {
		legacyCollectionCount, inspectErr := collectionHTTP.LegacyCollectionCount()
		if inspectErr != nil {
			logger.Error("failed to inspect unified collection state", "error", inspectErr)
			if abortErr := collectionHTTP.Abort(); abortErr != nil {
				logger.Error("failed to release collection store after inspection failure", "error", abortErr)
			}
			os.Exit(1)
		}
		if legacyCollectionCount != 0 {
			logger.Error("refusing canonical startup with legacy V2 collections; migrate them into tenant-aware V3 collections first",
				"path", indexPath+".collections", "legacy_collections", legacyCollectionCount)
			if abortErr := collectionHTTP.Abort(); abortErr != nil {
				logger.Error("failed to release collection store after migration refusal", "error", abortErr)
			}
			os.Exit(1)
		}
	}
	addr, grpcAddr, err := canonicalListenerAddresses(
		envInt("PORT", 8080),
		envInt("GRPC_PORT", 50051),
		os.Getenv("DEEPDATA_INSECURE_DEV_MODE") == "1",
		os.Getenv("DEEPDATA_BIND_HOST"),
	)
	if err != nil {
		logger.Error("refusing invalid API bind host", "error", err)
		if closeErr := collectionHTTP.Abort(); closeErr != nil {
			logger.Error("failed to release collection store after bind-host refusal", "error", closeErr)
		}
		os.Exit(1)
	}
	httpListener, grpcListener, err := bindAPIListeners(addr, grpcAddr)
	if err != nil {
		logger.Error("refusing to start without the complete API listener set", "error", err)
		if closeErr := collectionHTTP.Abort(); closeErr != nil {
			logger.Error("failed to release collection store after listener failure", "error", closeErr)
		}
		os.Exit(1)
	}

	// Wrap handler with h2c (HTTP/2 cleartext) for connection multiplexing
	// without TLS. HTTP/1.1 clients continue to work transparently.
	// Set HTTP_H2C=0 to disable.
	var finalHandler http.Handler = handler
	if os.Getenv("HTTP_H2C") != "0" {
		finalHandler = h2c.NewHandler(handler, &http2.Server{})
	}
	var httpRequests sync.WaitGroup
	trackedHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		httpRequests.Add(1)
		defer httpRequests.Done()
		finalHandler.ServeHTTP(w, r)
	})

	srv := &http.Server{
		Addr:              addr,
		Handler:           trackedHandler,
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       time.Duration(envInt("HTTP_READ_TIMEOUT_SEC", 60)) * time.Second,
		WriteTimeout:      time.Duration(envInt("HTTP_WRITE_TIMEOUT_SEC", 300)) * time.Second,
		IdleTimeout:       120 * time.Second,
		MaxHeaderBytes:    1 << 20,
	}

	// gRPC server (GRPC_PORT=0 to disable, default 50051)
	var grpcSrv *grpc.Server
	if grpcListener != nil {
		grpcSrv = grpc.NewServer(
			grpc.MaxRecvMsgSize(canonicalGRPCMaxReceiveBytes),
			grpc.MaxSendMsgSize(64*1024*1024),
			grpc.UnaryInterceptor(rt.grpcInterceptor(logger)),
		)
		deepdatav3.RegisterDeepDataServer(grpcSrv, &CollectionGRPCServer{
			tenants:  collectionHTTP.TenantManager(),
			embedder: collectionHTTP.embedder,
			persistenceHealth: func() error {
				if !collectionHTTP.IsDurable() {
					return errors.New("durable collection persistence is not initialized")
				}
				return collectionHTTP.PersistenceError()
			},
		})
	}

	type apiServeFailure struct {
		surface string
		err     error
	}
	serverErrCh := make(chan apiServeFailure, 2)
	logger.Info("http api listening", "addr", httpListener.Addr())
	go func() {
		if err := srv.Serve(httpListener); err != nil && !errors.Is(err, http.ErrServerClosed) {
			serverErrCh <- apiServeFailure{surface: "http", err: err}
		}
	}()
	if grpcSrv != nil {
		logger.Info("grpc api listening", "addr", grpcListener.Addr())
		go func() {
			if err := grpcSrv.Serve(grpcListener); err != nil && !errors.Is(err, grpc.ErrServerStopped) {
				serverErrCh <- apiServeFailure{surface: "grpc", err: err}
			}
		}()
	}

	// Setup graceful shutdown
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGTERM, syscall.SIGINT)

	// Wait for a shutdown signal or any unexpected listener/server failure.
	logging.Default().Info("server running, press Ctrl+C to stop")
	serveFailed := false
	select {
	case sig := <-sigCh:
		logging.Default().Info("received signal, initiating graceful shutdown", "signal", sig)
	case failure := <-serverErrCh:
		serveFailed = true
		logging.Default().Error("API server failed; initiating coordinated shutdown", "surface", failure.surface, "error", failure.err)
	}
	signal.Stop(sigCh)

	// Graceful shutdown sequence. The final collection checkpoint runs only
	// when every handler has drained.
	allHandlersDrained := true
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
			select {
			case <-grpcDone:
			case <-time.After(5 * time.Second):
				allHandlersDrained = false
				logging.Default().Error("gRPC handlers did not drain after forced stop")
			}
		}
	}
	logging.Default().Info("shutting down HTTP server")
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		logging.Default().Error("HTTP server shutdown error; forcing connection close", "error", err)
		if closeErr := srv.Close(); closeErr != nil && closeErr != http.ErrServerClosed {
			logging.Default().Error("HTTP server forced close error", "error", closeErr)
		}
	}
	httpDone := make(chan struct{})
	go func() {
		httpRequests.Wait()
		close(httpDone)
	}()
	select {
	case <-httpDone:
	case <-time.After(5 * time.Second):
		allHandlersDrained = false
		logging.Default().Error("HTTP handlers did not drain after shutdown")
	}

	if allHandlersDrained {
		if err := collectionHTTP.Close(); err != nil {
			logger.Error("failed to checkpoint and close canonical collection state", "error", err)
		} else {
			logger.Info("canonical collection state checkpointed and closed successfully")
		}
	} else {
		logger.Error("skipping final persistence checkpoint because handlers are still active; WAL artifacts retained")
	}

	logger.Info("shutdown complete")
	if serveFailed {
		logger.Error("exiting non-zero after API server failure")
		os.Exit(1)
	}
}

// bindAPIListeners proves the complete configured API surface is available
// before either protocol begins serving. If the second bind fails, the first
// listener is closed so a replacement process can start immediately.
func bindAPIListeners(httpAddr, grpcAddr string) (net.Listener, net.Listener, error) {
	httpListener, err := net.Listen("tcp", httpAddr)
	if err != nil {
		return nil, nil, fmt.Errorf("bind HTTP listener %q: %w", httpAddr, err)
	}
	if grpcAddr == "" {
		return httpListener, nil, nil
	}
	grpcListener, err := net.Listen("tcp", grpcAddr)
	if err != nil {
		return nil, nil, errors.Join(
			fmt.Errorf("bind gRPC listener %q: %w", grpcAddr, err),
			httpListener.Close(),
		)
	}
	return httpListener, grpcListener, nil
}

// canonicalListenerAddresses keeps the explicit credentialless development
// escape hatch loopback-only. Authenticated deployments retain wildcard binds
// by default so containers and orchestrators can publish the configured ports,
// while DEEPDATA_BIND_HOST lets an operator reduce exposure to one IP literal.
func canonicalListenerAddresses(httpPort, grpcPort int, insecureDevelopment bool, configuredHost string) (string, string, error) {
	host := ""
	if configuredHost != strings.TrimSpace(configuredHost) {
		return "", "", errors.New("DEEPDATA_BIND_HOST must not contain surrounding whitespace")
	}
	if configuredHost != "" {
		ip := net.ParseIP(configuredHost)
		if ip == nil {
			return "", "", fmt.Errorf("DEEPDATA_BIND_HOST=%q must be an IP literal", configuredHost)
		}
		if insecureDevelopment && !ip.IsLoopback() {
			return "", "", errors.New("DEEPDATA_INSECURE_DEV_MODE may bind only to a loopback IP")
		}
		host = configuredHost
	} else if insecureDevelopment {
		host = "127.0.0.1"
	}
	httpAddr := net.JoinHostPort(host, strconv.Itoa(httpPort))
	grpcAddr := ""
	if grpcPort > 0 {
		grpcAddr = net.JoinHostPort(host, strconv.Itoa(grpcPort))
	}
	return httpAddr, grpcAddr, nil
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
	checkNonNegInt("GRPC_PORT")
	checkNonNegInt("VECTOR_CAPACITY")
	checkPosInt64("WAL_MAX_BYTES")
	checkPosInt("WAL_MAX_OPS")
	checkPosInt("HNSW_M")
	checkPosFloat("HNSW_ML")
	checkPosInt("HNSW_EFSEARCH")
	checkPosInt("API_RPS")
	checkPosInt("MAX_RATE_LIMIT_KEYS")
	checkPosInt("AUTH_FAILURE_RPS")
	checkPosInt("AUTH_FAILURE_BURST")
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

func validateCanonicalAuthEnvironment() error {
	apiToken := os.Getenv("API_TOKEN")
	jwtSecret := os.Getenv("JWT_SECRET")
	hasStaticToken := apiToken != ""
	hasJWTSecret := jwtSecret != ""
	if hasStaticToken && hasJWTSecret {
		return errors.New("configure exactly one of API_TOKEN or JWT_SECRET; combined credential modes are unsupported")
	}
	if hasStaticToken {
		if err := validateCanonicalCredential("API_TOKEN", apiToken); err != nil {
			return err
		}
	}
	if hasJWTSecret {
		if err := validateCanonicalCredential("JWT_SECRET", jwtSecret); err != nil {
			return err
		}
	}
	if !hasStaticToken && !hasJWTSecret && os.Getenv("DEEPDATA_INSECURE_DEV_MODE") != "1" {
		return errors.New("API_TOKEN or JWT_SECRET is required; set DEEPDATA_INSECURE_DEV_MODE=1 only for isolated development")
	}
	return nil
}

const canonicalCredentialMinBytes = 32

func validateCanonicalCredential(name, value string) error {
	if strings.TrimSpace(value) != value {
		return fmt.Errorf("%s must not contain leading or trailing whitespace", name)
	}
	if len([]byte(value)) < canonicalCredentialMinBytes {
		return fmt.Errorf("%s must be at least %d bytes", name, canonicalCredentialMinBytes)
	}
	return nil
}

// grpcAuthInterceptor returns a gRPC unary interceptor that mirrors the HTTP
// guard middleware: JWT validation, legacy API-token checking, and requireAuth
// enforcement. On success it injects a *security.TenantContext into the
// context so downstream handlers can inspect tenant identity and permissions.
func grpcAuthInterceptor(jwtMgr *security.JWTManager, apiToken string, requireAuth bool, logger *logging.Logger) grpc.UnaryServerInterceptor {
	return grpcAuthInterceptorWithRateLimiters(jwtMgr, apiToken, requireAuth, logger, nil, nil)
}

func grpcAuthInterceptorWithTenantLimiter(jwtMgr *security.JWTManager, apiToken string, requireAuth bool, logger *logging.Logger, tenantLimiter *rateLimiter) grpc.UnaryServerInterceptor {
	return grpcAuthInterceptorWithRateLimiters(jwtMgr, apiToken, requireAuth, logger, tenantLimiter, nil)
}

func grpcAuthInterceptorWithRateLimiters(
	jwtMgr *security.JWTManager,
	apiToken string,
	requireAuth bool,
	logger *logging.Logger,
	tenantLimiter *rateLimiter,
	authFailureLimiter *authFailureLimiter,
) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (resp any, err error) {
		// Panic recovery — same as before, prevents crashes from taking down the process
		defer func() {
			if r := recover(); r != nil {
				logger.Error("panic recovered in gRPC handler", "error", r, "method", info.FullMethod)
				err = apierror.New(apierror.CodeInternal, "internal error").GRPC(ctx)
			}
		}()

		// Auth token and request id from gRPC metadata (mirrors the HTTP
		// Authorization header and X-Request-ID middleware): honour the
		// caller's x-request-id or mint one, echo it as a response header and
		// carry it in ctx so every apierror quotes it.
		token, requestID := "", ""
		if md, ok := metadata.FromIncomingContext(ctx); ok {
			if vals := md.Get("authorization"); len(vals) > 0 {
				token = strings.TrimPrefix(vals[0], "Bearer ")
			}
			if vals := md.Get("x-request-id"); len(vals) > 0 {
				requestID = truncateRequestID(strings.TrimSpace(vals[0]), 128)
			}
		}
		if requestID == "" {
			requestID = generateRequestID()
		}
		ctx = context.WithValue(ctx, logging.RequestIDKey, requestID)
		_ = grpc.SetHeader(ctx, metadata.Pairs("x-request-id", requestID))
		authPeerKey := grpcAuthPeerKey(ctx)
		authAttempt, allowed := authFailureLimiter.begin(authPeerKey)
		if !allowed {
			return nil, apierror.New(apierror.CodeRateLimited, "authentication rate limited").GRPC(ctx)
		}
		finishAuthAttempt := func(failed bool) {
			if authAttempt != nil {
				authAttempt.finish(failed)
				authAttempt = nil
			}
		}
		defer func() { finishAuthAttempt(false) }()

		var tenantCtx *security.TenantContext

		if jwtMgr != nil {
			if token == "" {
				if requireAuth {
					finishAuthAttempt(true)
					return nil, apierror.New(apierror.CodeUnauthenticated, "missing authentication token").GRPC(ctx)
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
					logging.Default().Error("gRPC JWT validation failed", "error", valErr)
					finishAuthAttempt(true)
					return nil, apierror.New(apierror.CodeUnauthenticated, "invalid token").GRPC(ctx)
				}
			}
		} else {
			authenticated := false
			if apiToken != "" {
				if security.SecureCompare(token, apiToken) {
					authenticated = true
				} else if token != "" {
					finishAuthAttempt(true)
					return nil, apierror.New(apierror.CodeUnauthenticated, "unauthorized").GRPC(ctx)
				}
			}
			if requireAuth && !authenticated {
				finishAuthAttempt(true)
				return nil, apierror.New(apierror.CodeUnauthenticated, "unauthorized").GRPC(ctx)
			}
			serverAdmin := authenticated || (jwtMgr == nil && apiToken == "")
			tenantCtx = &security.TenantContext{
				TenantID:    "default",
				Permissions: map[string]bool{"read": true, "write": true},
				Collections: make(map[string]bool),
				// A configured static server token is intentionally full control;
				// JWT claims provide scoped tenant/collection roles.
				IsAdmin:       serverAdmin,
				IsServerAdmin: serverAdmin,
			}
		}
		finishAuthAttempt(false)

		if tenantLimiter != nil {
			targetTenant := ""
			if request, ok := req.(interface{ GetTenantId() string }); ok {
				targetTenant = request.GetTenantId()
			}
			tenantKey := canonicalRateLimitTenant(tenantCtx, targetTenant)
			if !tenantLimiter.allow(tenantKey) {
				return nil, apierror.New(apierror.CodeRateLimited, "tenant rate limited").GRPC(ctx)
			}
		}

		ctx = context.WithValue(ctx, security.TenantContextKey, tenantCtx)
		return handler(ctx, req)
	}
}
