package main

import (
	"fmt"
	"sync"
)

// ===========================================================================================
// PRE-FILTERING WITH METADATA BITMAP INDEX
// Filter documents BEFORE HNSW search for 5-10x speedup
// ===========================================================================================

// SimpleBitmap represents a simple bitmap for document IDs
// In production, use github.com/RoaringBitmap/roaring for efficiency
type SimpleBitmap struct {
	bits map[uint64]bool
}

// NewSimpleBitmap creates a new bitmap
func NewSimpleBitmap() *SimpleBitmap {
	return &SimpleBitmap{
		bits: make(map[uint64]bool),
	}
}

// Add adds a document ID to the bitmap
func (sb *SimpleBitmap) Add(docID uint64) {
	sb.bits[docID] = true
}

// Remove removes a document ID from the bitmap
func (sb *SimpleBitmap) Remove(docID uint64) {
	delete(sb.bits, docID)
}

// Contains checks if a document ID is in the bitmap
func (sb *SimpleBitmap) Contains(docID uint64) bool {
	return sb.bits[docID]
}

// And performs bitwise AND with another bitmap
func (sb *SimpleBitmap) And(other *SimpleBitmap) *SimpleBitmap {
	result := NewSimpleBitmap()
	for id := range sb.bits {
		if other.Contains(id) {
			result.Add(id)
		}
	}
	return result
}

// Or performs bitwise OR with another bitmap
func (sb *SimpleBitmap) Or(other *SimpleBitmap) *SimpleBitmap {
	result := NewSimpleBitmap()
	for id := range sb.bits {
		result.Add(id)
	}
	for id := range other.bits {
		result.Add(id)
	}
	return result
}

// Count returns the number of set bits
func (sb *SimpleBitmap) Count() int {
	return len(sb.bits)
}

// ToSlice returns a slice of document IDs
func (sb *SimpleBitmap) ToSlice() []uint64 {
	ids := make([]uint64, 0, len(sb.bits))
	for id := range sb.bits {
		ids = append(ids, id)
	}
	return ids
}

// Clone creates a copy of the bitmap
func (sb *SimpleBitmap) Clone() *SimpleBitmap {
	clone := NewSimpleBitmap()
	for id := range sb.bits {
		clone.Add(id)
	}
	return clone
}

// ===========================================================================================
// METADATA INDEX
// ===========================================================================================

// MetadataIndex maintains an inverted index for metadata filtering
// Structure: metaKey -> metaValue -> bitmap of document IDs
type MetadataIndex struct {
	mu sync.RWMutex

	// Inverted index: meta_key -> meta_value -> set of doc IDs
	index map[string]map[string]*SimpleBitmap

	// Reverse index: doc_ID -> metadata (for updates)
	docMeta map[uint64]map[string]string

	// All document IDs (for "no filter" case)
	allDocs *SimpleBitmap

	// Statistics
	stats *PreFilterStats
}

// NewMetadataIndex creates a new metadata index
func NewMetadataIndex() *MetadataIndex {
	return &MetadataIndex{
		index:   make(map[string]map[string]*SimpleBitmap),
		docMeta: make(map[uint64]map[string]string),
		allDocs: NewSimpleBitmap(),
		stats:   NewPreFilterStats(),
	}
}

// AddDocument adds a document with metadata to the index
func (mi *MetadataIndex) AddDocument(docID uint64, metadata map[string]string) {
	mi.mu.Lock()
	defer mi.mu.Unlock()

	// Add to all docs
	mi.allDocs.Add(docID)

	// Store document metadata
	mi.docMeta[docID] = metadata

	// Add to inverted index
	for key, value := range metadata {
		// Ensure key exists
		if _, ok := mi.index[key]; !ok {
			mi.index[key] = make(map[string]*SimpleBitmap)
		}

		// Ensure value exists
		if _, ok := mi.index[key][value]; !ok {
			mi.index[key][value] = NewSimpleBitmap()
		}

		// Add document to bitmap
		mi.index[key][value].Add(docID)
	}

	mi.stats.RecordIndexSize(mi.getIndexSize())
}

// RemoveDocument removes a document from the index
func (mi *MetadataIndex) RemoveDocument(docID uint64) {
	mi.mu.Lock()
	defer mi.mu.Unlock()

	// Get document metadata
	metadata, ok := mi.docMeta[docID]
	if !ok {
		return // Document not in index
	}

	// Remove from all docs
	mi.allDocs.Remove(docID)

	// Remove from inverted index
	for key, value := range metadata {
		if keyMap, ok := mi.index[key]; ok {
			if bitmap, ok := keyMap[value]; ok {
				bitmap.Remove(docID)

				// Cleanup empty bitmaps
				if bitmap.Count() == 0 {
					delete(keyMap, value)
				}
			}

			// Cleanup empty key maps
			if len(keyMap) == 0 {
				delete(mi.index, key)
			}
		}
	}

	// Remove from document metadata
	delete(mi.docMeta, docID)

	mi.stats.RecordIndexSize(mi.getIndexSize())
}

// UpdateDocument updates a document's metadata
func (mi *MetadataIndex) UpdateDocument(docID uint64, newMetadata map[string]string) {
	mi.RemoveDocument(docID)
	mi.AddDocument(docID, newMetadata)
}

// GetMatchingDocs returns document IDs matching ALL filter conditions
func (mi *MetadataIndex) GetMatchingDocs(filter map[string]string) *SimpleBitmap {
	mi.mu.RLock()
	defer mi.mu.RUnlock()

	// No filter = all docs
	if len(filter) == 0 {
		mi.stats.RecordLookup(mi.allDocs.Count(), 1.0)
		return mi.allDocs.Clone()
	}

	// Start with all documents
	result := mi.allDocs.Clone()

	// Apply each filter condition (AND logic)
	for key, value := range filter {
		// Get bitmap for this key-value pair
		if keyMap, ok := mi.index[key]; ok {
			if bitmap, ok := keyMap[value]; ok {
				// Intersection with current result
				result = result.And(bitmap)
			} else {
				// Value doesn't exist - no matches
				mi.stats.RecordLookup(0, 0.0)
				return NewSimpleBitmap()
			}
		} else {
			// Key doesn't exist - no matches
			mi.stats.RecordLookup(0, 0.0)
			return NewSimpleBitmap()
		}
	}

	// Calculate selectivity
	selectivity := float64(result.Count()) / float64(mi.allDocs.Count())
	mi.stats.RecordLookup(result.Count(), selectivity)

	return result
}

// GetMatchingDocsOR returns document IDs matching ANY filter condition
func (mi *MetadataIndex) GetMatchingDocsOR(filter map[string]string) *SimpleBitmap {
	mi.mu.RLock()
	defer mi.mu.RUnlock()

	// No filter = all docs
	if len(filter) == 0 {
		return mi.allDocs.Clone()
	}

	// Start with empty bitmap
	result := NewSimpleBitmap()

	// Apply each filter condition (OR logic)
	for key, value := range filter {
		if keyMap, ok := mi.index[key]; ok {
			if bitmap, ok := keyMap[value]; ok {
				// Union with current result
				result = result.Or(bitmap)
			}
		}
	}

	selectivity := float64(result.Count()) / float64(mi.allDocs.Count())
	mi.stats.RecordLookup(result.Count(), selectivity)

	return result
}

// GetDocumentCount returns the total number of documents in the index
func (mi *MetadataIndex) GetDocumentCount() int {
	mi.mu.RLock()
	defer mi.mu.RUnlock()
	return mi.allDocs.Count()
}

// GetMetadataKeys returns all metadata keys in the index
func (mi *MetadataIndex) GetMetadataKeys() []string {
	mi.mu.RLock()
	defer mi.mu.RUnlock()

	keys := make([]string, 0, len(mi.index))
	for key := range mi.index {
		keys = append(keys, key)
	}
	return keys
}

// GetMetadataValues returns all values for a given metadata key
func (mi *MetadataIndex) GetMetadataValues(key string) []string {
	mi.mu.RLock()
	defer mi.mu.RUnlock()

	if keyMap, ok := mi.index[key]; ok {
		values := make([]string, 0, len(keyMap))
		for value := range keyMap {
			values = append(values, value)
		}
		return values
	}

	return nil
}

// GetStats returns pre-filtering statistics
func (mi *MetadataIndex) GetStats() map[string]any {
	return mi.stats.GetStats()
}

// getIndexSize calculates the current index size (for statistics)
func (mi *MetadataIndex) getIndexSize() int {
	size := 0
	for _, keyMap := range mi.index {
		size += len(keyMap)
	}
	return size
}

// ===========================================================================================
// PRE-FILTER STATISTICS
// ===========================================================================================

// PreFilterStats tracks pre-filtering statistics
type PreFilterStats struct {
	mu sync.RWMutex

	TotalLookups        int64
	TotalMatchedDocs    int64
	AverageSelectivity  float64
	TotalSelectivity    float64
	IndexSize           int
	HighSelectivity     int64 // >50% of docs matched
	MediumSelectivity   int64 // 10-50%
	LowSelectivity      int64 // 1-10%
	VeryLowSelectivity  int64 // <1%
}

// NewPreFilterStats creates a new stats tracker
func NewPreFilterStats() *PreFilterStats {
	return &PreFilterStats{}
}

// RecordLookup records a filter lookup
func (pfs *PreFilterStats) RecordLookup(matchedDocs int, selectivity float64) {
	pfs.mu.Lock()
	defer pfs.mu.Unlock()

	pfs.TotalLookups++
	pfs.TotalMatchedDocs += int64(matchedDocs)
	pfs.TotalSelectivity += selectivity
	pfs.AverageSelectivity = pfs.TotalSelectivity / float64(pfs.TotalLookups)

	// Categorize selectivity
	if selectivity > 0.5 {
		pfs.HighSelectivity++
	} else if selectivity > 0.1 {
		pfs.MediumSelectivity++
	} else if selectivity > 0.01 {
		pfs.LowSelectivity++
	} else {
		pfs.VeryLowSelectivity++
	}
}

// RecordIndexSize records the current index size
func (pfs *PreFilterStats) RecordIndexSize(size int) {
	pfs.mu.Lock()
	defer pfs.mu.Unlock()
	pfs.IndexSize = size
}

// GetStats returns current statistics
func (pfs *PreFilterStats) GetStats() map[string]any {
	pfs.mu.RLock()
	defer pfs.mu.RUnlock()

	avgMatchedDocs := float64(0)
	if pfs.TotalLookups > 0 {
		avgMatchedDocs = float64(pfs.TotalMatchedDocs) / float64(pfs.TotalLookups)
	}

	return map[string]any{
		"total_lookups":         pfs.TotalLookups,
		"total_matched_docs":    pfs.TotalMatchedDocs,
		"average_matched_docs":  avgMatchedDocs,
		"average_selectivity":   pfs.AverageSelectivity,
		"index_size":            pfs.IndexSize,
		"high_selectivity":      pfs.HighSelectivity,
		"medium_selectivity":    pfs.MediumSelectivity,
		"low_selectivity":       pfs.LowSelectivity,
		"very_low_selectivity":  pfs.VeryLowSelectivity,
	}
}

// ===========================================================================================
// USAGE EXAMPLE
// ===========================================================================================

/*
Example usage:

// Create metadata index
metaIndex := NewMetadataIndex()

// Add documents with metadata
metaIndex.AddDocument(1, map[string]string{
    "category": "python",
    "year": "2024",
    "author": "alice",
})

metaIndex.AddDocument(2, map[string]string{
    "category": "python",
    "year": "2023",
    "author": "bob",
})

metaIndex.AddDocument(3, map[string]string{
    "category": "golang",
    "year": "2024",
    "author": "alice",
})

// Query: Find all Python docs from 2024
filter := map[string]string{
    "category": "python",
    "year": "2024",
}

matchingDocs := metaIndex.GetMatchingDocs(filter)
// Result: [1] (only doc 1 matches both conditions)

// Now search ONLY within filtered set (much faster!)
// Instead of searching all 1M docs, search only matched 10k docs

// Performance impact:
// Without pre-filtering: Search 1M docs → 50ms
// With pre-filtering: Search 10k docs → 5ms (10x faster!)
*/

// ===========================================================================================
// FILTER MODE
// ===========================================================================================

// FilterMode defines how to apply metadata filters
type FilterMode int

const (
	// NoFilter: No filtering (search all docs)
	NoFilter FilterMode = iota

	// PreFilter: Filter BEFORE HNSW search (fast, recommended)
	PreFilter

	// PostFilter: Filter AFTER HNSW search (slow, current approach)
	PostFilter
)

func (fm FilterMode) String() string {
	switch fm {
	case NoFilter:
		return "no_filter"
	case PreFilter:
		return "pre_filter"
	case PostFilter:
		return "post_filter"
	default:
		return "unknown"
	}
}

// FilterConfig configures filtering behavior
type FilterConfig struct {
	Mode FilterMode // Filter mode (pre/post/none)

	// Pre-filter optimization settings
	MaxCandidates  int     // Max candidates to consider (default: 100k)
	MinSelectivity float64 // Min selectivity to use pre-filter (default: 0.01 = 1%)

	// If filter selectivity < MinSelectivity, fall back to no filter
	// (too few matches = overhead not worth it)
}

// DefaultFilterConfig returns default filter configuration
func DefaultFilterConfig() FilterConfig {
	return FilterConfig{
		Mode:           PreFilter,
		MaxCandidates:  100000,
		MinSelectivity: 0.01, // 1%
	}
}

// ShouldUsePreFilter determines if pre-filtering should be used
func (fc *FilterConfig) ShouldUsePreFilter(
	matchedDocs int,
	totalDocs int,
) bool {
	if fc.Mode != PreFilter {
		return false
	}

	// Calculate selectivity
	selectivity := float64(matchedDocs) / float64(totalDocs)

	// Too few matches? Not worth the overhead
	if selectivity < fc.MinSelectivity {
		return false
	}

	// Too many matches? Pre-filter won't help much
	if matchedDocs > fc.MaxCandidates {
		// Still use pre-filter if it significantly reduces search space
		return selectivity < 0.5
	}

	return true
}

// EstimateSpeedup estimates the speedup from pre-filtering
func (fc *FilterConfig) EstimateSpeedup(selectivity float64) float64 {
	if fc.Mode != PreFilter || selectivity >= 1.0 {
		return 1.0 // No speedup
	}

	// Approximate speedup based on selectivity
	// 0.1% selectivity = ~10x speedup
	// 1% selectivity = ~5x speedup
	// 10% selectivity = ~2x speedup
	// 50% selectivity = ~1.25x speedup

	if selectivity < 0.001 {
		return 10.0
	} else if selectivity < 0.01 {
		return 5.0 + (5.0 * (0.01 - selectivity) / 0.009)
	} else if selectivity < 0.1 {
		return 2.0 + (3.0 * (0.1 - selectivity) / 0.09)
	} else if selectivity < 0.5 {
		return 1.25 + (0.75 * (0.5 - selectivity) / 0.4)
	} else {
		return 1.0 + (0.25 * (1.0 - selectivity) / 0.5)
	}
}

// AnalyzeFilter analyzes a filter and returns optimization recommendations
func (mi *MetadataIndex) AnalyzeFilter(filter map[string]string) map[string]any {
	matchedDocs := mi.GetMatchingDocs(filter)
	totalDocs := mi.GetDocumentCount()

	selectivity := float64(0)
	if totalDocs > 0 {
		selectivity = float64(matchedDocs.Count()) / float64(totalDocs)
	}

	config := DefaultFilterConfig()
	shouldUsePreFilter := config.ShouldUsePreFilter(matchedDocs.Count(), totalDocs)
	estimatedSpeedup := config.EstimateSpeedup(selectivity)

	return map[string]any{
		"matched_docs":          matchedDocs.Count(),
		"total_docs":            totalDocs,
		"selectivity":           selectivity,
		"should_use_prefilter":  shouldUsePreFilter,
		"estimated_speedup":     fmt.Sprintf("%.1fx", estimatedSpeedup),
		"recommendation":        getFilterRecommendation(selectivity),
	}
}

func getFilterRecommendation(selectivity float64) string {
	if selectivity < 0.001 {
		return "Excellent candidate for pre-filtering (10x+ speedup expected)"
	} else if selectivity < 0.01 {
		return "Very good candidate for pre-filtering (5-10x speedup expected)"
	} else if selectivity < 0.1 {
		return "Good candidate for pre-filtering (2-5x speedup expected)"
	} else if selectivity < 0.5 {
		return "Moderate candidate for pre-filtering (1.25-2x speedup expected)"
	} else {
		return "Pre-filtering may not provide significant speedup"
	}
}
