package sparse

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"sync"
)

// Posting represents a document-score pair in an inverted list.
type Posting struct {
	DocID uint64
	Score float32
}

// InvertedIndex implements sparse vector search using inverted lists.
// This is the core data structure for BM25-style keyword search.
//
// Architecture:
//   - postings: term_id -> list of (doc_id, score) pairs
//   - docNorms: document L2 norms for normalization
//   - docLengths: document lengths (sum of values) for BM25
//   - avgDocLength: average document length across corpus
type InvertedIndex struct {
	mu sync.RWMutex

	dim            int                         // Maximum dimension
	postings       map[uint32][]Posting        // term_id -> postings list
	docNorms       map[uint64]float32          // doc_id -> L2 norm
	docLengths     map[uint64]float32          // doc_id -> sum of values
	docVectors     map[uint64]*SparseVector    // doc_id -> original vector
	avgDocLength   float32                     // average document length
	totalDocs      int                         // total documents indexed
	totalDocLength float32                     // sum of all document lengths (for incremental avgDocLength)
	totalPostings  int                         // total postings across all terms (for O(1) Stats)
	termDocCounts  map[uint32]int              // term_id -> number of docs containing term

	// BM25 parameters
	k1 float32 // Term frequency saturation (default: 1.2)
	b  float32 // Length normalization (default: 0.75)
}

// NewInvertedIndex creates a new inverted index for sparse vectors.
func NewInvertedIndex(dim int) *InvertedIndex {
	return &InvertedIndex{
		dim:           dim,
		postings:      make(map[uint32][]Posting),
		docNorms:      make(map[uint64]float32),
		docLengths:    make(map[uint64]float32),
		docVectors:    make(map[uint64]*SparseVector),
		termDocCounts: make(map[uint32]int),
		k1:            1.2,
		b:             0.75,
	}
}

// SetBM25Params sets the BM25 parameters.
func (idx *InvertedIndex) SetBM25Params(k1, b float32) {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	idx.k1 = k1
	idx.b = b
}

// Add inserts a sparse vector into the index.
func (idx *InvertedIndex) Add(ctx context.Context, docID uint64, vector *SparseVector) error {
	if vector.Dim != idx.dim {
		return fmt.Errorf("dimension mismatch: expected %d, got %d", idx.dim, vector.Dim)
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Check if document already exists
	if _, exists := idx.docVectors[docID]; exists {
		return fmt.Errorf("document %d already indexed", docID)
	}

	// Store vector and compute norms
	idx.docVectors[docID] = vector.Clone()
	idx.docNorms[docID] = vector.Norm()

	// Compute document length (sum of absolute values)
	docLength := float32(0)
	for _, v := range vector.Values {
		docLength += float32(math.Abs(float64(v)))
	}
	idx.docLengths[docID] = float32(docLength)

	// Add to inverted lists
	for i, termID := range vector.Indices {
		score := vector.Values[i]

		// Initialize posting list if needed
		if _, exists := idx.postings[termID]; !exists {
			idx.postings[termID] = make([]Posting, 0, 8)
			idx.termDocCounts[termID] = 0
		}

		// Add posting
		idx.postings[termID] = append(idx.postings[termID], Posting{
			DocID: docID,
			Score: score,
		})

		// Increment term document count
		idx.termDocCounts[termID]++
	}

	// Update statistics incrementally (O(1) instead of O(n))
	idx.totalDocs++
	idx.totalDocLength += docLength
	idx.avgDocLength = idx.totalDocLength / float32(idx.totalDocs)
	idx.totalPostings += len(vector.Indices)

	return nil
}

// SearchResult represents a search result with document ID and relevance score.
type SearchResult struct {
	DocID uint64
	Score float32
}

// Search performs BM25-based sparse vector search.
//
// BM25 Formula:
//   score(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))
//
// Where:
//   - IDF(qi) = log((N - df(qi) + 0.5) / (df(qi) + 0.5) + 1)
//   - f(qi, D) = frequency of query term qi in document D
//   - |D| = document length
//   - avgdl = average document length
//   - N = total documents
func (idx *InvertedIndex) Search(ctx context.Context, query *SparseVector, k int) ([]SearchResult, error) {
	if query.Dim != idx.dim {
		return nil, fmt.Errorf("query dimension mismatch: expected %d, got %d", idx.dim, query.Dim)
	}

	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if idx.totalDocs == 0 {
		return []SearchResult{}, nil
	}

	// Accumulate scores for each document
	scores := make(map[uint64]float32)

	// Process each query term
	for i, termID := range query.Indices {
		queryTermFreq := query.Values[i]

		// Get postings for this term
		postingsList, exists := idx.postings[termID]
		if !exists {
			continue // Term not in index
		}

		// Compute IDF: log((N - df + 0.5) / (df + 0.5) + 1)
		df := float64(idx.termDocCounts[termID])
		N := float64(idx.totalDocs)
		idf := float32(math.Log((N-df+0.5)/(df+0.5) + 1))

		// Score each document containing this term
		for _, posting := range postingsList {
			docID := posting.DocID
			termFreq := posting.Score // Term frequency in document

			// Get document length
			docLength := idx.docLengths[docID]

			// BM25 numerator: f(qi, D) * (k1 + 1)
			numerator := termFreq * (idx.k1 + 1)

			// BM25 denominator: f(qi, D) + k1 * (1 - b + b * |D| / avgdl)
			lengthNorm := 1 - idx.b + idx.b*(docLength/idx.avgDocLength)
			denominator := termFreq + idx.k1*lengthNorm

			// BM25 term score
			termScore := idf * (numerator / denominator)

			// Weight by query term frequency
			termScore *= queryTermFreq

			scores[docID] += termScore
		}
	}

	// Convert to results array
	results := make([]SearchResult, 0, len(scores))
	for docID, score := range scores {
		results = append(results, SearchResult{
			DocID: docID,
			Score: score,
		})
	}

	// Sort by score descending
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// Return top-k
	if k < len(results) {
		results = results[:k]
	}

	return results, nil
}

// SearchDotProduct performs dot product-based sparse search (alternative to BM25).
// Useful when vectors are already normalized or when BM25 is not appropriate.
func (idx *InvertedIndex) SearchDotProduct(ctx context.Context, query *SparseVector, k int) ([]SearchResult, error) {
	if query.Dim != idx.dim {
		return nil, fmt.Errorf("query dimension mismatch: expected %d, got %d", idx.dim, query.Dim)
	}

	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if idx.totalDocs == 0 {
		return []SearchResult{}, nil
	}

	// Accumulate scores
	scores := make(map[uint64]float32)

	// Process each query term
	for i, termID := range query.Indices {
		queryValue := query.Values[i]

		// Get postings for this term
		postingsList, exists := idx.postings[termID]
		if !exists {
			continue
		}

		// Accumulate dot product
		for _, posting := range postingsList {
			scores[posting.DocID] += posting.Score * queryValue
		}
	}

	// Convert to results
	results := make([]SearchResult, 0, len(scores))
	for docID, score := range scores {
		results = append(results, SearchResult{
			DocID: docID,
			Score: score,
		})
	}

	// Sort by score descending
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// Return top-k
	if k < len(results) {
		results = results[:k]
	}

	return results, nil
}

// SearchCosine performs cosine similarity-based sparse search.
func (idx *InvertedIndex) SearchCosine(ctx context.Context, query *SparseVector, k int) ([]SearchResult, error) {
	if query.Dim != idx.dim {
		return nil, fmt.Errorf("query dimension mismatch: expected %d, got %d", idx.dim, query.Dim)
	}

	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if idx.totalDocs == 0 {
		return []SearchResult{}, nil
	}

	queryNorm := query.Norm()
	if queryNorm == 0 {
		return []SearchResult{}, nil
	}

	// Accumulate dot products
	dotProducts := make(map[uint64]float32)

	for i, termID := range query.Indices {
		queryValue := query.Values[i]

		postingsList, exists := idx.postings[termID]
		if !exists {
			continue
		}

		for _, posting := range postingsList {
			dotProducts[posting.DocID] += posting.Score * queryValue
		}
	}

	// Compute cosine similarity: dot / (||q|| * ||d||)
	results := make([]SearchResult, 0, len(dotProducts))
	for docID, dot := range dotProducts {
		docNorm := idx.docNorms[docID]
		if docNorm > 0 {
			cosine := dot / (queryNorm * docNorm)
			results = append(results, SearchResult{
				DocID: docID,
				Score: cosine,
			})
		}
	}

	// Sort by score descending
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// Return top-k
	if k < len(results) {
		results = results[:k]
	}

	return results, nil
}

// Delete removes a document from the index.
func (idx *InvertedIndex) Delete(ctx context.Context, docID uint64) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	vector, exists := idx.docVectors[docID]
	if !exists {
		return fmt.Errorf("document %d not found", docID)
	}

	// Remove from postings lists
	for _, termID := range vector.Indices {
		postingsList := idx.postings[termID]

		// Find and remove posting
		for i, posting := range postingsList {
			if posting.DocID == docID {
				idx.postings[termID] = append(postingsList[:i], postingsList[i+1:]...)
				idx.totalPostings--
				break
			}
		}

		// Decrement term document count
		idx.termDocCounts[termID]--

		// Remove empty posting lists
		if len(idx.postings[termID]) == 0 {
			delete(idx.postings, termID)
			delete(idx.termDocCounts, termID)
		}
	}

	// Capture doc length before removing
	removedLength := idx.docLengths[docID]

	// Remove document data
	delete(idx.docVectors, docID)
	delete(idx.docNorms, docID)
	delete(idx.docLengths, docID)

	// Update statistics incrementally (O(1) instead of O(n))
	idx.totalDocs--
	idx.totalDocLength -= removedLength
	if idx.totalDocs > 0 {
		idx.avgDocLength = idx.totalDocLength / float32(idx.totalDocs)
	} else {
		idx.avgDocLength = 0
		idx.totalDocLength = 0 // Prevent floating-point drift
	}

	return nil
}

// Count returns the number of documents in the index.
func (idx *InvertedIndex) Count() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.totalDocs
}

// Stats returns index statistics.
type IndexStats struct {
	TotalDocs     int
	TotalTerms    int
	AvgDocLength  float32
	AvgPostings   float32
	MemoryUsage   int64
}

func (idx *InvertedIndex) Stats() IndexStats {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	avgPostings := float32(0)
	if len(idx.postings) > 0 {
		avgPostings = float32(idx.totalPostings) / float32(len(idx.postings))
	}

	// Estimate memory usage
	memoryUsage := int64(0)
	memoryUsage += int64(len(idx.postings)) * 8                    // map overhead
	memoryUsage += int64(idx.totalPostings) * 12                    // postings (8 bytes docID + 4 bytes score)
	memoryUsage += int64(len(idx.docVectors)) * (8 + 8 + 8)        // maps overhead
	memoryUsage += int64(len(idx.docVectors)) * 32                 // vector overhead estimate

	return IndexStats{
		TotalDocs:    idx.totalDocs,
		TotalTerms:   len(idx.postings),
		AvgDocLength: idx.avgDocLength,
		AvgPostings:  avgPostings,
		MemoryUsage:  memoryUsage,
	}
}

// persistedSparseIndex is the serialization format for InvertedIndex.
type persistedSparseIndex struct {
	Dim  int                          `json:"dim"`
	K1   float32                      `json:"k1"`
	B    float32                      `json:"b"`
	Docs map[string]*persistedSparseDoc `json:"docs"` // docID (string) -> sparse vector
}

type persistedSparseDoc struct {
	Indices []uint32  `json:"indices"`
	Values  []float32 `json:"values"`
}

// Export serializes the inverted index to JSON bytes.
// Only stores the source vectors — the posting lists and BM25 stats are
// rebuilt on Import via Add(), guaranteeing consistency.
func (idx *InvertedIndex) Export() ([]byte, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	state := persistedSparseIndex{
		Dim:  idx.dim,
		K1:   idx.k1,
		B:    idx.b,
		Docs: make(map[string]*persistedSparseDoc, len(idx.docVectors)),
	}

	for docID, vec := range idx.docVectors {
		state.Docs[fmt.Sprintf("%d", docID)] = &persistedSparseDoc{
			Indices: vec.Indices,
			Values:  vec.Values,
		}
	}

	return json.Marshal(state)
}

// Import restores the inverted index from JSON bytes produced by Export.
// Rebuilds all posting lists and BM25 statistics from the stored vectors.
func (idx *InvertedIndex) Import(data []byte) error {
	var state persistedSparseIndex
	if err := json.Unmarshal(data, &state); err != nil {
		return fmt.Errorf("unmarshal sparse index: %w", err)
	}

	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Reset state
	idx.dim = state.Dim
	idx.k1 = state.K1
	idx.b = state.B
	idx.postings = make(map[uint32][]Posting)
	idx.docNorms = make(map[uint64]float32)
	idx.docLengths = make(map[uint64]float32)
	idx.docVectors = make(map[uint64]*SparseVector)
	idx.termDocCounts = make(map[uint32]int)
	idx.totalDocs = 0
	idx.totalDocLength = 0
	idx.totalPostings = 0
	idx.avgDocLength = 0

	// Re-add each vector to rebuild posting lists and stats
	for idStr, doc := range state.Docs {
		var docID uint64
		fmt.Sscanf(idStr, "%d", &docID)

		vec, err := NewSparseVector(doc.Indices, doc.Values, state.Dim)
		if err != nil {
			return fmt.Errorf("restore vector %d: %w", docID, err)
		}

		// Inline the Add logic (we already hold the lock)
		idx.docVectors[docID] = vec.Clone()
		idx.docNorms[docID] = vec.Norm()

		docLength := float32(0)
		for i, termID := range vec.Indices {
			score := vec.Values[i]
			docLength += score

			idx.postings[termID] = append(idx.postings[termID], Posting{
				DocID: docID,
				Score: score,
			})
			idx.termDocCounts[termID]++
		}
		idx.docLengths[docID] = docLength
		idx.totalDocLength += docLength
		idx.totalPostings += len(vec.Indices)
		idx.totalDocs++
	}

	// Compute average doc length from accumulated total (O(1))
	if idx.totalDocs > 0 {
		idx.avgDocLength = idx.totalDocLength / float32(idx.totalDocs)
	}

	return nil
}
