package testdata

import (
	"fmt"
	"math/rand"

	"github.com/phenomenon0/vectordb/internal/sparse"
)

// Metadata generators for filtered search benchmarks.

// MetadataRecord holds metadata for a single vector.
type MetadataRecord struct {
	Category string
	Score    float64
	Tags     []string
	Region   string
	Active   bool
}

// GenerateUniformMetadata creates metadata with uniform distributions.
// Score is uniform [0, 100), category is one of 10 types, region is one of 5.
func GenerateUniformMetadata(count int, rng *rand.Rand) []map[string]interface{} {
	if rng == nil {
		rng = rand.New(rand.NewSource(42))
	}

	categories := []string{"electronics", "clothing", "books", "food", "sports",
		"health", "automotive", "home", "toys", "tools"}
	regions := []string{"us-east", "us-west", "eu-west", "ap-south", "ap-east"}
	tagPool := []string{"featured", "sale", "new", "premium", "clearance",
		"bestseller", "eco", "limited", "bundle", "exclusive"}

	records := make([]map[string]interface{}, count)
	for i := 0; i < count; i++ {
		numTags := 1 + rng.Intn(4)
		tags := make([]string, numTags)
		for t := 0; t < numTags; t++ {
			tags[t] = tagPool[rng.Intn(len(tagPool))]
		}

		records[i] = map[string]interface{}{
			"category": categories[rng.Intn(len(categories))],
			"score":    rng.Float64() * 100,
			"tags":     tags,
			"region":   regions[rng.Intn(len(regions))],
			"active":   rng.Float64() > 0.1, // 90% active
			"price":    10.0 + rng.Float64()*990.0,
			"year":     2020 + rng.Intn(6),
		}
	}
	return records
}

// GenerateSkewedMetadata creates metadata with power-law distributions.
// A few categories dominate; useful for testing filter selectivity edge cases.
func GenerateSkewedMetadata(count int, rng *rand.Rand) []map[string]interface{} {
	if rng == nil {
		rng = rand.New(rand.NewSource(42))
	}

	categories := []string{"dominant", "common", "uncommon", "rare", "very_rare"}
	weights := []float64{0.50, 0.25, 0.15, 0.08, 0.02} // Zipf-ish

	records := make([]map[string]interface{}, count)
	for i := 0; i < count; i++ {
		// Weighted category selection
		r := rng.Float64()
		cumulative := 0.0
		cat := categories[0]
		for j, w := range weights {
			cumulative += w
			if r < cumulative {
				cat = categories[j]
				break
			}
		}

		records[i] = map[string]interface{}{
			"category": cat,
			"score":    rng.Float64() * 100,
			"tenant":   fmt.Sprintf("tenant_%d", rng.Intn(10)),
			"priority": rng.Intn(5),
		}
	}
	return records
}

// GenerateSparseDocuments creates sparse vectors simulating tokenized documents.
// Each document has between minTokens and maxTokens non-zero entries from a
// Zipfian term frequency distribution.
func GenerateSparseDocuments(count, vocabSize, minTokens, maxTokens int, rng *rand.Rand) []*sparse.SparseVector {
	if rng == nil {
		rng = rand.New(rand.NewSource(42))
	}

	docs := make([]*sparse.SparseVector, count)
	for i := 0; i < count; i++ {
		numTokens := minTokens + rng.Intn(maxTokens-minTokens+1)

		// Use Zipf distribution for term selection (realistic)
		termFreq := make(map[uint32]float32)
		for t := 0; t < numTokens; t++ {
			// Zipf: term_id ~ 1/(rank^s), approximated by biasing toward lower IDs
			termID := uint32(rng.ExpFloat64() * float64(vocabSize) / 10.0)
			if termID >= uint32(vocabSize) {
				termID = uint32(rng.Intn(vocabSize))
			}
			termFreq[termID] += 1.0
		}

		indices := make([]uint32, 0, len(termFreq))
		values := make([]float32, 0, len(termFreq))
		for idx, freq := range termFreq {
			indices = append(indices, idx)
			values = append(values, freq)
		}

		// Sort indices for SparseVector requirement
		sortSparseByIndex(indices, values)

		sv, err := sparse.NewSparseVector(indices, values, vocabSize)
		if err != nil {
			// Fallback: create simple vector
			sv = &sparse.SparseVector{
				Indices: indices,
				Values:  values,
				Dim:     vocabSize,
			}
		}
		docs[i] = sv
	}
	return docs
}

// GenerateSparseQueries creates sparse query vectors (shorter than documents).
func GenerateSparseQueries(count, vocabSize, minTerms, maxTerms int, rng *rand.Rand) []*sparse.SparseVector {
	return GenerateSparseDocuments(count, vocabSize, minTerms, maxTerms, rng)
}

// sortSparseByIndex sorts indices and values together by index (ascending).
func sortSparseByIndex(indices []uint32, values []float32) {
	n := len(indices)
	for i := 1; i < n; i++ {
		for j := i; j > 0 && indices[j-1] > indices[j]; j-- {
			indices[j], indices[j-1] = indices[j-1], indices[j]
			values[j], values[j-1] = values[j-1], values[j]
		}
	}
}
