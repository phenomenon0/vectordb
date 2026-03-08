package collection

import (
	"context"
	"fmt"
	"strings"
	"unicode"

	"github.com/phenomenon0/vectordb/internal/sparse"
)

// MigrationTool helps migrate from v1 (single-index) to v2 (multi-vector) collections
type MigrationTool struct {
	manager *CollectionManager
}

// NewMigrationTool creates a new migration tool
func NewMigrationTool(manager *CollectionManager) *MigrationTool {
	return &MigrationTool{manager: manager}
}

// V1Document represents a document from v1 (single vector)
type V1Document struct {
	ID       string
	Vector   []float32
	Text     string
	Metadata map[string]interface{}
}

// MigrationConfig configures how to migrate a collection
type MigrationConfig struct {
	// Source v1 collection (if migrating from existing data)
	SourceCollection string

	// Target v2 collection name
	TargetCollection string

	// Dense vector field configuration
	DenseFieldName string
	DenseDimension int
	DenseIndexType IndexType
	DenseIndexParams map[string]interface{}

	// Sparse vector configuration
	EnableSparse bool
	SparseFieldName string
	SparseMethod string // "bm25" or "none"
	SparseDimension int
	SparseIndexParams map[string]interface{}

	// Generation settings
	GenerateSparseFromText bool // Generate sparse vectors from text field
	TextFieldName string // Field in metadata containing text
}

// DefaultMigrationConfig returns a sensible default configuration
func DefaultMigrationConfig(targetCollection string) MigrationConfig {
	return MigrationConfig{
		TargetCollection: targetCollection,
		DenseFieldName: "embedding",
		DenseDimension: 384,
		DenseIndexType: IndexTypeHNSW,
		DenseIndexParams: map[string]interface{}{
			"m": 16,
			"ef_construction": 200,
		},
		EnableSparse: true,
		SparseFieldName: "keywords",
		SparseMethod: "bm25",
		SparseDimension: 10000,
		SparseIndexParams: map[string]interface{}{
			"k1": 1.2,
			"b": 0.75,
		},
		GenerateSparseFromText: true,
		TextFieldName: "text",
	}
}

// CreateV2Collection creates a v2 collection based on migration config
func (mt *MigrationTool) CreateV2Collection(ctx context.Context, config MigrationConfig) error {
	// Build collection schema
	fields := []VectorField{
		{
			Name: config.DenseFieldName,
			Type: VectorTypeDense,
			Dim: config.DenseDimension,
			Index: IndexConfig{
				Type: config.DenseIndexType,
				Params: config.DenseIndexParams,
			},
		},
	}

	// Add sparse field if enabled
	if config.EnableSparse {
		fields = append(fields, VectorField{
			Name: config.SparseFieldName,
			Type: VectorTypeSparse,
			Dim: config.SparseDimension,
			Index: IndexConfig{
				Type: IndexTypeInverted,
				Params: config.SparseIndexParams,
			},
		})
	}

	schema := CollectionSchema{
		Name: config.TargetCollection,
		Fields: fields,
		Description: fmt.Sprintf("Migrated from v1 with hybrid search support"),
	}

	// Create collection
	_, err := mt.manager.CreateCollection(ctx, schema)
	return err
}

// MigrateDocuments migrates v1 documents to v2 format
func (mt *MigrationTool) MigrateDocuments(ctx context.Context, config MigrationConfig, v1Docs []V1Document) error {
	// Ensure target collection exists
	if !mt.manager.HasCollection(config.TargetCollection) {
		if err := mt.CreateV2Collection(ctx, config); err != nil {
			return fmt.Errorf("failed to create target collection: %w", err)
		}
	}

	// Convert and insert documents
	for _, v1Doc := range v1Docs {
		v2Doc, err := mt.convertDocument(config, v1Doc)
		if err != nil {
			return fmt.Errorf("failed to convert document %s: %w", v1Doc.ID, err)
		}

		if err := mt.manager.AddDocument(ctx, config.TargetCollection, &v2Doc); err != nil {
			return fmt.Errorf("failed to add document %s: %w", v1Doc.ID, err)
		}
	}

	return nil
}

// convertDocument converts a v1 document to v2 format with sparse vectors
func (mt *MigrationTool) convertDocument(config MigrationConfig, v1Doc V1Document) (Document, error) {
	// Create v2 document with dense vector
	vectors := make(map[string]interface{})
	vectors[config.DenseFieldName] = v1Doc.Vector

	// Generate sparse vector if enabled
	if config.EnableSparse && config.GenerateSparseFromText {
		text := v1Doc.Text
		if text == "" {
			// Try to get text from metadata
			if textVal, ok := v1Doc.Metadata[config.TextFieldName]; ok {
				if textStr, ok := textVal.(string); ok {
					text = textStr
				}
			}
		}

		if text != "" {
			sparseVec, err := mt.generateSparseVector(config, text)
			if err != nil {
				return Document{}, fmt.Errorf("failed to generate sparse vector: %w", err)
			}
			vectors[config.SparseFieldName] = sparseVec
		}
	}

	return Document{
		Vectors: vectors,
		Metadata: v1Doc.Metadata,
	}, nil
}

// generateSparseVector generates a sparse vector from text using BM25 tokenization
func (mt *MigrationTool) generateSparseVector(config MigrationConfig, text string) (*sparse.SparseVector, error) {
	// Tokenize text
	tokens := tokenize(text)

	// Count term frequencies
	termFreq := make(map[string]int)
	for _, token := range tokens {
		termFreq[token]++
	}

	// Convert to sparse vector format
	// Map tokens to indices (simple hash for now)
	indices := make([]uint32, 0, len(termFreq))
	values := make([]float32, 0, len(termFreq))

	for term, freq := range termFreq {
		// Simple hash to index mapping
		idx := hashTerm(term) % uint32(config.SparseDimension)
		indices = append(indices, idx)

		// TF score (can be improved with IDF if corpus stats available)
		tf := float32(freq)
		values = append(values, tf)
	}

	// Create sparse vector
	return sparse.NewSparseVector(indices, values, config.SparseDimension)
}

// tokenize splits text into tokens (words)
func tokenize(text string) []string {
	tokens := make([]string, 0, len(text)/4+1)
	var current strings.Builder

	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsNumber(r) {
			current.WriteRune(unicode.ToLower(r))
		} else {
			if current.Len() > 0 {
				token := current.String()
				if len(token) > 2 && !isStopword(token) {
					tokens = append(tokens, token)
				}
				current.Reset()
			}
		}
	}

	// Add last token
	if current.Len() > 0 {
		token := current.String()
		if len(token) > 2 && !isStopword(token) {
			tokens = append(tokens, token)
		}
	}

	return tokens
}

// hashTerm creates a hash for a term
func hashTerm(term string) uint32 {
	hash := uint32(5381)
	for _, c := range term {
		hash = ((hash << 5) + hash) + uint32(c)
	}
	return hash
}

// isStopword checks if a word is a common stopword
func isStopword(word string) bool {
	stopwords := map[string]bool{
		"the": true, "a": true, "an": true, "and": true, "or": true,
		"but": true, "in": true, "on": true, "at": true, "to": true,
		"for": true, "of": true, "as": true, "by": true, "is": true,
		"was": true, "are": true, "were": true, "be": true, "been": true,
		"have": true, "has": true, "had": true, "do": true, "does": true,
		"did": true, "will": true, "would": true, "could": true, "should": true,
	}
	return stopwords[word]
}

// MigrationStats tracks migration progress
type MigrationStats struct {
	TotalDocuments int
	MigratedDocuments int
	FailedDocuments int
	Errors []string
}

// MigrateBatch migrates documents in batches for better performance
func (mt *MigrationTool) MigrateBatch(ctx context.Context, config MigrationConfig, v1Docs []V1Document, batchSize int) (*MigrationStats, error) {
	stats := &MigrationStats{
		TotalDocuments: len(v1Docs),
		Errors: make([]string, 0),
	}

	// Ensure target collection exists
	if !mt.manager.HasCollection(config.TargetCollection) {
		if err := mt.CreateV2Collection(ctx, config); err != nil {
			return stats, fmt.Errorf("failed to create target collection: %w", err)
		}
	}

	// Process in batches
	for i := 0; i < len(v1Docs); i += batchSize {
		end := i + batchSize
		if end > len(v1Docs) {
			end = len(v1Docs)
		}

		batch := v1Docs[i:end]
		v2Docs := make([]Document, 0, len(batch))

		// Convert batch
		for _, v1Doc := range batch {
			v2Doc, err := mt.convertDocument(config, v1Doc)
			if err != nil {
				stats.FailedDocuments++
				stats.Errors = append(stats.Errors, fmt.Sprintf("doc %s: %v", v1Doc.ID, err))
				continue
			}
			v2Docs = append(v2Docs, v2Doc)
		}

		// Batch insert
		if err := mt.manager.BatchAddDocuments(ctx, config.TargetCollection, v2Docs); err != nil {
			stats.FailedDocuments += len(v2Docs)
			stats.Errors = append(stats.Errors, fmt.Sprintf("batch %d-%d: %v", i, end, err))
			continue
		}

		stats.MigratedDocuments += len(v2Docs)
	}

	return stats, nil
}

// ValidateMigration checks if migration was successful
func (mt *MigrationTool) ValidateMigration(ctx context.Context, config MigrationConfig, expectedCount int) error {
	info, err := mt.manager.GetCollectionInfo(config.TargetCollection)
	if err != nil {
		return fmt.Errorf("target collection not found: %w", err)
	}

	if info.DocCount != expectedCount {
		return fmt.Errorf("document count mismatch: expected %d, got %d", expectedCount, info.DocCount)
	}

	// Verify fields
	expectedFields := 1 // Dense field
	if config.EnableSparse {
		expectedFields++
	}

	if len(info.Fields) != expectedFields {
		return fmt.Errorf("field count mismatch: expected %d, got %d", expectedFields, len(info.Fields))
	}

	return nil
}

// ExampleMigration demonstrates how to use the migration tool
func ExampleMigration() {
	// This is example code showing migration workflow

	// 1. Prepare v1 documents (normally loaded from existing v1 collection)
	v1Docs := []V1Document{
		{
			ID: "doc1",
			Vector: []float32{0.1, 0.2, 0.3}, // ... 384 dims
			Text: "Machine learning is transforming artificial intelligence research",
			Metadata: map[string]interface{}{
				"category": "AI",
				"year": "2025",
			},
		},
		{
			ID: "doc2",
			Vector: []float32{0.2, 0.3, 0.4}, // ... 384 dims
			Text: "Natural language processing enables better text understanding",
			Metadata: map[string]interface{}{
				"category": "NLP",
				"year": "2025",
			},
		},
	}

	// 2. Create migration tool
	manager := NewCollectionManager("./data")
	migrator := NewMigrationTool(manager)

	// 3. Configure migration
	config := DefaultMigrationConfig("articles_v2")
	config.DenseDimension = 384
	config.SparseDimension = 10000

	// 4. Run migration
	ctx := context.Background()
	stats, err := migrator.MigrateBatch(ctx, config, v1Docs, 100)
	if err != nil {
		fmt.Printf("Migration failed: %v\n", err)
		return
	}

	// 5. Validate
	if err := migrator.ValidateMigration(ctx, config, len(v1Docs)); err != nil {
		fmt.Printf("Validation failed: %v\n", err)
		return
	}

	fmt.Printf("Migration successful: %d/%d documents migrated\n",
		stats.MigratedDocuments, stats.TotalDocuments)
}
