package extraction

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
	"sync"
	"time"
)

// Pipeline orchestrates the extraction process from documents to knowledge graphs.
type Pipeline struct {
	extractor Extractor
	chunker   Chunker
	cfg       PipelineConfig

	// Statistics
	mu    sync.RWMutex
	stats PipelineStats
}

// Chunker defines the interface for text chunking.
type Chunker interface {
	Chunk(text string) []Chunk
}

// Chunk represents a text chunk with metadata.
type Chunk struct {
	ID     string
	Text   string
	Offset int // Character offset in original document
	Index  int // Chunk index
}

// PipelineConfig configures the extraction pipeline.
type PipelineConfig struct {
	// ChunkSize is the target size for text chunks
	ChunkSize int `json:"chunk_size"`

	// ChunkOverlap is the overlap between chunks
	ChunkOverlap int `json:"chunk_overlap"`

	// EnableTemporal enables temporal event extraction
	EnableTemporal bool `json:"enable_temporal"`

	// MergeResults merges all chunk results into a single graph
	MergeResults bool `json:"merge_results"`

	// DeduplicateNodes removes duplicate nodes by ID
	DeduplicateNodes bool `json:"deduplicate_nodes"`

	// Concurrency for parallel extraction
	Concurrency int `json:"concurrency"`

	// SkipEmptyChunks skips chunks with insufficient content
	SkipEmptyChunks bool `json:"skip_empty_chunks"`

	// MinChunkLength is the minimum chunk length to process
	MinChunkLength int `json:"min_chunk_length"`
}

// DefaultPipelineConfig returns sensible defaults.
func DefaultPipelineConfig() PipelineConfig {
	return PipelineConfig{
		ChunkSize:        2000,
		ChunkOverlap:     200,
		EnableTemporal:   false,
		MergeResults:     true,
		DeduplicateNodes: true,
		Concurrency:      4,
		SkipEmptyChunks:  true,
		MinChunkLength:   50,
	}
}

// PipelineStats tracks pipeline execution statistics.
type PipelineStats struct {
	TotalDocuments   int64
	TotalChunks      int64
	ProcessedChunks  int64
	FailedChunks     int64
	TotalNodes       int64
	TotalEdges       int64
	TotalEvents      int64
	TotalDuration    time.Duration
	AverageChunkTime time.Duration
}

// NewPipeline creates a new extraction pipeline.
func NewPipeline(extractor Extractor, cfg PipelineConfig) *Pipeline {
	return &Pipeline{
		extractor: extractor,
		chunker:   &SimpleChunker{Size: cfg.ChunkSize, Overlap: cfg.ChunkOverlap},
		cfg:       cfg,
	}
}

// SetChunker sets a custom chunker.
func (p *Pipeline) SetChunker(c Chunker) {
	p.chunker = c
}

// Process extracts a knowledge graph from a document.
func (p *Pipeline) Process(ctx context.Context, docID, content string) (*ExtractionResult, error) {
	start := time.Now()

	// Update stats
	p.mu.Lock()
	p.stats.TotalDocuments++
	p.mu.Unlock()

	// Chunk the content
	chunks := p.chunker.Chunk(content)

	// Filter empty chunks
	if p.cfg.SkipEmptyChunks {
		filtered := make([]Chunk, 0, len(chunks))
		for _, c := range chunks {
			if len(strings.TrimSpace(c.Text)) >= p.cfg.MinChunkLength {
				filtered = append(filtered, c)
			}
		}
		chunks = filtered
	}

	p.mu.Lock()
	p.stats.TotalChunks += int64(len(chunks))
	p.mu.Unlock()

	result := &ExtractionResult{
		DocumentID: docID,
		Chunks:     make([]ChunkResult, len(chunks)),
		Stats: ExtractionStats{
			TotalChunks: len(chunks),
		},
	}

	if len(chunks) == 0 {
		result.Stats.Duration = time.Since(start)
		return result, nil
	}

	// Extract from chunks (concurrently or sequentially based on config)
	chunkTexts := make([]string, len(chunks))
	for i, c := range chunks {
		chunkTexts[i] = c.Text
	}

	var graphs []*KnowledgeGraph
	var temporalGraphs []*TemporalKnowledgeGraph
	var extractErr error

	if p.cfg.EnableTemporal {
		temporalGraphs, extractErr = p.extractTemporalBatch(ctx, chunkTexts)
		// Convert to regular graphs for merging
		graphs = make([]*KnowledgeGraph, len(temporalGraphs))
		for i, tg := range temporalGraphs {
			if tg != nil {
				graphs[i] = &tg.KnowledgeGraph
			}
		}
	} else {
		graphs, extractErr = p.extractor.ExtractBatch(ctx, chunkTexts)
	}

	// Build chunk results
	for i, chunk := range chunks {
		cr := ChunkResult{
			ChunkID:   chunk.ID,
			ChunkText: chunk.Text,
		}

		if i < len(graphs) && graphs[i] != nil {
			cr.Graph = *graphs[i]
			result.Stats.ProcessedChunks++
			result.Stats.TotalNodes += len(graphs[i].Nodes)
			result.Stats.TotalEdges += len(graphs[i].Edges)
		} else {
			result.Stats.FailedChunks++
			if extractErr != nil {
				cr.Error = extractErr.Error()
			}
		}

		if p.cfg.EnableTemporal && i < len(temporalGraphs) && temporalGraphs[i] != nil {
			cr.Temporal = temporalGraphs[i]
			result.Stats.TotalEvents += len(temporalGraphs[i].Events)
		}

		result.Chunks[i] = cr
	}

	// Merge results
	if p.cfg.MergeResults {
		result.Merged = p.mergeGraphs(graphs)
		result.Stats.UniqueNodes = len(result.Merged.Nodes)
		result.Stats.UniqueEdges = len(result.Merged.Edges)
	}

	result.Stats.Duration = time.Since(start)

	// Update global stats
	p.mu.Lock()
	p.stats.ProcessedChunks += int64(result.Stats.ProcessedChunks)
	p.stats.FailedChunks += int64(result.Stats.FailedChunks)
	p.stats.TotalNodes += int64(result.Stats.TotalNodes)
	p.stats.TotalEdges += int64(result.Stats.TotalEdges)
	p.stats.TotalEvents += int64(result.Stats.TotalEvents)
	p.stats.TotalDuration += result.Stats.Duration
	if p.stats.ProcessedChunks > 0 {
		p.stats.AverageChunkTime = p.stats.TotalDuration / time.Duration(p.stats.ProcessedChunks)
	}
	p.mu.Unlock()

	return result, extractErr
}

// extractTemporalBatch extracts temporal graphs from multiple chunks.
func (p *Pipeline) extractTemporalBatch(ctx context.Context, chunks []string) ([]*TemporalKnowledgeGraph, error) {
	results := make([]*TemporalKnowledgeGraph, len(chunks))
	errors := make([]error, len(chunks))

	sem := make(chan struct{}, p.cfg.Concurrency)
	var wg sync.WaitGroup

	for i, chunk := range chunks {
		wg.Add(1)
		go func(idx int, text string) {
			defer wg.Done()

			select {
			case sem <- struct{}{}:
				defer func() { <-sem }()
			case <-ctx.Done():
				errors[idx] = ctx.Err()
				return
			}

			tkg, err := p.extractor.ExtractTemporal(ctx, text)
			if err != nil {
				errors[idx] = err
				return
			}
			results[idx] = tkg
		}(i, chunk)
	}

	wg.Wait()

	for _, err := range errors {
		if err != nil {
			return results, err
		}
	}

	return results, nil
}

// mergeGraphs merges multiple graphs into one, deduplicating nodes.
func (p *Pipeline) mergeGraphs(graphs []*KnowledgeGraph) KnowledgeGraph {
	merged := KnowledgeGraph{
		Nodes: make([]Node, 0),
		Edges: make([]Edge, 0),
	}

	for _, g := range graphs {
		if g != nil {
			merged.Merge(g)
		}
	}

	return merged
}

// ProcessBatch processes multiple documents.
func (p *Pipeline) ProcessBatch(ctx context.Context, docs map[string]string) ([]*ExtractionResult, error) {
	results := make([]*ExtractionResult, 0, len(docs))

	for docID, content := range docs {
		result, err := p.Process(ctx, docID, content)
		if err != nil {
			// Continue processing other docs, but record the error
			if result == nil {
				result = &ExtractionResult{DocumentID: docID}
			}
		}
		results = append(results, result)
	}

	return results, nil
}

// Stats returns the current pipeline statistics.
func (p *Pipeline) Stats() PipelineStats {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.stats
}

// ResetStats resets the pipeline statistics.
func (p *Pipeline) ResetStats() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.stats = PipelineStats{}
}

// SimpleChunker implements basic text chunking.
type SimpleChunker struct {
	Size    int
	Overlap int
}

// Chunk splits text into overlapping chunks.
func (c *SimpleChunker) Chunk(text string) []Chunk {
	if c.Size <= 0 {
		c.Size = 2000
	}
	if c.Overlap < 0 {
		c.Overlap = 0
	}
	if c.Overlap >= c.Size {
		c.Overlap = c.Size / 4
	}

	var chunks []Chunk

	// Try to split on sentence boundaries
	sentences := splitSentences(text)

	var currentChunk strings.Builder
	chunkIndex := 0
	startOffset := 0
	currentOffset := 0

	for _, sentence := range sentences {
		// If adding this sentence exceeds chunk size, save current chunk
		if currentChunk.Len() > 0 && currentChunk.Len()+len(sentence) > c.Size {
			chunk := Chunk{
				ID:     generateChunkID(text, chunkIndex),
				Text:   currentChunk.String(),
				Offset: startOffset,
				Index:  chunkIndex,
			}
			chunks = append(chunks, chunk)
			chunkIndex++

			// Calculate overlap start position
			overlapText := currentChunk.String()
			if len(overlapText) > c.Overlap {
				overlapText = overlapText[len(overlapText)-c.Overlap:]
			}

			currentChunk.Reset()
			currentChunk.WriteString(overlapText)
			startOffset = currentOffset - len(overlapText)
		}

		currentChunk.WriteString(sentence)
		currentOffset += len(sentence)
	}

	// Don't forget the last chunk
	if currentChunk.Len() > 0 {
		chunk := Chunk{
			ID:     generateChunkID(text, chunkIndex),
			Text:   currentChunk.String(),
			Offset: startOffset,
			Index:  chunkIndex,
		}
		chunks = append(chunks, chunk)
	}

	return chunks
}

// splitSentences splits text into sentences (simple approach).
func splitSentences(text string) []string {
	var sentences []string
	var current strings.Builder

	for i := 0; i < len(text); i++ {
		current.WriteByte(text[i])

		// Check for sentence end
		if text[i] == '.' || text[i] == '!' || text[i] == '?' {
			// Look ahead for space or end
			if i+1 >= len(text) || text[i+1] == ' ' || text[i+1] == '\n' {
				sentences = append(sentences, current.String())
				current.Reset()
				// Skip the space after sentence end
				if i+1 < len(text) && text[i+1] == ' ' {
					i++
				}
			}
		} else if text[i] == '\n' && current.Len() > 0 {
			// Also split on newlines for code/markdown
			if i+1 < len(text) && text[i+1] == '\n' {
				sentences = append(sentences, current.String())
				current.Reset()
			}
		}
	}

	// Add remaining text
	if current.Len() > 0 {
		sentences = append(sentences, current.String())
	}

	return sentences
}

// generateChunkID creates a deterministic ID for a chunk.
func generateChunkID(text string, index int) string {
	hash := sha256.Sum256([]byte(fmt.Sprintf("%s:%d", text, index)))
	return hex.EncodeToString(hash[:8])
}

// Cognify is a convenience function that creates a pipeline and processes a document.
// This mimics Cognee's cognify() function.
func Cognify(ctx context.Context, content string, opts ...CognifyOption) (*ExtractionResult, error) {
	cfg := CognifyConfig{
		ExtractorConfig: DefaultConfig(),
		PipelineConfig:  DefaultPipelineConfig(),
	}

	for _, opt := range opts {
		opt(&cfg)
	}

	extractor, err := NewExtractor(cfg.ExtractorConfig)
	if err != nil {
		return nil, fmt.Errorf("create extractor: %w", err)
	}

	pipeline := NewPipeline(extractor, cfg.PipelineConfig)

	docID := cfg.DocumentID
	if docID == "" {
		hash := sha256.Sum256([]byte(content))
		docID = hex.EncodeToString(hash[:16])
	}

	return pipeline.Process(ctx, docID, content)
}

// CognifyConfig holds configuration for the Cognify convenience function.
type CognifyConfig struct {
	ExtractorConfig ExtractorConfig
	PipelineConfig  PipelineConfig
	DocumentID      string
}

// CognifyOption is a functional option for Cognify.
type CognifyOption func(*CognifyConfig)

// WithOllama configures Ollama as the extractor.
func WithOllama(baseURL, model string) CognifyOption {
	return func(c *CognifyConfig) {
		c.ExtractorConfig.Provider = "ollama"
		c.ExtractorConfig.BaseURL = baseURL
		c.ExtractorConfig.Model = model
	}
}

// WithOpenAI configures OpenAI as the extractor.
func WithOpenAI(apiKey, model string) CognifyOption {
	return func(c *CognifyConfig) {
		c.ExtractorConfig.Provider = "openai"
		c.ExtractorConfig.APIKey = apiKey
		c.ExtractorConfig.Model = model
	}
}

// WithTemporal enables temporal event extraction.
func WithTemporal() CognifyOption {
	return func(c *CognifyConfig) {
		c.PipelineConfig.EnableTemporal = true
	}
}

// WithChunkSize sets the chunk size.
func WithChunkSize(size int) CognifyOption {
	return func(c *CognifyConfig) {
		c.PipelineConfig.ChunkSize = size
	}
}

// WithDocumentID sets the document ID.
func WithDocumentID(id string) CognifyOption {
	return func(c *CognifyConfig) {
		c.DocumentID = id
	}
}

// WithCustomPrompt sets a custom extraction prompt.
func WithCustomPrompt(prompt string) CognifyOption {
	return func(c *CognifyConfig) {
		c.ExtractorConfig.CustomPrompt = prompt
	}
}

// WithConcurrency sets the extraction concurrency.
func WithConcurrency(n int) CognifyOption {
	return func(c *CognifyConfig) {
		c.PipelineConfig.Concurrency = n
		c.ExtractorConfig.Concurrency = n
	}
}
