package main

import (
	"fmt"
	"net/http"
	"os"
	"strconv"
	"time"
)

// ======================================================================================
// Mode-Aware Embedder Factory
// ======================================================================================
// Creates embedders based on the current VectorDB mode (LOCAL or PRO)
// - LOCAL: Prioritizes ONNX (offline, free), falls back to Ollama, then hash
// - PRO: Uses OpenAI only (requires API key)
// ======================================================================================

// EmbedderFactory creates embedders based on mode configuration
type EmbedderFactory struct {
	mode        *ModeConfig
	costTracker *CostTracker
}

// NewEmbedderFactory creates a new factory for the given mode
func NewEmbedderFactory(mode *ModeConfig, costTracker *CostTracker) *EmbedderFactory {
	return &EmbedderFactory{
		mode:        mode,
		costTracker: costTracker,
	}
}

// CreateEmbedder creates an embedder appropriate for the current mode
func (f *EmbedderFactory) CreateEmbedder() (Embedder, error) {
	switch f.mode.Mode {
	case ModeLocal:
		return f.createLocalEmbedder()
	case ModePro:
		return f.createProEmbedder()
	default:
		return nil, fmt.Errorf("unknown mode: %s", f.mode.Mode)
	}
}

// createLocalEmbedder creates an embedder for LOCAL mode
// Priority: ONNX > Ollama > Hash (fallback)
func (f *EmbedderFactory) createLocalEmbedder() (Embedder, error) {
	// Priority 1: ONNX embeddings (local, good quality, requires model files)
	onnxEmb, err := f.tryOnnxEmbedder(384) // BGE-small is always 384d
	if err == nil && onnxEmb != nil {
		fmt.Println(">>> [LOCAL] Using ONNX embedder (BGE-small, 384d)")
		// Update mode dimension to match ONNX
		f.mode.Dimension = 384
		f.mode.EmbedderType = "onnx"
		f.mode.EmbedderModel = "bge-small-en-v1.5"
		return onnxEmb, nil
	}

	// Priority 2: Ollama embeddings (local, good quality)
	ollamaEmb := f.tryOllamaEmbedder()
	if ollamaEmb != nil {
		fmt.Println(">>> [LOCAL] Using Ollama embedder (nomic-embed-text, 768d)")
		// Update mode dimension to match Ollama
		f.mode.Dimension = 768
		f.mode.EmbedderType = "ollama"
		f.mode.EmbedderModel = "nomic-embed-text"
		return ollamaEmb, nil
	}

	// Priority 3: Hash embedder (fallback, low quality)
	fmt.Println(">>> [LOCAL] Using hash embedder (install Ollama for better quality)")
	fmt.Println("           Run: ollama pull nomic-embed-text")
	// Hash embedder can use any dimension, default to 384 for compatibility
	f.mode.Dimension = 384
	f.mode.EmbedderType = "hash"
	f.mode.EmbedderModel = "hash-384"
	return NewHashEmbedder(384), nil
}

// createProEmbedder creates an embedder for PRO mode
// Only uses OpenAI (requires API key)
func (f *EmbedderFactory) createProEmbedder() (Embedder, error) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("PRO mode requires OPENAI_API_KEY environment variable")
	}

	fmt.Println(">>> [PRO] Using OpenAI embedder (text-embedding-3-small, 1536d)")
	
	// Wrap OpenAI embedder with cost tracking
	baseEmb := NewOpenAIEmbedder(apiKey)
	if f.costTracker != nil {
		return NewTrackedEmbedder(baseEmb, f.costTracker), nil
	}
	return baseEmb, nil
}

// tryOnnxEmbedder attempts to create an ONNX embedder
func (f *EmbedderFactory) tryOnnxEmbedder(dim int) (Embedder, error) {
	// Check for model files
	defaultModel := "vectordb/models/bge-small-en-v1.5/model.onnx"
	defaultTok := "vectordb/models/bge-small-en-v1.5/tokenizer.json"

	// Also check in the data directory
	dataDir := GetDataDirectory(f.mode.Mode)
	altModel := dataDir + "/models/bge-small-en-v1.5/model.onnx"
	altTok := dataDir + "/models/bge-small-en-v1.5/tokenizer.json"

	modelPath := os.Getenv("ONNX_EMBED_MODEL")
	tokPath := os.Getenv("ONNX_EMBED_TOKENIZER")

	// Try default paths if not specified
	if modelPath == "" {
		for _, p := range []string{defaultModel, altModel, "./models/bge-small-en-v1.5/model.onnx"} {
			if _, err := os.Stat(p); err == nil {
				modelPath = p
				break
			}
		}
	}
	if tokPath == "" {
		for _, p := range []string{defaultTok, altTok, "./models/bge-small-en-v1.5/tokenizer.json"} {
			if _, err := os.Stat(p); err == nil {
				tokPath = p
				break
			}
		}
	}

	if modelPath == "" || tokPath == "" {
		return nil, fmt.Errorf("ONNX model files not found")
	}

	maxLen := 512
	if env := os.Getenv("ONNX_EMBED_MAX_LEN"); env != "" {
		if v, err := strconv.Atoi(env); err == nil && v >= 0 {
			maxLen = v
		}
	}

	emb, err := NewOnnxEmbedder(modelPath, tokPath, dim, maxLen)
	if err != nil {
		return nil, fmt.Errorf("ONNX init failed: %w", err)
	}
	return emb, nil
}

// tryOllamaEmbedder attempts to connect to Ollama
func (f *EmbedderFactory) tryOllamaEmbedder() Embedder {
	ollamaURL := os.Getenv("OLLAMA_URL")
	if ollamaURL == "" {
		ollamaURL = "http://localhost:11434"
	}
	ollamaModel := os.Getenv("OLLAMA_EMBED_MODEL")
	if ollamaModel == "" {
		ollamaModel = "nomic-embed-text"
	}

	// Test if Ollama is available
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get(ollamaURL + "/api/tags")
	if err != nil {
		return nil
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil
	}

	return NewOllamaEmbedder(ollamaURL, ollamaModel)
}

// ======================================================================================
// Tracked Embedder (wraps any embedder with cost tracking)
// ======================================================================================

// TrackedEmbedder wraps an embedder and tracks costs
type TrackedEmbedder struct {
	inner       Embedder
	costTracker *CostTracker
	model       string
}

// NewTrackedEmbedder creates a new cost-tracking embedder wrapper
func NewTrackedEmbedder(inner Embedder, costTracker *CostTracker) *TrackedEmbedder {
	model := "unknown"
	if _, ok := inner.(*OpenAIEmbedder); ok {
		model = "text-embedding-3-small"
	}
	return &TrackedEmbedder{
		inner:       inner,
		costTracker: costTracker,
		model:       model,
	}
}

func (t *TrackedEmbedder) Dim() int {
	return t.inner.Dim()
}

func (t *TrackedEmbedder) Embed(text string) ([]float32, error) {
	// Estimate tokens (rough: ~4 chars per token for English)
	tokens := len(text) / 4
	if tokens < 1 {
		tokens = 1
	}

	// Track cost before embedding (so we record even on failure)
	if t.costTracker != nil {
		t.costTracker.RecordEmbedding(tokens, t.model, "embed")
	}

	return t.inner.Embed(text)
}

// ======================================================================================
// Mode-Aware Initialization (replaces initEmbedder for mode-based startup)
// ======================================================================================

// InitEmbedderForMode creates an embedder based on the current mode configuration
func InitEmbedderForMode(mode *ModeConfig, costTracker *CostTracker) (Embedder, error) {
	factory := NewEmbedderFactory(mode, costTracker)
	return factory.CreateEmbedder()
}

// InitRerankerForMode creates a reranker appropriate for the mode
func InitRerankerForMode(embedder Embedder) Reranker {
	// Try ONNX reranker first (works in both modes)
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
			fmt.Println(">>> Using ONNX reranker")
			return rr
		}
	}

	// Fall back to simple reranker (uses embedder for cosine similarity)
	return &SimpleReranker{Embedder: embedder}
}
