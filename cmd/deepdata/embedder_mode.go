package main

import (
	"fmt"
	"net/http"
	"os"
	"strconv"
	"strings"
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
	mode *ModeConfig
}

// NewEmbedderFactory creates a new factory for the given mode
func NewEmbedderFactory(mode *ModeConfig) *EmbedderFactory {
	return &EmbedderFactory{mode: mode}
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

// createLocalEmbedder creates an embedder for LOCAL mode.
// OpenAI is only selected when explicitly requested via EMBEDDER_TYPE=openai.
// Default local behavior remains ONNX > Ollama > Hash.
func (f *EmbedderFactory) createLocalEmbedder() (Embedder, error) {
	if strings.EqualFold(os.Getenv("EMBEDDER_TYPE"), "hash") {
		dim := 384
		if d := os.Getenv("EMBED_DIM"); d != "" {
			if v, err := strconv.Atoi(d); err == nil {
				dim = v
			}
		}
		fmt.Printf(">>> [LOCAL] Using hash embedder (%dd)\n", dim)
		f.mode.Dimension = dim
		f.mode.EmbedderType = "hash"
		f.mode.EmbedderModel = fmt.Sprintf("hash-%d", dim)
		return NewHashEmbedder(dim), nil
	}

	// Explicit provider selection via EMBEDDER_TYPE
	embType := strings.ToLower(os.Getenv("EMBEDDER_TYPE"))

	if embType == "openai" {
		apiKey := os.Getenv("OPENAI_API_KEY")
		if apiKey == "" {
			return nil, fmt.Errorf("EMBEDDER_TYPE=openai requires OPENAI_API_KEY")
		}
		fmt.Println(">>> [LOCAL] Using OpenAI embedder (text-embedding-3-small, 1536d)")
		f.mode.Dimension = 1536
		f.mode.EmbedderType = "openai"
		f.mode.EmbedderModel = "text-embedding-3-small"
		f.mode.CostPer1MToken = 0.02
		return NewOpenAIEmbedder(apiKey), nil
	}

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

// createProEmbedder creates an embedder for PRO mode. OpenAI is the only
// hosted provider the release candidate ships; the Gemini/Voyage/Jina/Cohere/
// Mistral vendors were retired under SYS-03.
func (f *EmbedderFactory) createProEmbedder() (Embedder, error) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("PRO mode requires OPENAI_API_KEY environment variable")
	}

	fmt.Println(">>> [PRO] Using OpenAI embedder (text-embedding-3-small, 1536d)")
	f.mode.Dimension = 1536
	f.mode.EmbedderType = "openai"
	f.mode.EmbedderModel = "text-embedding-3-small"
	f.mode.CostPer1MToken = 0.02

	return NewOpenAIEmbedder(apiKey), nil
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

	// Test if Ollama is available.
	// Same reasoning as the probe in main.go: ollamaURL comes from the OLLAMA_URL
	// environment variable, defaulted to loopback above, and is never influenced by
	// a request. The taint analyser cannot see that os.Getenv is a trust boundary.
	client := &http.Client{Timeout: 3 * time.Second}
	// #nosec G704 -- URL is operator configuration (OLLAMA_URL env), not attacker-controlled input.
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
