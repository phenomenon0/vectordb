package main

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/phenomenon0/Agent-GO/vectordb/extraction"
)

// ExtractionHandler handles knowledge graph extraction HTTP endpoints.
type ExtractionHandler struct {
	extractor extraction.Extractor
	mu        sync.RWMutex
}

// extractionHandlerInstance is the global singleton (lazy-initialized)
var (
	extractionHandlerOnce     sync.Once
	extractionHandlerInstance *ExtractionHandler
)

// GetExtractionHandler returns the singleton extraction handler.
func GetExtractionHandler() *ExtractionHandler {
	extractionHandlerOnce.Do(func() {
		cfg := extraction.DefaultConfig()

		// Configure from environment
		if provider := os.Getenv("EXTRACTION_PROVIDER"); provider != "" {
			cfg.Provider = provider
		}
		if model := os.Getenv("EXTRACTION_MODEL"); model != "" {
			cfg.Model = model
		}
		if url := os.Getenv("OLLAMA_HOST"); url != "" {
			cfg.BaseURL = url
		}
		if apiKey := os.Getenv("OPENAI_API_KEY"); apiKey != "" {
			cfg.APIKey = apiKey
		}

		extractor, err := extraction.NewExtractor(cfg)
		if err != nil {
			// Log error but don't fail - extraction is optional
			extractor = nil
		}

		extractionHandlerInstance = &ExtractionHandler{
			extractor: extractor,
		}
	})
	return extractionHandlerInstance
}

// ========== HTTP Request/Response Types ==========

// ExtractRequest is the request body for extraction.
type ExtractRequest struct {
	Content           string `json:"content"`                       // Text to extract from (required)
	Provider          string `json:"provider,omitempty"`            // "ollama" or "openai" (uses default if not specified)
	Model             string `json:"model,omitempty"`               // Model name (uses default if not specified)
	Temporal          bool   `json:"temporal,omitempty"`            // Extract temporal events
	StoreToCollection string `json:"store_to_collection,omitempty"` // Auto-store extracted entities
}

// ExtractResponse is the response from extraction.
type ExtractResponse struct {
	Nodes  []extraction.Node  `json:"nodes"`
	Edges  []extraction.Edge  `json:"edges"`
	Events []TemporalEventDTO `json:"events,omitempty"`
	Stats  ExtractStats       `json:"stats"`
}

// TemporalEventDTO is a DTO for temporal events
type TemporalEventDTO struct {
	ID          string   `json:"id"`
	Description string   `json:"description"`
	Year        int      `json:"year,omitempty"`
	DateText    string   `json:"date_text,omitempty"`
	Entities    []string `json:"entities,omitempty"`
}

// ExtractStats contains extraction statistics.
type ExtractStats struct {
	TotalNodes  int   `json:"total_nodes"`
	TotalEdges  int   `json:"total_edges"`
	TotalEvents int   `json:"total_events,omitempty"`
	DurationMs  int64 `json:"duration_ms"`
	ContentLen  int   `json:"content_length"`
}

// BatchExtractRequest is the request body for batch extraction.
type BatchExtractRequest struct {
	Contents []string `json:"contents"` // Multiple text chunks
	Provider string   `json:"provider,omitempty"`
	Model    string   `json:"model,omitempty"`
	Temporal bool     `json:"temporal,omitempty"`
}

// BatchExtractResponse contains batch extraction results.
type BatchExtractResponse struct {
	Results   []ExtractResponse `json:"results"`
	Merged    *ExtractResponse  `json:"merged,omitempty"` // Merged/deduplicated graph
	Processed int               `json:"processed"`
	Failed    int               `json:"failed"`
	Errors    []string          `json:"errors,omitempty"`
}

// ========== HTTP Handlers ==========

// HandleExtract handles POST /v2/extract
func (h *ExtractionHandler) HandleExtract(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if h.extractor == nil {
		http.Error(w, "extraction not configured - set OLLAMA_HOST or OPENAI_API_KEY", http.StatusServiceUnavailable)
		return
	}

	var req ExtractRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	if req.Content == "" {
		http.Error(w, "content is required", http.StatusBadRequest)
		return
	}

	// Use custom extractor if provider/model specified
	extractor := h.extractor
	if req.Provider != "" || req.Model != "" {
		cfg := extraction.DefaultConfig()
		if req.Provider != "" {
			cfg.Provider = req.Provider
		}
		if req.Model != "" {
			cfg.Model = req.Model
		}
		if req.Provider == "openai" {
			cfg.APIKey = os.Getenv("OPENAI_API_KEY")
		}
		if newExt, err := extraction.NewExtractor(cfg); err == nil {
			extractor = newExt
		}
	}

	ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
	defer cancel()

	start := time.Now()
	var resp ExtractResponse

	if req.Temporal {
		// Extract with temporal events
		tkg, err := extractor.ExtractTemporal(ctx, req.Content)
		if err != nil {
			http.Error(w, "extraction failed: "+err.Error(), http.StatusInternalServerError)
			return
		}
		resp.Nodes = tkg.Nodes
		resp.Edges = tkg.Edges

		// Convert temporal events
		resp.Events = make([]TemporalEventDTO, len(tkg.Events))
		for i, e := range tkg.Events {
			resp.Events[i] = TemporalEventDTO{
				ID:          e.ID,
				Description: e.Description,
				Year:        e.Year,
				DateText:    e.DateText,
				Entities:    e.Entities,
			}
		}
		resp.Stats.TotalEvents = len(tkg.Events)
	} else {
		// Standard extraction
		kg, err := extractor.Extract(ctx, req.Content)
		if err != nil {
			http.Error(w, "extraction failed: "+err.Error(), http.StatusInternalServerError)
			return
		}
		resp.Nodes = kg.Nodes
		resp.Edges = kg.Edges
	}

	resp.Stats = ExtractStats{
		TotalNodes:  len(resp.Nodes),
		TotalEdges:  len(resp.Edges),
		TotalEvents: len(resp.Events),
		DurationMs:  time.Since(start).Milliseconds(),
		ContentLen:  len(req.Content),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// HandleExtractBatch handles POST /v2/extract/batch
func (h *ExtractionHandler) HandleExtractBatch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if h.extractor == nil {
		http.Error(w, "extraction not configured", http.StatusServiceUnavailable)
		return
	}

	var req BatchExtractRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	if len(req.Contents) == 0 {
		http.Error(w, "contents is required", http.StatusBadRequest)
		return
	}

	// Use custom extractor if provider/model specified
	extractor := h.extractor
	if req.Provider != "" || req.Model != "" {
		cfg := extraction.DefaultConfig()
		if req.Provider != "" {
			cfg.Provider = req.Provider
		}
		if req.Model != "" {
			cfg.Model = req.Model
		}
		if req.Provider == "openai" {
			cfg.APIKey = os.Getenv("OPENAI_API_KEY")
		}
		if newExt, err := extraction.NewExtractor(cfg); err == nil {
			extractor = newExt
		}
	}

	ctx, cancel := context.WithTimeout(r.Context(), 120*time.Second)
	defer cancel()

	// Process batch
	kgs, err := extractor.ExtractBatch(ctx, req.Contents)
	if err != nil {
		http.Error(w, "batch extraction failed: "+err.Error(), http.StatusInternalServerError)
		return
	}

	resp := BatchExtractResponse{
		Results: make([]ExtractResponse, len(kgs)),
	}

	// Convert results
	var mergedKG extraction.KnowledgeGraph
	for i, kg := range kgs {
		if kg == nil {
			resp.Failed++
			resp.Errors = append(resp.Errors, "extraction failed for chunk "+string(rune(i)))
			continue
		}

		resp.Results[i] = ExtractResponse{
			Nodes: kg.Nodes,
			Edges: kg.Edges,
			Stats: ExtractStats{
				TotalNodes: len(kg.Nodes),
				TotalEdges: len(kg.Edges),
				ContentLen: len(req.Contents[i]),
			},
		}
		resp.Processed++

		// Merge into combined graph
		mergedKG.Merge(kg)
	}

	// Add merged result
	resp.Merged = &ExtractResponse{
		Nodes: mergedKG.Nodes,
		Edges: mergedKG.Edges,
		Stats: ExtractStats{
			TotalNodes: len(mergedKG.Nodes),
			TotalEdges: len(mergedKG.Edges),
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// HandleExtractTemporal handles POST /v2/extract/temporal
// Convenience endpoint specifically for temporal extraction.
func (h *ExtractionHandler) HandleExtractTemporal(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if h.extractor == nil {
		http.Error(w, "extraction not configured", http.StatusServiceUnavailable)
		return
	}

	var req ExtractRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	if req.Content == "" {
		http.Error(w, "content is required", http.StatusBadRequest)
		return
	}

	// Force temporal mode
	req.Temporal = true

	// Delegate to main handler
	h.HandleExtract(w, r)
}

// HandleExtractStatus handles GET /v2/extract/status
func (h *ExtractionHandler) HandleExtractStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	status := struct {
		Enabled  bool   `json:"enabled"`
		Provider string `json:"provider,omitempty"`
		Model    string `json:"model,omitempty"`
	}{
		Enabled: h.extractor != nil,
	}

	if h.extractor != nil {
		status.Provider = h.extractor.Provider()
		status.Model = h.extractor.Model()
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

// RegisterExtractionHandlers registers all extraction endpoints on the mux.
func RegisterExtractionHandlers(mux *http.ServeMux) {
	h := GetExtractionHandler()

	mux.HandleFunc("/v2/extract", h.HandleExtract)
	mux.HandleFunc("/v2/extract/batch", h.HandleExtractBatch)
	mux.HandleFunc("/v2/extract/temporal", h.HandleExtractTemporal)
	mux.HandleFunc("/v2/extract/status", h.HandleExtractStatus)
}
