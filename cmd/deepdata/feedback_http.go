package main

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/phenomenon0/vectordb/internal/feedback"
)

// FeedbackHandler handles feedback HTTP endpoints.
type FeedbackHandler struct {
	store     *feedback.Store
	processor *feedback.Processor
	mu        sync.RWMutex
}

// feedbackHandlerInstance is the global singleton (lazy-initialized)
var (
	feedbackHandlerOnce     sync.Once
	feedbackHandlerInstance *FeedbackHandler
)

// GetFeedbackHandler returns the singleton feedback handler.
func GetFeedbackHandler() *FeedbackHandler {
	feedbackHandlerOnce.Do(func() {
		cfg := feedback.DefaultFeedbackConfig()

		// Use data directory from environment or default
		dataDir := os.Getenv("VECTORDB_DATA_DIR")
		if dataDir == "" {
			dataDir = "./data"
		}
		cfg.StorePath = filepath.Join(dataDir, "feedback")

		// Enable LLM sentiment if configured
		if os.Getenv("FEEDBACK_LLM_ENABLED") == "1" || os.Getenv("FEEDBACK_LLM_ENABLED") == "true" {
			cfg.EnableLLMSentiment = true
			if provider := os.Getenv("FEEDBACK_LLM_PROVIDER"); provider != "" {
				cfg.LLMProvider = provider
			}
			if model := os.Getenv("FEEDBACK_LLM_MODEL"); model != "" {
				cfg.LLMModel = model
			}
			if url := os.Getenv("OLLAMA_HOST"); url != "" {
				cfg.LLMURL = url
			}
			if key := os.Getenv("OPENAI_API_KEY"); key != "" {
				cfg.LLMAPIKey = key
			}
		}

		store, err := feedback.NewStore(cfg)
		if err != nil {
			// Log error but don't fail - feedback is optional
			store = nil
		}

		var processor *feedback.Processor
		if store != nil {
			processor = feedback.NewProcessor(store, cfg)
		}

		feedbackHandlerInstance = &FeedbackHandler{
			store:     store,
			processor: processor,
		}
	})
	return feedbackHandlerInstance
}

// CloseFeedbackHandler closes the feedback handler.
func CloseFeedbackHandler() error {
	h := feedbackHandlerInstance
	if h != nil && h.store != nil {
		return h.store.Close()
	}
	return nil
}

// ========== HTTP Request/Response Types ==========

// FeedbackRequest represents a feedback submission.
type FeedbackRequest struct {
	InteractionID string            `json:"interaction_id"` // Required
	Type          string            `json:"type"`           // "natural", "explicit", "implicit"
	Text          string            `json:"text,omitempty"` // For natural feedback
	Rating        int               `json:"rating,omitempty"`
	ClickedIDs    []string          `json:"clicked_ids,omitempty"`
	RejectedIDs   []string          `json:"rejected_ids,omitempty"`
	SignalType    string            `json:"signal_type,omitempty"` // For implicit: click, dwell, ignore, requery
	DurationMs    int               `json:"duration_ms,omitempty"` // For dwell time
	UserID        string            `json:"user_id,omitempty"`
	Metadata      map[string]string `json:"metadata,omitempty"`
}

// FeedbackResponse is the response after submitting feedback.
type FeedbackResponse struct {
	ID        string  `json:"id"`
	Sentiment float32 `json:"sentiment,omitempty"`
	Recorded  bool    `json:"recorded"`
}

// InteractionRequest records a search interaction.
type InteractionRequest struct {
	Query      string            `json:"query"`
	ResultIDs  []string          `json:"result_ids"`
	Scores     []float32         `json:"scores,omitempty"`
	UserID     string            `json:"user_id,omitempty"`
	SessionID  string            `json:"session_id,omitempty"`
	Collection string            `json:"collection,omitempty"`
	SearchMode string            `json:"search_mode,omitempty"` // vector, graph, hybrid
	DurationMs int64             `json:"duration_ms,omitempty"`
	Metadata   map[string]string `json:"metadata,omitempty"`
}

// InteractionResponse is the response after recording an interaction.
type InteractionResponse struct {
	ID        string `json:"id"`
	QueryHash string `json:"query_hash"`
	Recorded  bool   `json:"recorded"`
}

// FeedbackStatsResponse contains feedback system statistics.
type FeedbackStatsResponse struct {
	TotalInteractions  int  `json:"total_interactions"`
	TotalFeedback      int  `json:"total_feedback"`
	TotalBoosts        int  `json:"total_boosts"`
	TotalQueryPatterns int  `json:"total_query_patterns"`
	Enabled            bool `json:"enabled"`
}

// BatchFeedbackRequest contains multiple feedback items.
type BatchFeedbackRequest struct {
	Items []FeedbackRequest `json:"items"`
}

// BatchFeedbackResponse contains results for batch submission.
type BatchFeedbackResponse struct {
	Processed int      `json:"processed"`
	Failed    int      `json:"failed"`
	Errors    []string `json:"errors,omitempty"`
}

// BoostQueryRequest requests boosts for specific targets.
type BoostQueryRequest struct {
	TargetIDs []string `json:"target_ids"`
	QueryHash string   `json:"query_hash,omitempty"`
}

// BoostQueryResponse contains boosts for targets.
type BoostQueryResponse struct {
	Boosts map[string]float32 `json:"boosts"`
}

// ========== HTTP Handlers ==========

// HandleFeedback handles POST /v2/feedback
func (h *FeedbackHandler) HandleFeedback(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if h.store == nil || h.processor == nil {
		sendJSON(w, FeedbackResponse{Recorded: false})
		return
	}

	var req FeedbackRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	if req.InteractionID == "" {
		http.Error(w, "interaction_id is required", http.StatusBadRequest)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()

	var fb *feedback.Feedback
	var err error

	switch req.Type {
	case "natural":
		fb, err = h.processor.ProcessNaturalFeedback(ctx, req.InteractionID, req.Text)
	case "explicit":
		fb, err = h.processor.ProcessExplicitFeedback(ctx, req.InteractionID, req.Rating, req.ClickedIDs, req.RejectedIDs)
	case "implicit":
		signal := feedback.ImplicitSignal{
			Type:       req.SignalType,
			TargetIDs:  req.ClickedIDs,
			DurationMs: req.DurationMs,
		}
		fb, err = h.processor.ProcessImplicitFeedback(ctx, req.InteractionID, signal)
	default:
		// Default to explicit if type not specified but rating is provided
		if req.Rating > 0 {
			fb, err = h.processor.ProcessExplicitFeedback(ctx, req.InteractionID, req.Rating, req.ClickedIDs, req.RejectedIDs)
		} else if req.Text != "" {
			fb, err = h.processor.ProcessNaturalFeedback(ctx, req.InteractionID, req.Text)
		} else {
			http.Error(w, "type is required or provide rating/text", http.StatusBadRequest)
			return
		}
	}

	if err != nil {
		http.Error(w, "failed to process feedback: "+err.Error(), http.StatusInternalServerError)
		return
	}

	resp := FeedbackResponse{
		ID:        fb.ID,
		Sentiment: fb.Sentiment,
		Recorded:  true,
	}
	sendJSON(w, resp)
}

// HandleFeedbackBatch handles POST /v2/feedback/batch
func (h *FeedbackHandler) HandleFeedbackBatch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if h.store == nil || h.processor == nil {
		sendJSON(w, BatchFeedbackResponse{Processed: 0, Failed: 0})
		return
	}

	var req BatchFeedbackRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	// Convert to feedback.FeedbackItem format
	items := make([]feedback.FeedbackItem, len(req.Items))
	for i, item := range req.Items {
		items[i] = feedback.FeedbackItem{
			Type:          item.Type,
			InteractionID: item.InteractionID,
			Text:          item.Text,
			Rating:        item.Rating,
			ClickedIDs:    item.ClickedIDs,
			RejectedIDs:   item.RejectedIDs,
			SignalType:    item.SignalType,
			DurationMs:    item.DurationMs,
		}
	}

	results, _ := h.processor.ProcessBatch(ctx, items)

	processed := 0
	var errors []string
	for i, result := range results {
		if result != nil {
			processed++
		} else {
			errors = append(errors, "failed to process item "+string(rune(i)))
		}
	}

	sendJSON(w, BatchFeedbackResponse{
		Processed: processed,
		Failed:    len(req.Items) - processed,
		Errors:    errors,
	})
}

// HandleInteraction handles POST /v2/interaction
func (h *FeedbackHandler) HandleInteraction(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if h.store == nil {
		sendJSON(w, InteractionResponse{Recorded: false})
		return
	}

	var req InteractionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	if req.Query == "" || len(req.ResultIDs) == 0 {
		http.Error(w, "query and result_ids are required", http.StatusBadRequest)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	interaction := feedback.NewInteraction(req.Query, req.ResultIDs, req.Scores)
	interaction.UserID = req.UserID
	interaction.SessionID = req.SessionID
	interaction.Collection = req.Collection
	interaction.SearchMode = req.SearchMode
	interaction.DurationMs = req.DurationMs
	interaction.Metadata = req.Metadata

	if err := h.store.RecordInteraction(ctx, interaction); err != nil {
		http.Error(w, "failed to record interaction: "+err.Error(), http.StatusInternalServerError)
		return
	}

	sendJSON(w, InteractionResponse{
		ID:        interaction.ID,
		QueryHash: interaction.QueryHash,
		Recorded:  true,
	})
}

// HandleFeedbackStats handles GET /v2/feedback/stats
func (h *FeedbackHandler) HandleFeedbackStats(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if h.store == nil {
		sendJSON(w, FeedbackStatsResponse{Enabled: false})
		return
	}

	stats := h.store.Stats()
	sendJSON(w, FeedbackStatsResponse{
		TotalInteractions:  stats.TotalInteractions,
		TotalFeedback:      stats.TotalFeedback,
		TotalBoosts:        stats.TotalBoosts,
		TotalQueryPatterns: stats.TotalQueryPatterns,
		Enabled:            true,
	})
}

// HandleBoostQuery handles POST /v2/feedback/boosts
func (h *FeedbackHandler) HandleBoostQuery(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if h.store == nil {
		sendJSON(w, BoostQueryResponse{Boosts: make(map[string]float32)})
		return
	}

	var req BoostQueryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	boosts := h.store.GetBoosts(req.TargetIDs, req.QueryHash)
	sendJSON(w, BoostQueryResponse{Boosts: boosts})
}

// HandleImplicitFeedback handles POST /v2/feedback/implicit
// Convenience endpoint for implicit signals (click, dwell, etc.)
func (h *FeedbackHandler) HandleImplicitFeedback(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if h.store == nil || h.processor == nil {
		sendJSON(w, FeedbackResponse{Recorded: false})
		return
	}

	var req struct {
		InteractionID string   `json:"interaction_id"`
		SignalType    string   `json:"signal_type"` // click, dwell, ignore, requery
		TargetIDs     []string `json:"target_ids"`
		DurationMs    int      `json:"duration_ms,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	if req.InteractionID == "" || req.SignalType == "" {
		http.Error(w, "interaction_id and signal_type are required", http.StatusBadRequest)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	signal := feedback.ImplicitSignal{
		Type:       req.SignalType,
		TargetIDs:  req.TargetIDs,
		DurationMs: req.DurationMs,
	}

	fb, err := h.processor.ProcessImplicitFeedback(ctx, req.InteractionID, signal)
	if err != nil {
		http.Error(w, "failed to process feedback: "+err.Error(), http.StatusInternalServerError)
		return
	}

	sendJSON(w, FeedbackResponse{
		ID:        fb.ID,
		Sentiment: fb.Sentiment,
		Recorded:  true,
	})
}

// sendJSON sends a JSON response
func sendJSON(w http.ResponseWriter, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}

// RegisterFeedbackHandlers registers all feedback endpoints on the mux.
func RegisterFeedbackHandlers(mux *http.ServeMux) {
	h := GetFeedbackHandler()

	mux.HandleFunc("/v2/feedback", h.HandleFeedback)
	mux.HandleFunc("/v2/feedback/batch", h.HandleFeedbackBatch)
	mux.HandleFunc("/v2/feedback/stats", h.HandleFeedbackStats)
	mux.HandleFunc("/v2/feedback/boosts", h.HandleBoostQuery)
	mux.HandleFunc("/v2/feedback/implicit", h.HandleImplicitFeedback)
	mux.HandleFunc("/v2/interaction", h.HandleInteraction)
}
