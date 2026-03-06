// Package feedback provides a feedback-driven learning system for improving
// retrieval quality over time. It tracks search interactions, processes user
// feedback, and applies learned adjustments to future searches.
//
// This package implements functionality similar to Cognee's feedback system,
// allowing the knowledge graph and vector search to improve based on user signals.
package feedback

import (
	"crypto/sha256"
	"encoding/hex"
	"time"
)

// Interaction represents a single search interaction.
type Interaction struct {
	ID         string            `json:"id"`
	Query      string            `json:"query"`
	QueryHash  string            `json:"query_hash"`
	ResultIDs  []string          `json:"result_ids"`
	Scores     []float32         `json:"scores"`
	Timestamp  time.Time         `json:"timestamp"`
	UserID     string            `json:"user_id,omitempty"`
	SessionID  string            `json:"session_id,omitempty"`
	Collection string            `json:"collection,omitempty"`
	SearchMode string            `json:"search_mode,omitempty"` // vector, graph, hybrid
	DurationMs int64             `json:"duration_ms,omitempty"`
	Metadata   map[string]string `json:"metadata,omitempty"`
}

// Feedback represents user feedback on an interaction.
type Feedback struct {
	ID            string            `json:"id"`
	InteractionID string            `json:"interaction_id"`
	Type          FeedbackType      `json:"type"`
	Text          string            `json:"text,omitempty"`         // Natural language feedback
	Sentiment     float32           `json:"sentiment"`              // -1 to 1
	ClickedIDs    []string          `json:"clicked_ids,omitempty"`  // Which results were clicked/used
	RejectedIDs   []string          `json:"rejected_ids,omitempty"` // Which results were rejected
	Rating        int               `json:"rating,omitempty"`       // 1-5 explicit rating
	Timestamp     time.Time         `json:"timestamp"`
	UserID        string            `json:"user_id,omitempty"`
	Metadata      map[string]string `json:"metadata,omitempty"`
}

// FeedbackType categorizes the type of feedback.
type FeedbackType string

const (
	// Explicit feedback types
	FeedbackTypeExplicit   FeedbackType = "explicit"    // User explicitly rated/commented
	FeedbackTypeThumbsUp   FeedbackType = "thumbs_up"   // Positive signal
	FeedbackTypeThumbsDown FeedbackType = "thumbs_down" // Negative signal

	// Implicit feedback types
	FeedbackTypeClick   FeedbackType = "click"   // User clicked a result
	FeedbackTypeDwell   FeedbackType = "dwell"   // User spent time on result
	FeedbackTypeIgnore  FeedbackType = "ignore"  // User ignored result
	FeedbackTypeRequery FeedbackType = "requery" // User rephrased query (negative signal)

	// Natural language feedback
	FeedbackTypeNatural FeedbackType = "natural" // Free-form text feedback
)

// BoostScore represents a learned adjustment for a document or edge.
type BoostScore struct {
	TargetID   string    `json:"target_id"`   // Document ID or edge key
	TargetType string    `json:"target_type"` // "document", "edge", "node"
	QueryHash  string    `json:"query_hash"`  // For query-specific boosts (empty = global)
	Boost      float32   `json:"boost"`       // Multiplicative factor (1.0 = neutral)
	Confidence float32   `json:"confidence"`  // How confident we are in this boost (0-1)
	Count      int       `json:"count"`       // Number of feedback signals contributing
	LastUpdate time.Time `json:"last_update"`
	DecayRate  float32   `json:"decay_rate"` // How fast the boost decays (0 = no decay)
}

// DecayedBoost returns the boost after applying time-based decay.
func (b *BoostScore) DecayedBoost(now time.Time) float32 {
	if b.DecayRate <= 0 {
		return b.Boost
	}

	elapsed := now.Sub(b.LastUpdate).Hours() / 24 // Days
	decay := float32(1.0) - (b.DecayRate * float32(elapsed))
	if decay < 0 {
		decay = 0
	}

	// Decay towards 1.0 (neutral)
	return 1.0 + (b.Boost-1.0)*decay
}

// QueryFeedbackStats aggregates feedback for a specific query pattern.
type QueryFeedbackStats struct {
	QueryHash        string    `json:"query_hash"`
	QueryExample     string    `json:"query_example"` // One example query
	TotalSearches    int       `json:"total_searches"`
	PositiveFeedback int       `json:"positive_feedback"`
	NegativeFeedback int       `json:"negative_feedback"`
	AvgClickThrough  float32   `json:"avg_click_through"`
	TopClickedIDs    []string  `json:"top_clicked_ids"`
	LastSearch       time.Time `json:"last_search"`
}

// FeedbackConfig configures the feedback system.
type FeedbackConfig struct {
	// Storage
	StorePath       string `json:"store_path"`       // Path for WAL/persistence
	MaxInteractions int    `json:"max_interactions"` // Max interactions to keep
	RetentionDays   int    `json:"retention_days"`   // How long to keep data

	// Boost calculation
	DefaultDecayRate    float32 `json:"default_decay_rate"`   // Default boost decay (per day)
	PositiveBoostDelta  float32 `json:"positive_boost_delta"` // How much positive feedback boosts
	NegativeBoostDelta  float32 `json:"negative_boost_delta"` // How much negative feedback penalizes
	MinBoost            float32 `json:"min_boost"`            // Minimum allowed boost
	MaxBoost            float32 `json:"max_boost"`            // Maximum allowed boost
	ConfidenceThreshold float32 `json:"confidence_threshold"` // Min confidence to apply boost

	// LLM sentiment analysis
	EnableLLMSentiment bool   `json:"enable_llm_sentiment"` // Use LLM for natural language feedback
	LLMProvider        string `json:"llm_provider"`         // "ollama", "openai"
	LLMModel           string `json:"llm_model"`
	LLMURL             string `json:"llm_url"`
	LLMAPIKey          string `json:"llm_api_key,omitempty"`

	// Implicit feedback
	DwellTimeThresholdMs int     `json:"dwell_time_threshold_ms"` // Min dwell time for positive signal
	ClickPositiveWeight  float32 `json:"click_positive_weight"`
	DwellPositiveWeight  float32 `json:"dwell_positive_weight"`
	IgnoreNegativeWeight float32 `json:"ignore_negative_weight"`
}

// DefaultFeedbackConfig returns sensible defaults.
func DefaultFeedbackConfig() FeedbackConfig {
	return FeedbackConfig{
		MaxInteractions:      10000,
		RetentionDays:        90,
		DefaultDecayRate:     0.01,  // 1% per day
		PositiveBoostDelta:   0.1,   // +10% per positive signal
		NegativeBoostDelta:   0.15,  // -15% per negative signal
		MinBoost:             0.1,   // Don't go below 10%
		MaxBoost:             3.0,   // Don't go above 300%
		ConfidenceThreshold:  0.3,   // Need 30% confidence to apply
		EnableLLMSentiment:   false, // Off by default
		LLMProvider:          "ollama",
		LLMModel:             "llama3.2",
		LLMURL:               "http://localhost:11434",
		DwellTimeThresholdMs: 5000, // 5 seconds
		ClickPositiveWeight:  0.5,
		DwellPositiveWeight:  0.3,
		IgnoreNegativeWeight: 0.1,
	}
}

// NewInteraction creates a new interaction record.
func NewInteraction(query string, resultIDs []string, scores []float32) *Interaction {
	now := time.Now()
	hash := sha256.Sum256([]byte(query))

	return &Interaction{
		ID:        generateID(query, now),
		Query:     query,
		QueryHash: hex.EncodeToString(hash[:8]),
		ResultIDs: resultIDs,
		Scores:    scores,
		Timestamp: now,
	}
}

// NewFeedback creates a new feedback record.
func NewFeedback(interactionID string, feedbackType FeedbackType) *Feedback {
	now := time.Now()

	return &Feedback{
		ID:            generateID(interactionID, now),
		InteractionID: interactionID,
		Type:          feedbackType,
		Timestamp:     now,
	}
}

// generateID creates a unique ID.
func generateID(seed string, t time.Time) string {
	hash := sha256.Sum256([]byte(seed + t.Format(time.RFC3339Nano)))
	return hex.EncodeToString(hash[:12])
}
