package feedback

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"sort"
)

// Booster applies learned feedback adjustments to search results.
type Booster struct {
	store *Store
	cfg   BoosterConfig
}

// BoosterConfig configures the booster.
type BoosterConfig struct {
	// EnableQuerySpecific uses query-specific boosts when available
	EnableQuerySpecific bool `json:"enable_query_specific"`

	// EnableGlobalFallback falls back to global boosts
	EnableGlobalFallback bool `json:"enable_global_fallback"`

	// MinConfidence is the minimum confidence to apply boosts
	MinConfidence float32 `json:"min_confidence"`

	// MaxBoostImpact limits how much boosts can change scores
	MaxBoostImpact float32 `json:"max_boost_impact"` // e.g., 0.5 = 50% max change
}

// DefaultBoosterConfig returns sensible defaults.
func DefaultBoosterConfig() BoosterConfig {
	return BoosterConfig{
		EnableQuerySpecific:  true,
		EnableGlobalFallback: true,
		MinConfidence:        0.2,
		MaxBoostImpact:       0.5,
	}
}

// NewBooster creates a new booster.
func NewBooster(store *Store, cfg BoosterConfig) *Booster {
	return &Booster{
		store: store,
		cfg:   cfg,
	}
}

// SearchResult represents a search result to be boosted.
type SearchResult struct {
	ID    string  `json:"id"`
	Score float32 `json:"score"`
	Doc   string  `json:"doc,omitempty"`
}

// BoostedResult is a search result with boost information.
type BoostedResult struct {
	SearchResult
	OriginalScore float32 `json:"original_score"`
	Boost         float32 `json:"boost"`
	BoostedScore  float32 `json:"boosted_score"`
	BoostSource   string  `json:"boost_source"` // "query", "global", "none"
}

// ApplyBoosts applies learned boosts to search results.
func (b *Booster) ApplyBoosts(ctx context.Context, query string, results []SearchResult) []BoostedResult {
	if len(results) == 0 {
		return []BoostedResult{} // Return empty slice, not nil
	}

	// Compute query hash
	queryHash := hashQuery(query)

	// Get IDs
	ids := make([]string, len(results))
	for i, r := range results {
		ids[i] = r.ID
	}

	// Get boosts from store
	boosts := b.store.GetBoosts(ids, queryHash)

	// Apply boosts
	boosted := make([]BoostedResult, len(results))
	for i, r := range results {
		br := BoostedResult{
			SearchResult:  r,
			OriginalScore: r.Score,
			Boost:         1.0,
			BoostedScore:  r.Score,
			BoostSource:   "none",
		}

		if boost, ok := boosts[r.ID]; ok && boost != 1.0 {
			// Limit boost impact
			limitedBoost := b.limitBoost(boost)
			br.Boost = limitedBoost
			br.BoostedScore = r.Score * limitedBoost

			// Determine source
			if b.cfg.EnableQuerySpecific {
				br.BoostSource = "query"
			} else {
				br.BoostSource = "global"
			}
		}

		boosted[i] = br
	}

	// Re-sort by boosted score
	sort.Slice(boosted, func(i, j int) bool {
		return boosted[i].BoostedScore > boosted[j].BoostedScore
	})

	return boosted
}

// limitBoost constrains boost to configured limits.
func (b *Booster) limitBoost(boost float32) float32 {
	if b.cfg.MaxBoostImpact <= 0 {
		return boost
	}

	minBoost := 1.0 - b.cfg.MaxBoostImpact
	maxBoost := 1.0 + b.cfg.MaxBoostImpact

	if boost < minBoost {
		return minBoost
	}
	if boost > maxBoost {
		return maxBoost
	}
	return boost
}

// RecordAndBoost records an interaction and returns boosted results.
// This is a convenience method that combines recording and boosting.
func (b *Booster) RecordAndBoost(ctx context.Context, query string, results []SearchResult, opts RecordOptions) ([]BoostedResult, string, error) {
	// First apply boosts
	boosted := b.ApplyBoosts(ctx, query, results)

	// Then record the interaction (with original order for feedback purposes)
	if opts.SaveInteraction {
		ids := make([]string, len(results))
		scores := make([]float32, len(results))
		for i, r := range results {
			ids[i] = r.ID
			scores[i] = r.Score
		}

		interaction := NewInteraction(query, ids, scores)
		interaction.UserID = opts.UserID
		interaction.SessionID = opts.SessionID
		interaction.Collection = opts.Collection
		interaction.SearchMode = opts.SearchMode
		interaction.Metadata = opts.Metadata

		if err := b.store.RecordInteraction(ctx, interaction); err != nil {
			return boosted, "", err
		}

		return boosted, interaction.ID, nil
	}

	return boosted, "", nil
}

// RecordOptions configures interaction recording.
type RecordOptions struct {
	SaveInteraction bool              `json:"save_interaction"`
	UserID          string            `json:"user_id,omitempty"`
	SessionID       string            `json:"session_id,omitempty"`
	Collection      string            `json:"collection,omitempty"`
	SearchMode      string            `json:"search_mode,omitempty"`
	Metadata        map[string]string `json:"metadata,omitempty"`
}

// GetTopBoostedIDs returns the IDs most positively boosted for a query.
func (b *Booster) GetTopBoostedIDs(query string, limit int) []string {
	queryHash := hashQuery(query)

	b.store.mu.RLock()
	defer b.store.mu.RUnlock()

	type boostEntry struct {
		id    string
		boost float32
	}

	var entries []boostEntry

	// Collect boosts for this query
	for _, boost := range b.store.boosts {
		if boost.QueryHash == queryHash && boost.Boost > 1.0 && boost.Confidence >= b.cfg.MinConfidence {
			entries = append(entries, boostEntry{id: boost.TargetID, boost: boost.Boost})
		}
	}

	// Sort by boost descending
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].boost > entries[j].boost
	})

	// Return top N IDs
	result := make([]string, 0, limit)
	for i, e := range entries {
		if i >= limit {
			break
		}
		result = append(result, e.id)
	}

	return result
}

// GetRecommendedResults returns results that have been positively boosted.
// This can be used for "recommended for you" type features.
func (b *Booster) GetRecommendedResults(limit int) []string {
	b.store.mu.RLock()
	defer b.store.mu.RUnlock()

	type boostEntry struct {
		id         string
		boost      float32
		confidence float32
	}

	var entries []boostEntry

	// Collect global boosts
	for _, boost := range b.store.boosts {
		if boost.QueryHash == "" && boost.Boost > 1.0 && boost.Confidence >= b.cfg.MinConfidence {
			entries = append(entries, boostEntry{
				id:         boost.TargetID,
				boost:      boost.Boost,
				confidence: boost.Confidence,
			})
		}
	}

	// Sort by boost * confidence
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].boost*entries[i].confidence > entries[j].boost*entries[j].confidence
	})

	// Return top N IDs
	result := make([]string, 0, limit)
	for i, e := range entries {
		if i >= limit {
			break
		}
		result = append(result, e.id)
	}

	return result
}

// EdgeBoost represents a boost to apply to graph edges.
type EdgeBoost struct {
	SourceID string  `json:"source_id"`
	TargetID string  `json:"target_id"`
	Boost    float32 `json:"boost"`
}

// GetEdgeBoosts returns boosts for graph edges based on co-clicked results.
// When two results are frequently clicked together, their edge weight should increase.
func (b *Booster) GetEdgeBoosts() []EdgeBoost {
	b.store.mu.RLock()
	defer b.store.mu.RUnlock()

	// Build co-click matrix
	coClicks := make(map[string]map[string]int)

	for _, feedbackList := range b.store.feedback {
		for _, fb := range feedbackList {
			if len(fb.ClickedIDs) < 2 {
				continue
			}

			// All pairs of clicked IDs
			for i := 0; i < len(fb.ClickedIDs); i++ {
				for j := i + 1; j < len(fb.ClickedIDs); j++ {
					id1, id2 := fb.ClickedIDs[i], fb.ClickedIDs[j]
					if id1 > id2 {
						id1, id2 = id2, id1 // Normalize order
					}

					if coClicks[id1] == nil {
						coClicks[id1] = make(map[string]int)
					}
					coClicks[id1][id2]++
				}
			}
		}
	}

	// Convert to edge boosts
	var boosts []EdgeBoost
	for source, targets := range coClicks {
		for target, count := range targets {
			if count >= 2 { // Minimum co-click threshold
				boost := 1.0 + 0.1*float32(count) // 10% boost per co-click
				if boost > 2.0 {
					boost = 2.0 // Cap at 2x
				}
				boosts = append(boosts, EdgeBoost{
					SourceID: source,
					TargetID: target,
					Boost:    boost,
				})
			}
		}
	}

	return boosts
}

// hashQuery creates a hash of a query for lookup.
func hashQuery(query string) string {
	hash := sha256.Sum256([]byte(query))
	return hex.EncodeToString(hash[:8])
}

// FeedbackAwareSearch wraps a search function with feedback-based boosting.
type FeedbackAwareSearch struct {
	booster   *Booster
	processor *Processor
}

// NewFeedbackAwareSearch creates a new feedback-aware search wrapper.
func NewFeedbackAwareSearch(store *Store, cfg FeedbackConfig) *FeedbackAwareSearch {
	boosterCfg := DefaultBoosterConfig()
	return &FeedbackAwareSearch{
		booster:   NewBooster(store, boosterCfg),
		processor: NewProcessor(store, cfg),
	}
}

// Search wraps a search function with feedback boosting.
func (f *FeedbackAwareSearch) Search(
	ctx context.Context,
	query string,
	searchFn func(ctx context.Context, query string) ([]SearchResult, error),
	opts RecordOptions,
) ([]BoostedResult, string, error) {
	// Execute the search
	results, err := searchFn(ctx, query)
	if err != nil {
		return nil, "", err
	}

	// Apply boosts and optionally record
	return f.booster.RecordAndBoost(ctx, query, results, opts)
}

// Feedback records feedback for a previous interaction.
func (f *FeedbackAwareSearch) Feedback(ctx context.Context, item FeedbackItem) (*Feedback, error) {
	results, err := f.processor.ProcessBatch(ctx, []FeedbackItem{item})
	if err != nil {
		return nil, err
	}
	if len(results) > 0 {
		return results[0], nil
	}
	return nil, nil
}

// Booster returns the underlying booster for direct access.
func (f *FeedbackAwareSearch) Booster() *Booster {
	return f.booster
}

// Processor returns the underlying processor for direct access.
func (f *FeedbackAwareSearch) Processor() *Processor {
	return f.processor
}
