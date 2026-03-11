package feedback

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

// Store persists and retrieves feedback data.
type Store struct {
	cfg FeedbackConfig

	mu           sync.RWMutex
	interactions map[string]*Interaction        // ID -> Interaction
	feedback     map[string][]*Feedback         // InteractionID -> Feedback[]
	boosts       map[string]*BoostScore         // TargetID|QueryHash -> BoostScore
	queryStats   map[string]*QueryFeedbackStats // QueryHash -> Stats

	// WAL for durability
	walPath string
	walFile *os.File
}

// NewStore creates a new feedback store.
func NewStore(cfg FeedbackConfig) (*Store, error) {
	s := &Store{
		cfg:          cfg,
		interactions: make(map[string]*Interaction),
		feedback:     make(map[string][]*Feedback),
		boosts:       make(map[string]*BoostScore),
		queryStats:   make(map[string]*QueryFeedbackStats),
	}

	if cfg.StorePath != "" {
		if err := os.MkdirAll(cfg.StorePath, 0755); err != nil {
			return nil, fmt.Errorf("create store directory: %w", err)
		}

		s.walPath = filepath.Join(cfg.StorePath, "feedback.wal")

		// Load existing data
		if err := s.load(); err != nil {
			// Log but continue - start fresh
			fmt.Fprintf(os.Stderr, "Warning: failed to load feedback data: %v\n", err)
		}

		// Open WAL for appending
		f, err := os.OpenFile(s.walPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
		if err != nil {
			return nil, fmt.Errorf("open WAL: %w", err)
		}
		s.walFile = f
	}

	return s, nil
}

// Close closes the store and flushes data.
func (s *Store) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.walFile != nil {
		s.walFile.Close()
	}

	return s.saveLocked()
}

// RecordInteraction stores a search interaction.
func (s *Store) RecordInteraction(ctx context.Context, interaction *Interaction) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Store interaction
	s.interactions[interaction.ID] = interaction

	// Update query stats
	stats, ok := s.queryStats[interaction.QueryHash]
	if !ok {
		stats = &QueryFeedbackStats{
			QueryHash:    interaction.QueryHash,
			QueryExample: interaction.Query,
		}
		s.queryStats[interaction.QueryHash] = stats
	}
	stats.TotalSearches++
	stats.LastSearch = interaction.Timestamp

	// Append to WAL
	return s.appendWAL("interaction", interaction)
}

// RecordFeedback stores feedback for an interaction.
func (s *Store) RecordFeedback(ctx context.Context, fb *Feedback) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Validate interaction exists
	interaction, ok := s.interactions[fb.InteractionID]
	if !ok {
		return fmt.Errorf("interaction %s not found", fb.InteractionID)
	}

	// Store feedback
	s.feedback[fb.InteractionID] = append(s.feedback[fb.InteractionID], fb)

	// Update boosts based on feedback
	s.updateBoostsLocked(interaction, fb)

	// Update query stats
	if stats, ok := s.queryStats[interaction.QueryHash]; ok {
		if fb.Sentiment > 0 || fb.Type == FeedbackTypeThumbsUp || fb.Type == FeedbackTypeClick {
			stats.PositiveFeedback++
		} else if fb.Sentiment < 0 || fb.Type == FeedbackTypeThumbsDown || fb.Type == FeedbackTypeRequery {
			stats.NegativeFeedback++
		}

		// Track clicked IDs (deduplicate, cap at 100)
		for _, clickedID := range fb.ClickedIDs {
			found := false
			for _, existing := range stats.TopClickedIDs {
				if existing == clickedID {
					found = true
					break
				}
			}
			if !found && len(stats.TopClickedIDs) < 100 {
				stats.TopClickedIDs = append(stats.TopClickedIDs, clickedID)
			}
		}
	}

	// Append to WAL
	return s.appendWAL("feedback", fb)
}

// updateBoostsLocked updates boost scores based on feedback.
func (s *Store) updateBoostsLocked(interaction *Interaction, fb *Feedback) {
	// Calculate the impact of this feedback
	var delta float32
	switch fb.Type {
	case FeedbackTypeThumbsUp, FeedbackTypeClick:
		delta = s.cfg.PositiveBoostDelta
	case FeedbackTypeThumbsDown, FeedbackTypeRequery:
		delta = -s.cfg.NegativeBoostDelta
	case FeedbackTypeDwell:
		delta = s.cfg.DwellPositiveWeight * s.cfg.PositiveBoostDelta
	case FeedbackTypeIgnore:
		delta = -s.cfg.IgnoreNegativeWeight * s.cfg.NegativeBoostDelta
	case FeedbackTypeNatural, FeedbackTypeExplicit:
		// Use sentiment directly, but guard against NaN/Inf
		sentiment := fb.Sentiment
		if isFinite(sentiment) {
			if sentiment > 0 {
				delta = sentiment * s.cfg.PositiveBoostDelta
			} else {
				delta = sentiment * s.cfg.NegativeBoostDelta // Note: sentiment is negative
			}
		}
	}

	// Guard against NaN/Inf delta and zero delta
	if delta == 0 || !isFinite(delta) {
		return
	}

	// Apply boost to clicked results (positive) or all results (if negative)
	var targetIDs []string
	if len(fb.ClickedIDs) > 0 && delta > 0 {
		targetIDs = fb.ClickedIDs
	} else if len(fb.RejectedIDs) > 0 && delta < 0 {
		targetIDs = fb.RejectedIDs
	} else {
		// Apply to all results
		targetIDs = interaction.ResultIDs
	}

	for _, targetID := range targetIDs {
		// Query-specific boost
		key := boostKey(targetID, interaction.QueryHash)
		boost := s.getOrCreateBoost(key, targetID, interaction.QueryHash)
		s.applyDelta(boost, delta)

		// Global boost (less weight)
		globalKey := boostKey(targetID, "")
		globalBoost := s.getOrCreateBoost(globalKey, targetID, "")
		s.applyDelta(globalBoost, delta*0.5) // Half weight for global
	}
}

// getOrCreateBoost gets or creates a boost score.
func (s *Store) getOrCreateBoost(key, targetID, queryHash string) *BoostScore {
	if boost, ok := s.boosts[key]; ok {
		return boost
	}

	boost := &BoostScore{
		TargetID:   targetID,
		TargetType: "document",
		QueryHash:  queryHash,
		Boost:      1.0, // Neutral
		Confidence: 0.0,
		Count:      0,
		LastUpdate: time.Now(),
		DecayRate:  s.cfg.DefaultDecayRate,
	}
	s.boosts[key] = boost
	return boost
}

// applyDelta applies a delta to a boost score.
func (s *Store) applyDelta(boost *BoostScore, delta float32) {
	boost.Boost += delta
	boost.Count++
	boost.LastUpdate = time.Now()

	// Update confidence based on count
	boost.Confidence = float32(boost.Count) / float32(boost.Count+5) // Asymptotic to 1

	// Clamp to bounds
	if boost.Boost < s.cfg.MinBoost {
		boost.Boost = s.cfg.MinBoost
	}
	if boost.Boost > s.cfg.MaxBoost {
		boost.Boost = s.cfg.MaxBoost
	}
}

// GetBoost returns the boost for a target, optionally query-specific.
func (s *Store) GetBoost(targetID, queryHash string) float32 {
	s.mu.RLock()
	defer s.mu.RUnlock()

	now := time.Now()

	// Try query-specific boost first
	if queryHash != "" {
		key := boostKey(targetID, queryHash)
		if boost, ok := s.boosts[key]; ok && boost.Confidence >= s.cfg.ConfidenceThreshold {
			return boost.DecayedBoost(now)
		}
	}

	// Fall back to global boost
	globalKey := boostKey(targetID, "")
	if boost, ok := s.boosts[globalKey]; ok && boost.Confidence >= s.cfg.ConfidenceThreshold {
		return boost.DecayedBoost(now)
	}

	return 1.0 // Neutral
}

// GetBoosts returns all boosts for re-ranking results.
func (s *Store) GetBoosts(targetIDs []string, queryHash string) map[string]float32 {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result := make(map[string]float32, len(targetIDs))
	now := time.Now()

	for _, targetID := range targetIDs {
		// Try query-specific first
		if queryHash != "" {
			key := boostKey(targetID, queryHash)
			if boost, ok := s.boosts[key]; ok && boost.Confidence >= s.cfg.ConfidenceThreshold {
				result[targetID] = boost.DecayedBoost(now)
				continue
			}
		}

		// Fall back to global
		globalKey := boostKey(targetID, "")
		if boost, ok := s.boosts[globalKey]; ok && boost.Confidence >= s.cfg.ConfidenceThreshold {
			result[targetID] = boost.DecayedBoost(now)
		} else {
			result[targetID] = 1.0
		}
	}

	return result
}

// GetInteraction retrieves an interaction by ID.
func (s *Store) GetInteraction(id string) (*Interaction, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	interaction, ok := s.interactions[id]
	return interaction, ok
}

// GetRecentInteractions returns the N most recent interactions.
func (s *Store) GetRecentInteractions(n int) []*Interaction {
	s.mu.RLock()
	defer s.mu.RUnlock()

	interactions := make([]*Interaction, 0, len(s.interactions))
	for _, i := range s.interactions {
		interactions = append(interactions, i)
	}

	// Sort by timestamp descending
	sort.Slice(interactions, func(i, j int) bool {
		return interactions[i].Timestamp.After(interactions[j].Timestamp)
	})

	if n > len(interactions) {
		n = len(interactions)
	}

	return interactions[:n]
}

// GetFeedbackForInteraction returns all feedback for an interaction.
func (s *Store) GetFeedbackForInteraction(interactionID string) []*Feedback {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.feedback[interactionID]
}

// GetQueryStats returns statistics for a query pattern.
func (s *Store) GetQueryStats(queryHash string) (*QueryFeedbackStats, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	stats, ok := s.queryStats[queryHash]
	return stats, ok
}

// Prune removes old interactions beyond retention period.
func (s *Store) Prune() int {
	s.mu.Lock()
	defer s.mu.Unlock()

	cutoff := time.Now().AddDate(0, 0, -s.cfg.RetentionDays)
	pruned := 0

	for id, interaction := range s.interactions {
		if interaction.Timestamp.Before(cutoff) {
			delete(s.interactions, id)
			delete(s.feedback, id)
			pruned++
		}
	}

	// Also prune if over max
	if len(s.interactions) > s.cfg.MaxInteractions {
		// Get oldest interactions
		interactions := make([]*Interaction, 0, len(s.interactions))
		for _, i := range s.interactions {
			interactions = append(interactions, i)
		}
		sort.Slice(interactions, func(i, j int) bool {
			return interactions[i].Timestamp.Before(interactions[j].Timestamp)
		})

		toRemove := len(s.interactions) - s.cfg.MaxInteractions
		for i := 0; i < toRemove; i++ {
			id := interactions[i].ID
			delete(s.interactions, id)
			delete(s.feedback, id)
			pruned++
		}
	}

	return pruned
}

// Stats returns store statistics.
func (s *Store) Stats() StoreStats {
	s.mu.RLock()
	defer s.mu.RUnlock()

	totalFeedback := 0
	for _, fb := range s.feedback {
		totalFeedback += len(fb)
	}

	return StoreStats{
		TotalInteractions:  len(s.interactions),
		TotalFeedback:      totalFeedback,
		TotalBoosts:        len(s.boosts),
		TotalQueryPatterns: len(s.queryStats),
	}
}

// StoreStats contains store statistics.
type StoreStats struct {
	TotalInteractions  int `json:"total_interactions"`
	TotalFeedback      int `json:"total_feedback"`
	TotalBoosts        int `json:"total_boosts"`
	TotalQueryPatterns int `json:"total_query_patterns"`
}

// boostKey creates a key for the boost map.
func boostKey(targetID, queryHash string) string {
	if queryHash == "" {
		return targetID + "|global"
	}
	return targetID + "|" + queryHash
}

// appendWAL appends an entry to the WAL.
func (s *Store) appendWAL(entryType string, data any) error {
	if s.walFile == nil {
		return nil
	}

	entry := map[string]any{
		"type": entryType,
		"data": data,
		"ts":   time.Now().UnixNano(),
	}

	bytes, err := json.Marshal(entry)
	if err != nil {
		return err
	}

	bytes = append(bytes, '\n')
	_, err = s.walFile.Write(bytes)
	return err
}

// load loads data from the store path.
func (s *Store) load() error {
	if s.cfg.StorePath == "" {
		return nil
	}

	// Load snapshot if exists
	snapshotPath := filepath.Join(s.cfg.StorePath, "feedback.json")
	data, err := os.ReadFile(snapshotPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // No snapshot yet
		}
		return err
	}

	var snapshot struct {
		Interactions map[string]*Interaction        `json:"interactions"`
		Feedback     map[string][]*Feedback         `json:"feedback"`
		Boosts       map[string]*BoostScore         `json:"boosts"`
		QueryStats   map[string]*QueryFeedbackStats `json:"query_stats"`
	}

	if err := json.Unmarshal(data, &snapshot); err != nil {
		return err
	}

	if snapshot.Interactions != nil {
		s.interactions = snapshot.Interactions
	}
	if snapshot.Feedback != nil {
		s.feedback = snapshot.Feedback
	}
	if snapshot.Boosts != nil {
		s.boosts = snapshot.Boosts
	}
	if snapshot.QueryStats != nil {
		s.queryStats = snapshot.QueryStats
	}

	// Replay WAL if exists
	return s.replayWAL()
}

// replayWAL replays the WAL to catch up.
func (s *Store) replayWAL() error {
	if s.walPath == "" {
		return nil
	}

	data, err := os.ReadFile(s.walPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}

	// Process line by line
	for _, line := range splitLines(data) {
		if len(line) == 0 {
			continue
		}

		var entry struct {
			Type string          `json:"type"`
			Data json.RawMessage `json:"data"`
		}

		if err := json.Unmarshal(line, &entry); err != nil {
			continue // Skip malformed entries
		}

		switch entry.Type {
		case "interaction":
			var interaction Interaction
			if err := json.Unmarshal(entry.Data, &interaction); err == nil {
				s.interactions[interaction.ID] = &interaction
			}
		case "feedback":
			var fb Feedback
			if err := json.Unmarshal(entry.Data, &fb); err == nil {
				s.feedback[fb.InteractionID] = append(s.feedback[fb.InteractionID], &fb)
			}
		}
	}

	return nil
}

// saveLocked saves a snapshot (caller must hold lock).
func (s *Store) saveLocked() error {
	if s.cfg.StorePath == "" {
		return nil
	}

	snapshot := struct {
		Interactions map[string]*Interaction        `json:"interactions"`
		Feedback     map[string][]*Feedback         `json:"feedback"`
		Boosts       map[string]*BoostScore         `json:"boosts"`
		QueryStats   map[string]*QueryFeedbackStats `json:"query_stats"`
	}{
		Interactions: s.interactions,
		Feedback:     s.feedback,
		Boosts:       s.boosts,
		QueryStats:   s.queryStats,
	}

	data, err := json.MarshalIndent(snapshot, "", "  ")
	if err != nil {
		return err
	}

	snapshotPath := filepath.Join(s.cfg.StorePath, "feedback.json")
	tmpPath := snapshotPath + ".tmp"

	if err := os.WriteFile(tmpPath, data, 0644); err != nil {
		return err
	}

	if err := os.Rename(tmpPath, snapshotPath); err != nil {
		return err
	}

	// Truncate WAL after successful snapshot
	if s.walFile != nil {
		s.walFile.Close()
		os.Truncate(s.walPath, 0)
		f, err := os.OpenFile(s.walPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
		if err == nil {
			s.walFile = f
		}
	}

	return nil
}

// splitLines splits bytes into lines.
func splitLines(data []byte) [][]byte {
	var lines [][]byte
	start := 0
	for i, b := range data {
		if b == '\n' {
			if i > start {
				lines = append(lines, data[start:i])
			}
			start = i + 1
		}
	}
	if start < len(data) {
		lines = append(lines, data[start:])
	}
	return lines
}

// isFinite returns true if the float32 is not NaN or Inf.
func isFinite(f float32) bool {
	return f == f && f != f+1 // NaN != NaN, and Inf == Inf+1
}
