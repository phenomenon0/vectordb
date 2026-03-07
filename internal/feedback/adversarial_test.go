package feedback

import (
	"context"
	"math"
	"os"
	"strings"
	"sync"
	"testing"
	"time"
)

// ============================================================================
// ADVERSARIAL TESTS - Edge cases, race conditions, malformed inputs
// ============================================================================

// TestEmptyInputs tests handling of empty/nil inputs
func TestEmptyInputs(t *testing.T) {
	cfg := DefaultFeedbackConfig()
	cfg.StorePath = ""
	cfg.ConfidenceThreshold = 0.1

	store, err := NewStore(cfg)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}

	ctx := context.Background()

	t.Run("empty query interaction", func(t *testing.T) {
		interaction := NewInteraction("", []string{}, []float32{})
		err := store.RecordInteraction(ctx, interaction)
		if err != nil {
			t.Errorf("should accept empty query: %v", err)
		}
	})

	t.Run("nil result IDs", func(t *testing.T) {
		interaction := NewInteraction("test", nil, nil)
		err := store.RecordInteraction(ctx, interaction)
		if err != nil {
			t.Errorf("should accept nil slices: %v", err)
		}
	})

	t.Run("feedback for nonexistent interaction", func(t *testing.T) {
		fb := NewFeedback("nonexistent_interaction_id", FeedbackTypeThumbsUp)
		err := store.RecordFeedback(ctx, fb)
		if err == nil {
			t.Error("should reject feedback for nonexistent interaction")
		}
	})

	t.Run("get boost for empty target", func(t *testing.T) {
		boost := store.GetBoost("", "")
		if boost != 1.0 {
			t.Errorf("empty target should return neutral boost, got %f", boost)
		}
	})

	t.Run("get boosts with empty slice", func(t *testing.T) {
		boosts := store.GetBoosts([]string{}, "queryhash")
		if len(boosts) != 0 {
			t.Error("empty input should return empty map")
		}
	})

	t.Run("get boosts with nil slice", func(t *testing.T) {
		boosts := store.GetBoosts(nil, "queryhash")
		if boosts == nil {
			t.Error("should return empty map, not nil")
		}
	})
}

// TestConcurrentAccess tests race conditions with concurrent read/write
func TestConcurrentAccess(t *testing.T) {
	cfg := DefaultFeedbackConfig()
	cfg.StorePath = ""
	cfg.ConfidenceThreshold = 0.1

	store, err := NewStore(cfg)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}

	ctx := context.Background()

	// Create initial interaction
	interaction := NewInteraction("concurrent test", []string{"doc1", "doc2"}, []float32{0.9, 0.8})
	store.RecordInteraction(ctx, interaction)

	var wg sync.WaitGroup
	errors := make(chan error, 100)

	// Concurrent writers
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				fb := NewFeedback(interaction.ID, FeedbackTypeThumbsUp)
				fb.ClickedIDs = []string{"doc1"}
				if err := store.RecordFeedback(ctx, fb); err != nil {
					errors <- err
				}
			}
		}(i)
	}

	// Concurrent readers
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				_ = store.GetBoost("doc1", interaction.QueryHash)
				_ = store.GetBoosts([]string{"doc1", "doc2"}, interaction.QueryHash)
				_ = store.Stats()
			}
		}()
	}

	// Concurrent prune
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 10; i++ {
			store.Prune()
			time.Sleep(time.Millisecond)
		}
	}()

	wg.Wait()
	close(errors)

	var errCount int
	for err := range errors {
		t.Logf("concurrent error: %v", err)
		errCount++
	}

	if errCount > 0 {
		t.Errorf("got %d errors during concurrent access", errCount)
	}
}

// TestBoostOverflowUnderflow tests extreme boost values
func TestBoostOverflowUnderflow(t *testing.T) {
	cfg := DefaultFeedbackConfig()
	cfg.StorePath = ""
	cfg.ConfidenceThreshold = 0.0 // Always apply boost
	cfg.PositiveBoostDelta = 1.0  // Large positive delta
	cfg.NegativeBoostDelta = 1.0  // Large negative delta
	cfg.MaxBoost = 10.0
	cfg.MinBoost = 0.01

	store, err := NewStore(cfg)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}

	ctx := context.Background()

	t.Run("boost overflow clamped to max", func(t *testing.T) {
		interaction := NewInteraction("overflow test", []string{"doc1"}, []float32{0.9})
		store.RecordInteraction(ctx, interaction)

		// Apply many positive feedbacks
		for i := 0; i < 100; i++ {
			fb := NewFeedback(interaction.ID, FeedbackTypeThumbsUp)
			fb.ClickedIDs = []string{"doc1"}
			store.RecordFeedback(ctx, fb)
		}

		boost := store.GetBoost("doc1", interaction.QueryHash)
		if boost > cfg.MaxBoost {
			t.Errorf("boost %f exceeds max %f", boost, cfg.MaxBoost)
		}
		if boost < cfg.MaxBoost*0.9 {
			t.Errorf("boost %f should be near max %f", boost, cfg.MaxBoost)
		}
	})

	t.Run("boost underflow clamped to min", func(t *testing.T) {
		interaction := NewInteraction("underflow test", []string{"doc2"}, []float32{0.9})
		store.RecordInteraction(ctx, interaction)

		// Apply many negative feedbacks
		for i := 0; i < 100; i++ {
			fb := NewFeedback(interaction.ID, FeedbackTypeThumbsDown)
			fb.RejectedIDs = []string{"doc2"}
			store.RecordFeedback(ctx, fb)
		}

		boost := store.GetBoost("doc2", interaction.QueryHash)
		// Boost should be clamped at min, allowing small epsilon for float comparison
		if boost < cfg.MinBoost*0.99 {
			t.Errorf("boost %f significantly below min %f", boost, cfg.MinBoost)
		}
		// Should be close to min
		if boost > cfg.MinBoost*1.5 {
			t.Errorf("boost %f should be closer to min %f", boost, cfg.MinBoost)
		}
	})

	t.Run("NaN sentiment handled", func(t *testing.T) {
		interaction := NewInteraction("nan test", []string{"doc3"}, []float32{0.9})
		store.RecordInteraction(ctx, interaction)

		fb := NewFeedback(interaction.ID, FeedbackTypeNatural)
		fb.Sentiment = float32(math.NaN())
		fb.ClickedIDs = []string{"doc3"}

		// Should not panic
		err := store.RecordFeedback(ctx, fb)
		if err != nil {
			t.Logf("NaN sentiment error (acceptable): %v", err)
		}

		boost := store.GetBoost("doc3", interaction.QueryHash)
		if math.IsNaN(float64(boost)) {
			t.Error("boost should not be NaN")
		}
	})

	t.Run("Inf sentiment handled", func(t *testing.T) {
		interaction := NewInteraction("inf test", []string{"doc4"}, []float32{0.9})
		store.RecordInteraction(ctx, interaction)

		fb := NewFeedback(interaction.ID, FeedbackTypeNatural)
		fb.Sentiment = float32(math.Inf(1))
		fb.ClickedIDs = []string{"doc4"}

		// Should not panic
		err := store.RecordFeedback(ctx, fb)
		if err != nil {
			t.Logf("Inf sentiment error (acceptable): %v", err)
		}

		boost := store.GetBoost("doc4", interaction.QueryHash)
		if math.IsInf(float64(boost), 0) {
			t.Error("boost should not be Inf")
		}
	})
}

// TestMalformedWAL tests recovery from corrupted WAL
func TestMalformedWAL(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "feedback-wal-test-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Write corrupted WAL
	walPath := tmpDir + "/feedback.wal"
	corruptedData := []byte(`
{"type":"interaction","data":{"id":"test1","query":"valid"},"ts":1234567890}
not valid json at all
{"type":"feedback","data":{"id":"fb1","interaction_id":"test1"},"ts":1234567891}
{"type":"unknown","data":{},"ts":1234567892}
{"truncated json
`)
	os.WriteFile(walPath, corruptedData, 0644)

	cfg := DefaultFeedbackConfig()
	cfg.StorePath = tmpDir

	// Should not panic on corrupted WAL
	store, err := NewStore(cfg)
	if err != nil {
		t.Fatalf("should recover from corrupted WAL: %v", err)
	}
	defer store.Close()

	// Should have recovered valid entries
	stats := store.Stats()
	t.Logf("Recovered %d interactions, %d feedback from corrupted WAL",
		stats.TotalInteractions, stats.TotalFeedback)
}

// TestVeryLongStrings tests handling of extremely long inputs
func TestVeryLongStrings(t *testing.T) {
	cfg := DefaultFeedbackConfig()
	cfg.StorePath = ""

	store, err := NewStore(cfg)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}

	ctx := context.Background()

	t.Run("very long query", func(t *testing.T) {
		longQuery := strings.Repeat("a", 1_000_000) // 1MB query
		interaction := NewInteraction(longQuery, []string{"doc1"}, []float32{0.9})
		err := store.RecordInteraction(ctx, interaction)
		if err != nil {
			t.Errorf("should accept long query: %v", err)
		}
	})

	t.Run("many result IDs", func(t *testing.T) {
		ids := make([]string, 10000)
		scores := make([]float32, 10000)
		for i := range ids {
			ids[i] = strings.Repeat("x", 100)
			scores[i] = 0.5
		}
		interaction := NewInteraction("many results", ids, scores)
		err := store.RecordInteraction(ctx, interaction)
		if err != nil {
			t.Errorf("should accept many results: %v", err)
		}
	})

	t.Run("unicode and special characters", func(t *testing.T) {
		query := "测试 🚀 émojis \x00\x01\x02 nulls \"quotes\" 'apostrophes' <tags>"
		interaction := NewInteraction(query, []string{"doc1"}, []float32{0.9})
		err := store.RecordInteraction(ctx, interaction)
		if err != nil {
			t.Errorf("should accept unicode/special chars: %v", err)
		}

		// Verify query hash is consistent
		interaction2 := NewInteraction(query, []string{"doc1"}, []float32{0.9})
		if interaction.QueryHash != interaction2.QueryHash {
			t.Error("query hash should be deterministic")
		}
	})
}

// TestDecayEdgeCases tests boost decay with extreme time values
func TestDecayEdgeCases(t *testing.T) {
	t.Run("zero decay rate", func(t *testing.T) {
		boost := &BoostScore{
			Boost:      2.0,
			LastUpdate: time.Now().Add(-365 * 24 * time.Hour), // 1 year ago
			DecayRate:  0.0,
		}
		decayed := boost.DecayedBoost(time.Now())
		if decayed != 2.0 {
			t.Errorf("zero decay rate should not decay: got %f", decayed)
		}
	})

	t.Run("100% decay rate", func(t *testing.T) {
		boost := &BoostScore{
			Boost:      2.0,
			LastUpdate: time.Now().Add(-2 * 24 * time.Hour), // 2 days ago
			DecayRate:  1.0,                                 // 100% per day
		}
		decayed := boost.DecayedBoost(time.Now())
		if decayed != 1.0 {
			t.Errorf("full decay should return neutral: got %f", decayed)
		}
	})

	t.Run("future timestamp", func(t *testing.T) {
		boost := &BoostScore{
			Boost:      2.0,
			LastUpdate: time.Now().Add(24 * time.Hour), // 1 day in future
			DecayRate:  0.1,
		}
		decayed := boost.DecayedBoost(time.Now())
		// Negative elapsed time - decay formula may produce unexpected results
		t.Logf("Future timestamp decay: %f (may be > 2.0)", decayed)
	})

	t.Run("very old timestamp", func(t *testing.T) {
		boost := &BoostScore{
			Boost:      2.0,
			LastUpdate: time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC),
			DecayRate:  0.01,
		}
		decayed := boost.DecayedBoost(time.Now())
		if decayed != 1.0 {
			t.Errorf("very old boost should decay to neutral: got %f", decayed)
		}
	})
}

// TestProcessorEdgeCases tests sentiment processor with edge cases
func TestProcessorEdgeCases(t *testing.T) {
	cfg := DefaultFeedbackConfig()
	cfg.StorePath = ""
	cfg.EnableLLMSentiment = false // Use rule-based only

	store, err := NewStore(cfg)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}

	processor := NewProcessor(store, cfg)
	ctx := context.Background()

	// Create a base interaction for feedback
	interaction := NewInteraction("processor test", []string{"doc1"}, []float32{0.9})
	store.RecordInteraction(ctx, interaction)

	testCases := []struct {
		name   string
		text   string
		rating int
	}{
		{"empty text", "", 0},
		{"whitespace only", "   \t\n  ", 0},
		{"positive text negative rating", "this is great", -5},
		{"negative text positive rating", "this is terrible", 5},
		{"extreme positive rating", "", 1000000},
		{"extreme negative rating", "", -1000000},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Should not panic
			var fb *Feedback
			var err error

			if tc.text != "" {
				fb, err = processor.ProcessNaturalFeedback(ctx, interaction.ID, tc.text)
			} else {
				fb, err = processor.ProcessExplicitFeedback(ctx, interaction.ID, tc.rating, nil, nil)
			}

			if err != nil {
				t.Logf("processor error (may be acceptable): %v", err)
				return
			}

			if fb != nil {
				sentiment := fb.Sentiment
				if math.IsNaN(float64(sentiment)) || math.IsInf(float64(sentiment), 0) {
					t.Errorf("sentiment should be finite: got %f", sentiment)
				}
			}
		})
	}
}

// TestBoosterEdgeCases tests booster with edge cases
func TestBoosterEdgeCases(t *testing.T) {
	cfg := DefaultFeedbackConfig()
	cfg.StorePath = ""
	cfg.ConfidenceThreshold = 0.0

	store, err := NewStore(cfg)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}

	boosterCfg := DefaultBoosterConfig()
	booster := NewBooster(store, boosterCfg)

	ctx := context.Background()

	t.Run("empty results", func(t *testing.T) {
		boosted := booster.ApplyBoosts(ctx, "test", nil)
		if boosted == nil {
			t.Error("should return empty slice, not nil")
		}
	})

	t.Run("zero scores", func(t *testing.T) {
		results := []SearchResult{
			{ID: "doc1", Score: 0},
			{ID: "doc2", Score: 0},
		}
		boosted := booster.ApplyBoosts(ctx, "test", results)
		for _, b := range boosted {
			if math.IsNaN(float64(b.BoostedScore)) {
				t.Error("boosted score should not be NaN")
			}
		}
	})

	t.Run("negative scores", func(t *testing.T) {
		results := []SearchResult{
			{ID: "doc1", Score: -1.0},
			{ID: "doc2", Score: -0.5},
		}
		boosted := booster.ApplyBoosts(ctx, "test", results)
		// Negative scores * boost might produce unexpected ordering
		t.Logf("Negative score results: %+v", boosted)
	})
}

// TestStoreClose tests proper cleanup on close
func TestStoreClose(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "feedback-close-test-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	cfg := DefaultFeedbackConfig()
	cfg.StorePath = tmpDir

	store, err := NewStore(cfg)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}

	ctx := context.Background()

	// Add some data
	interaction := NewInteraction("close test", []string{"doc1"}, []float32{0.9})
	store.RecordInteraction(ctx, interaction)

	fb := NewFeedback(interaction.ID, FeedbackTypeThumbsUp)
	fb.ClickedIDs = []string{"doc1"}
	store.RecordFeedback(ctx, fb)

	// Close store
	err = store.Close()
	if err != nil {
		t.Errorf("close should not error: %v", err)
	}

	// Double close should not panic
	err = store.Close()
	// May or may not error, but should not panic

	// Verify data was persisted
	store2, err := NewStore(cfg)
	if err != nil {
		t.Fatalf("failed to reopen store: %v", err)
	}
	defer store2.Close()

	stats := store2.Stats()
	if stats.TotalInteractions == 0 {
		t.Error("data should be persisted after close")
	}
}
