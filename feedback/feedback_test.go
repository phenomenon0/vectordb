package feedback

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestNewInteraction(t *testing.T) {
	query := "test query"
	resultIDs := []string{"doc1", "doc2", "doc3"}
	scores := []float32{0.9, 0.8, 0.7}

	interaction := NewInteraction(query, resultIDs, scores)

	if interaction.ID == "" {
		t.Error("interaction ID should not be empty")
	}

	if interaction.Query != query {
		t.Errorf("expected query '%s', got '%s'", query, interaction.Query)
	}

	if interaction.QueryHash == "" {
		t.Error("query hash should not be empty")
	}

	if len(interaction.ResultIDs) != 3 {
		t.Errorf("expected 3 result IDs, got %d", len(interaction.ResultIDs))
	}
}

func TestNewFeedback(t *testing.T) {
	fb := NewFeedback("interaction-123", FeedbackTypeThumbsUp)

	if fb.ID == "" {
		t.Error("feedback ID should not be empty")
	}

	if fb.InteractionID != "interaction-123" {
		t.Errorf("expected interaction ID 'interaction-123', got '%s'", fb.InteractionID)
	}

	if fb.Type != FeedbackTypeThumbsUp {
		t.Errorf("expected type %s, got %s", FeedbackTypeThumbsUp, fb.Type)
	}
}

func TestBoostScoreDecay(t *testing.T) {
	boost := &BoostScore{
		TargetID:   "doc1",
		Boost:      1.5,
		DecayRate:  0.1,                          // 10% per day
		LastUpdate: time.Now().AddDate(0, 0, -5), // 5 days ago
	}

	decayed := boost.DecayedBoost(time.Now())

	// After 5 days with 10% decay per day: 1 + (1.5-1)*(1-0.5) = 1.25
	expected := float32(1.25)
	if decayed < expected-0.01 || decayed > expected+0.01 {
		t.Errorf("expected decayed boost around %f, got %f", expected, decayed)
	}
}

func TestBoostScoreNoDecay(t *testing.T) {
	boost := &BoostScore{
		TargetID:   "doc1",
		Boost:      1.5,
		DecayRate:  0,                             // No decay
		LastUpdate: time.Now().AddDate(0, 0, -30), // 30 days ago
	}

	decayed := boost.DecayedBoost(time.Now())

	if decayed != 1.5 {
		t.Errorf("expected boost 1.5 with no decay, got %f", decayed)
	}
}

func TestFeedbackStore(t *testing.T) {
	cfg := DefaultFeedbackConfig()
	cfg.StorePath = "" // In-memory only

	store, err := NewStore(cfg)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}

	ctx := context.Background()

	// Record an interaction
	interaction := NewInteraction("test query", []string{"doc1", "doc2"}, []float32{0.9, 0.8})
	err = store.RecordInteraction(ctx, interaction)
	if err != nil {
		t.Fatalf("failed to record interaction: %v", err)
	}

	// Verify it was stored
	retrieved, ok := store.GetInteraction(interaction.ID)
	if !ok {
		t.Fatal("interaction not found")
	}
	if retrieved.Query != "test query" {
		t.Errorf("expected query 'test query', got '%s'", retrieved.Query)
	}

	// Record feedback
	fb := NewFeedback(interaction.ID, FeedbackTypeThumbsUp)
	fb.ClickedIDs = []string{"doc1"}
	err = store.RecordFeedback(ctx, fb)
	if err != nil {
		t.Fatalf("failed to record feedback: %v", err)
	}

	// Check stats
	stats := store.Stats()
	if stats.TotalInteractions != 1 {
		t.Errorf("expected 1 interaction, got %d", stats.TotalInteractions)
	}
	if stats.TotalFeedback != 1 {
		t.Errorf("expected 1 feedback, got %d", stats.TotalFeedback)
	}
}

func TestFeedbackStoreBoosts(t *testing.T) {
	cfg := DefaultFeedbackConfig()
	cfg.StorePath = ""
	cfg.ConfidenceThreshold = 0.1 // Lower threshold so single feedback is enough

	store, err := NewStore(cfg)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}

	ctx := context.Background()

	// Create interaction
	interaction := NewInteraction("test query", []string{"doc1", "doc2"}, []float32{0.9, 0.8})
	store.RecordInteraction(ctx, interaction)

	// Record positive feedback for doc1
	fb := NewFeedback(interaction.ID, FeedbackTypeThumbsUp)
	fb.ClickedIDs = []string{"doc1"}
	fb.Sentiment = 1.0
	store.RecordFeedback(ctx, fb)

	// Check boost was applied
	boost := store.GetBoost("doc1", interaction.QueryHash)
	if boost <= 1.0 {
		t.Errorf("expected positive boost for doc1, got %f", boost)
	}

	// Doc2 should not have a boost yet (wasn't clicked)
	boost2 := store.GetBoost("doc2", interaction.QueryHash)
	if boost2 != 1.0 {
		t.Errorf("expected neutral boost for doc2, got %f", boost2)
	}
}

func TestFeedbackStorePersistence(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "feedback-test-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	cfg := DefaultFeedbackConfig()
	cfg.StorePath = tmpDir

	ctx := context.Background()

	// Create store and add data
	store1, err := NewStore(cfg)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}

	interaction := NewInteraction("persistent query", []string{"doc1"}, []float32{0.9})
	store1.RecordInteraction(ctx, interaction)

	fb := NewFeedback(interaction.ID, FeedbackTypeClick)
	fb.ClickedIDs = []string{"doc1"}
	store1.RecordFeedback(ctx, fb)

	store1.Close()

	// Create new store and verify data was loaded
	store2, err := NewStore(cfg)
	if err != nil {
		t.Fatalf("failed to create second store: %v", err)
	}
	defer store2.Close()

	retrieved, ok := store2.GetInteraction(interaction.ID)
	if !ok {
		t.Fatal("interaction not found after reload")
	}
	if retrieved.Query != "persistent query" {
		t.Errorf("expected query 'persistent query', got '%s'", retrieved.Query)
	}

	// Check that snapshot file exists
	snapshotPath := filepath.Join(tmpDir, "feedback.json")
	if _, err := os.Stat(snapshotPath); os.IsNotExist(err) {
		t.Error("snapshot file should exist")
	}
}

func TestProcessor(t *testing.T) {
	cfg := DefaultFeedbackConfig()
	cfg.StorePath = ""
	cfg.EnableLLMSentiment = false // Use rule-based only

	store, err := NewStore(cfg)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}

	processor := NewProcessor(store, cfg)

	ctx := context.Background()

	// Create interaction first
	interaction := NewInteraction("test", []string{"doc1"}, []float32{0.9})
	store.RecordInteraction(ctx, interaction)

	// Test natural feedback processing
	fb, err := processor.ProcessNaturalFeedback(ctx, interaction.ID, "Great results, very helpful!")
	if err != nil {
		t.Fatalf("failed to process natural feedback: %v", err)
	}

	if fb.Sentiment <= 0 {
		t.Errorf("expected positive sentiment for positive text, got %f", fb.Sentiment)
	}

	// Test negative feedback
	interaction2 := NewInteraction("test2", []string{"doc2"}, []float32{0.8})
	store.RecordInteraction(ctx, interaction2)

	fb2, err := processor.ProcessNaturalFeedback(ctx, interaction2.ID, "Terrible, completely wrong results")
	if err != nil {
		t.Fatalf("failed to process natural feedback: %v", err)
	}

	if fb2.Sentiment >= 0 {
		t.Errorf("expected negative sentiment for negative text, got %f", fb2.Sentiment)
	}
}

func TestBooster(t *testing.T) {
	cfg := DefaultFeedbackConfig()
	cfg.StorePath = ""
	cfg.ConfidenceThreshold = 0.1 // Lower threshold so single feedback is enough

	store, err := NewStore(cfg)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}

	boosterCfg := DefaultBoosterConfig()
	booster := NewBooster(store, boosterCfg)

	ctx := context.Background()

	// Set up some boosts manually
	interaction := NewInteraction("boost test", []string{"doc1", "doc2", "doc3"}, []float32{0.9, 0.8, 0.7})
	store.RecordInteraction(ctx, interaction)

	// Positive feedback for doc1
	fb := NewFeedback(interaction.ID, FeedbackTypeThumbsUp)
	fb.ClickedIDs = []string{"doc1"}
	fb.Sentiment = 1.0
	store.RecordFeedback(ctx, fb)

	// Apply boosts
	results := []SearchResult{
		{ID: "doc3", Score: 0.95},
		{ID: "doc1", Score: 0.85},
		{ID: "doc2", Score: 0.75},
	}

	boosted := booster.ApplyBoosts(ctx, "boost test", results)

	// Doc1 should be boosted and potentially reorder
	if len(boosted) != 3 {
		t.Fatalf("expected 3 boosted results, got %d", len(boosted))
	}

	// Find doc1 and check its boost
	var doc1Boost float32
	for _, b := range boosted {
		if b.ID == "doc1" {
			doc1Boost = b.Boost
			break
		}
	}

	if doc1Boost <= 1.0 {
		t.Errorf("expected doc1 to have positive boost, got %f", doc1Boost)
	}
}

func TestBoosterRecordAndBoost(t *testing.T) {
	cfg := DefaultFeedbackConfig()
	cfg.StorePath = ""

	store, err := NewStore(cfg)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}

	boosterCfg := DefaultBoosterConfig()
	booster := NewBooster(store, boosterCfg)

	ctx := context.Background()

	results := []SearchResult{
		{ID: "doc1", Score: 0.9},
		{ID: "doc2", Score: 0.8},
	}

	opts := RecordOptions{
		SaveInteraction: true,
		UserID:          "user123",
		SessionID:       "session456",
	}

	boosted, interactionID, err := booster.RecordAndBoost(ctx, "combined test", results, opts)
	if err != nil {
		t.Fatalf("failed to record and boost: %v", err)
	}

	if interactionID == "" {
		t.Error("expected interaction ID when SaveInteraction is true")
	}

	if len(boosted) != 2 {
		t.Errorf("expected 2 boosted results, got %d", len(boosted))
	}

	// Verify interaction was recorded
	interaction, ok := store.GetInteraction(interactionID)
	if !ok {
		t.Error("interaction should be found in store")
	}

	if interaction.UserID != "user123" {
		t.Errorf("expected user ID 'user123', got '%s'", interaction.UserID)
	}
}

func TestFeedbackAwareSearch(t *testing.T) {
	cfg := DefaultFeedbackConfig()
	cfg.StorePath = ""

	store, err := NewStore(cfg)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}

	fas := NewFeedbackAwareSearch(store, cfg)

	ctx := context.Background()

	// Mock search function
	searchFn := func(ctx context.Context, query string) ([]SearchResult, error) {
		return []SearchResult{
			{ID: "result1", Score: 0.9, Doc: "Document 1"},
			{ID: "result2", Score: 0.8, Doc: "Document 2"},
		}, nil
	}

	opts := RecordOptions{SaveInteraction: true}

	boosted, interactionID, err := fas.Search(ctx, "test search", searchFn, opts)
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	if len(boosted) != 2 {
		t.Errorf("expected 2 results, got %d", len(boosted))
	}

	if interactionID == "" {
		t.Error("expected interaction ID")
	}

	// Submit feedback
	item := FeedbackItem{
		Type:          "natural",
		InteractionID: interactionID,
		Text:          "Very helpful!",
	}

	fb, err := fas.Feedback(ctx, item)
	if err != nil {
		t.Fatalf("feedback failed: %v", err)
	}

	if fb == nil {
		t.Error("expected feedback object")
	}
}

func TestEdgeBoosts(t *testing.T) {
	cfg := DefaultFeedbackConfig()
	cfg.StorePath = ""

	store, err := NewStore(cfg)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}

	boosterCfg := DefaultBoosterConfig()
	booster := NewBooster(store, boosterCfg)

	ctx := context.Background()

	// Create interactions with co-clicked results
	for i := 0; i < 5; i++ {
		interaction := NewInteraction("edge test", []string{"doc1", "doc2", "doc3"}, []float32{0.9, 0.8, 0.7})
		store.RecordInteraction(ctx, interaction)

		// Click doc1 and doc2 together
		fb := NewFeedback(interaction.ID, FeedbackTypeClick)
		fb.ClickedIDs = []string{"doc1", "doc2"}
		store.RecordFeedback(ctx, fb)
	}

	// Get edge boosts
	edgeBoosts := booster.GetEdgeBoosts()

	// Should have at least one edge boost between doc1 and doc2
	found := false
	for _, eb := range edgeBoosts {
		if (eb.SourceID == "doc1" && eb.TargetID == "doc2") ||
			(eb.SourceID == "doc2" && eb.TargetID == "doc1") {
			found = true
			if eb.Boost <= 1.0 {
				t.Errorf("expected positive edge boost, got %f", eb.Boost)
			}
			break
		}
	}

	if !found {
		t.Error("expected edge boost between doc1 and doc2")
	}
}

func TestPrune(t *testing.T) {
	cfg := DefaultFeedbackConfig()
	cfg.StorePath = ""
	cfg.RetentionDays = 30 // Use normal retention (not 0, as that prunes everything)
	cfg.MaxInteractions = 2

	store, err := NewStore(cfg)
	if err != nil {
		t.Fatalf("failed to create store: %v", err)
	}

	ctx := context.Background()

	// Add 5 interactions
	for i := 0; i < 5; i++ {
		interaction := NewInteraction("prune test", []string{"doc1"}, []float32{0.9})
		store.RecordInteraction(ctx, interaction)
	}

	// Prune should remove old interactions
	pruned := store.Prune()

	if pruned != 3 {
		t.Errorf("expected 3 pruned interactions, got %d", pruned)
	}

	stats := store.Stats()
	if stats.TotalInteractions != 2 {
		t.Errorf("expected 2 interactions after prune, got %d", stats.TotalInteractions)
	}
}

func BenchmarkRecordInteraction(b *testing.B) {
	cfg := DefaultFeedbackConfig()
	cfg.StorePath = ""

	store, err := NewStore(cfg)
	if err != nil {
		b.Fatalf("failed to create store: %v", err)
	}

	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		interaction := NewInteraction("benchmark query", []string{"doc1", "doc2"}, []float32{0.9, 0.8})
		store.RecordInteraction(ctx, interaction)
	}
}

func BenchmarkApplyBoosts(b *testing.B) {
	cfg := DefaultFeedbackConfig()
	cfg.StorePath = ""

	store, err := NewStore(cfg)
	if err != nil {
		b.Fatalf("failed to create store: %v", err)
	}

	boosterCfg := DefaultBoosterConfig()
	booster := NewBooster(store, boosterCfg)

	ctx := context.Background()

	// Create results
	results := make([]SearchResult, 100)
	for i := 0; i < 100; i++ {
		results[i] = SearchResult{ID: string(rune('a' + i%26)), Score: float32(100-i) / 100.0}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		booster.ApplyBoosts(ctx, "benchmark query", results)
	}
}
