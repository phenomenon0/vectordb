package main

import (
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

// ======================================================================================
// Cost Tracker - SQLite-based cost tracking for PRO mode
// ======================================================================================
// Tracks embedding costs, token usage, and provides analytics
// Only active in PRO mode (LOCAL mode has zero cost)
// ======================================================================================

// OpenAI pricing (as of Dec 2024)
const (
	// text-embedding-3-small: $0.02 per 1M tokens
	OpenAIPricePerToken = 0.00000002 // $0.02 / 1,000,000
)

// CostTracker tracks embedding costs using SQLite
type CostTracker struct {
	mu     sync.Mutex
	db     *sql.DB
	mode   VectorDBMode
	dbPath string

	// In-memory session stats (for fast access)
	sessionStart  time.Time
	sessionTokens int64
	sessionCost   float64
	sessionOps    int64
}

// CostStats holds aggregated cost statistics
type CostStats struct {
	// Session stats
	SessionTokens int64   `json:"session_tokens"`
	SessionCost   float64 `json:"session_cost_usd"`
	SessionOps    int64   `json:"session_operations"`
	SessionStart  string  `json:"session_start"`

	// Today stats
	TodayTokens int64   `json:"today_tokens"`
	TodayCost   float64 `json:"today_cost_usd"`
	TodayOps    int64   `json:"today_operations"`

	// All-time stats
	TotalTokens int64   `json:"total_tokens"`
	TotalCost   float64 `json:"total_cost_usd"`
	TotalOps    int64   `json:"total_operations"`

	// Mode info
	Mode       string  `json:"mode"`
	PricePerMT float64 `json:"price_per_million_tokens"`
}

// NewCostTracker creates a new cost tracker
// Returns nil if mode is LOCAL (no cost tracking needed)
func NewCostTracker(mode VectorDBMode) (*CostTracker, error) {
	// No cost tracking for LOCAL mode
	if mode == ModeLocal {
		return nil, nil
	}

	dbPath := GetCostDBPath(mode)

	// Ensure directory exists
	dir := filepath.Dir(dbPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create cost DB directory: %w", err)
	}

	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open cost database: %w", err)
	}

	tracker := &CostTracker{
		db:           db,
		mode:         mode,
		dbPath:       dbPath,
		sessionStart: time.Now(),
	}

	if err := tracker.initSchema(); err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to initialize schema: %w", err)
	}

	return tracker, nil
}

// initSchema creates the database schema
func (ct *CostTracker) initSchema() error {
	schema := `
	CREATE TABLE IF NOT EXISTS embeddings (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
		mode TEXT NOT NULL,
		model TEXT NOT NULL,
		tokens INTEGER NOT NULL,
		cost_usd REAL NOT NULL,
		operation TEXT NOT NULL,
		collection TEXT
	);

	CREATE TABLE IF NOT EXISTS daily_rollup (
		date DATE PRIMARY KEY,
		mode TEXT NOT NULL,
		total_tokens INTEGER DEFAULT 0,
		total_cost_usd REAL DEFAULT 0,
		embedding_count INTEGER DEFAULT 0
	);

	CREATE INDEX IF NOT EXISTS idx_embeddings_timestamp ON embeddings(timestamp);
	CREATE INDEX IF NOT EXISTS idx_embeddings_mode ON embeddings(mode);
	CREATE INDEX IF NOT EXISTS idx_embeddings_date ON embeddings(date(timestamp));
	`

	_, err := ct.db.Exec(schema)
	return err
}

// RecordEmbedding records an embedding operation and its cost
func (ct *CostTracker) RecordEmbedding(tokens int, model string, operation string) {
	if ct == nil || ct.db == nil {
		return
	}

	cost := float64(tokens) * OpenAIPricePerToken

	ct.mu.Lock()
	defer ct.mu.Unlock()

	// Update in-memory session stats
	ct.sessionTokens += int64(tokens)
	ct.sessionCost += cost
	ct.sessionOps++

	// Insert into database
	_, err := ct.db.Exec(`
		INSERT INTO embeddings (mode, model, tokens, cost_usd, operation)
		VALUES (?, ?, ?, ?, ?)
	`, string(ct.mode), model, tokens, cost, operation)

	if err != nil {
		fmt.Printf("Warning: failed to record embedding cost: %v\n", err)
	}

	// Update daily rollup (async-safe with UPSERT)
	today := time.Now().Format("2006-01-02")
	_, err = ct.db.Exec(`
		INSERT INTO daily_rollup (date, mode, total_tokens, total_cost_usd, embedding_count)
		VALUES (?, ?, ?, ?, 1)
		ON CONFLICT(date) DO UPDATE SET
			total_tokens = total_tokens + excluded.total_tokens,
			total_cost_usd = total_cost_usd + excluded.total_cost_usd,
			embedding_count = embedding_count + 1
	`, today, string(ct.mode), tokens, cost)

	if err != nil {
		fmt.Printf("Warning: failed to update daily rollup: %v\n", err)
	}
}

// RecordBatchEmbedding records a batch embedding operation
func (ct *CostTracker) RecordBatchEmbedding(tokens int, model string, count int) {
	if ct == nil {
		return
	}
	ct.RecordEmbedding(tokens, model, fmt.Sprintf("batch_%d", count))
}

// GetStats returns current cost statistics
func (ct *CostTracker) GetStats() CostStats {
	if ct == nil {
		return CostStats{
			Mode:       string(ModeLocal),
			PricePerMT: 0,
		}
	}

	ct.mu.Lock()
	sessionTokens := ct.sessionTokens
	sessionCost := ct.sessionCost
	sessionOps := ct.sessionOps
	sessionStart := ct.sessionStart
	ct.mu.Unlock()

	stats := CostStats{
		SessionTokens: sessionTokens,
		SessionCost:   sessionCost,
		SessionOps:    sessionOps,
		SessionStart:  sessionStart.Format(time.RFC3339),
		Mode:          string(ct.mode),
		PricePerMT:    OpenAIPricePerToken * 1_000_000,
	}

	// Get today's stats
	today := time.Now().Format("2006-01-02")
	row := ct.db.QueryRow(`
		SELECT COALESCE(SUM(tokens), 0), COALESCE(SUM(cost_usd), 0), COUNT(*)
		FROM embeddings
		WHERE date(timestamp) = ?
	`, today)
	row.Scan(&stats.TodayTokens, &stats.TodayCost, &stats.TodayOps)

	// Get all-time stats
	row = ct.db.QueryRow(`
		SELECT COALESCE(SUM(tokens), 0), COALESCE(SUM(cost_usd), 0), COUNT(*)
		FROM embeddings
	`)
	row.Scan(&stats.TotalTokens, &stats.TotalCost, &stats.TotalOps)

	return stats
}

// GetDailyStats returns stats for the last N days
func (ct *CostTracker) GetDailyStats(days int) ([]DailyStats, error) {
	if ct == nil {
		return nil, nil
	}

	rows, err := ct.db.Query(`
		SELECT date, total_tokens, total_cost_usd, embedding_count
		FROM daily_rollup
		WHERE date >= date('now', ?)
		ORDER BY date DESC
	`, fmt.Sprintf("-%d days", days))
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var stats []DailyStats
	for rows.Next() {
		var s DailyStats
		if err := rows.Scan(&s.Date, &s.Tokens, &s.Cost, &s.Operations); err != nil {
			continue
		}
		stats = append(stats, s)
	}
	return stats, nil
}

// DailyStats holds stats for a single day
type DailyStats struct {
	Date       string  `json:"date"`
	Tokens     int64   `json:"tokens"`
	Cost       float64 `json:"cost_usd"`
	Operations int64   `json:"operations"`
}

// EstimateCost estimates the cost for embedding a text
func (ct *CostTracker) EstimateCost(text string) float64 {
	if ct == nil || ct.mode == ModeLocal {
		return 0
	}
	// Rough estimate: ~4 chars per token for English
	tokens := len(text) / 4
	if tokens < 1 {
		tokens = 1
	}
	return float64(tokens) * OpenAIPricePerToken
}

// EstimateBatchCost estimates the cost for embedding multiple texts
func (ct *CostTracker) EstimateBatchCost(texts []string) float64 {
	if ct == nil || ct.mode == ModeLocal {
		return 0
	}
	totalChars := 0
	for _, t := range texts {
		totalChars += len(t)
	}
	tokens := totalChars / 4
	if tokens < 1 {
		tokens = 1
	}
	return float64(tokens) * OpenAIPricePerToken
}

// GetRecentOperations returns the most recent embedding operations
func (ct *CostTracker) GetRecentOperations(limit int) ([]EmbeddingRecord, error) {
	if ct == nil {
		return nil, nil
	}

	rows, err := ct.db.Query(`
		SELECT timestamp, model, tokens, cost_usd, operation
		FROM embeddings
		ORDER BY timestamp DESC
		LIMIT ?
	`, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var records []EmbeddingRecord
	for rows.Next() {
		var r EmbeddingRecord
		if err := rows.Scan(&r.Timestamp, &r.Model, &r.Tokens, &r.Cost, &r.Operation); err != nil {
			continue
		}
		records = append(records, r)
	}
	return records, nil
}

// EmbeddingRecord represents a single embedding operation
type EmbeddingRecord struct {
	Timestamp string  `json:"timestamp"`
	Model     string  `json:"model"`
	Tokens    int     `json:"tokens"`
	Cost      float64 `json:"cost_usd"`
	Operation string  `json:"operation"`
}

// Close closes the database connection
func (ct *CostTracker) Close() error {
	if ct == nil || ct.db == nil {
		return nil
	}
	return ct.db.Close()
}

// ExportCSV exports cost data to CSV format
func (ct *CostTracker) ExportCSV() (string, error) {
	if ct == nil {
		return "", nil
	}

	rows, err := ct.db.Query(`
		SELECT timestamp, mode, model, tokens, cost_usd, operation
		FROM embeddings
		ORDER BY timestamp DESC
	`)
	if err != nil {
		return "", err
	}
	defer rows.Close()

	csv := "timestamp,mode,model,tokens,cost_usd,operation\n"
	for rows.Next() {
		var timestamp, mode, model, operation string
		var tokens int
		var cost float64
		if err := rows.Scan(&timestamp, &mode, &model, &tokens, &cost, &operation); err != nil {
			continue
		}
		csv += fmt.Sprintf("%s,%s,%s,%d,%.8f,%s\n", timestamp, mode, model, tokens, cost, operation)
	}
	return csv, nil
}

// ResetSession resets session-level statistics
func (ct *CostTracker) ResetSession() {
	if ct == nil {
		return
	}
	ct.mu.Lock()
	defer ct.mu.Unlock()
	ct.sessionTokens = 0
	ct.sessionCost = 0
	ct.sessionOps = 0
	ct.sessionStart = time.Now()
}
