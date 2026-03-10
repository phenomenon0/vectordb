package obsidian

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Logger is a minimal logging interface to avoid import cycles with the
// server's logging package.
type Logger interface {
	Info(msg string, args ...any)
	Warn(msg string, args ...any)
	Error(msg string, args ...any)
}

// SyncConfig holds all configuration for the background vault sync loop.
type SyncConfig struct {
	VaultPath   string          `json:"vault"`
	Collection  string          `json:"collection"`
	BatchSize   int             `json:"batch_size,omitempty"`
	Interval    time.Duration   `json:"-"`
	IntervalStr string          `json:"interval,omitempty"` // for JSON marshal
	StripFM     bool            `json:"strip_fm,omitempty"`
	Prune       bool            `json:"prune,omitempty"`
	ExcludeDirs map[string]bool `json:"-"`
	StateFile   string          `json:"-"`
	Enabled     bool            `json:"enabled"`
}

// DefaultConfig returns a SyncConfig with sensible defaults.
func DefaultConfig() SyncConfig {
	return SyncConfig{
		Collection:  "obsidian",
		BatchSize:   10,
		Interval:    2 * time.Minute,
		IntervalStr: "2m",
		StripFM:     false,
		Prune:       true,
		ExcludeDirs: DefaultExcludes(),
	}
}

// persistedConfig is the JSON shape stored in obsidian.json.
type persistedConfig struct {
	Vault      string `json:"vault"`
	Collection string `json:"collection"`
	Interval   string `json:"interval"`
	Enabled    bool   `json:"enabled"`
}

// LoadOrDetectConfig loads persisted obsidian config from dataDir/obsidian.json,
// falling back to defaults. Does NOT auto-enable.
func LoadOrDetectConfig(dataDir string) SyncConfig {
	cfg := DefaultConfig()

	configPath := filepath.Join(dataDir, "obsidian.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return cfg
	}

	var pc persistedConfig
	if err := json.Unmarshal(data, &pc); err != nil {
		return cfg
	}

	cfg.VaultPath = pc.Vault
	cfg.Enabled = pc.Enabled
	if pc.Collection != "" {
		cfg.Collection = pc.Collection
	}
	if pc.Interval != "" {
		if d, err := time.ParseDuration(pc.Interval); err == nil {
			cfg.Interval = d
			cfg.IntervalStr = pc.Interval
		}
	}

	return cfg
}

// SaveConfig persists the sync config to dataDir/obsidian.json.
func SaveConfig(dataDir string, cfg SyncConfig) error {
	pc := persistedConfig{
		Vault:      cfg.VaultPath,
		Collection: cfg.Collection,
		Interval:   cfg.Interval.String(),
		Enabled:    cfg.Enabled,
	}
	data, err := json.MarshalIndent(pc, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dataDir, "obsidian.json"), data, 0644)
}

// ApplyEnvOverrides reads OBSIDIAN_* env vars and applies them to cfg.
func ApplyEnvOverrides(cfg *SyncConfig) {
	if v := os.Getenv("OBSIDIAN_VAULT"); v != "" {
		cfg.VaultPath = v
		cfg.Enabled = true
	}
	if v := os.Getenv("OBSIDIAN_COLLECTION"); v != "" {
		cfg.Collection = v
	}
	if v := os.Getenv("OBSIDIAN_INTERVAL"); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			cfg.Interval = d
		}
	}
	if v := os.Getenv("OBSIDIAN_BATCH_SIZE"); v != "" {
		var n int
		if _, err := fmt.Sscanf(v, "%d", &n); err == nil && n > 0 {
			cfg.BatchSize = n
		}
	}
	if os.Getenv("OBSIDIAN_STRIP_FM") == "1" {
		cfg.StripFM = true
	}
	if os.Getenv("OBSIDIAN_PRUNE") == "0" {
		cfg.Prune = false
	}
}

// SyncLoop runs continuous vault sync. Blocks until ctx is cancelled.
//
// Callbacks avoid import cycles — the server wires store.Upsert, store.Delete,
// etc. into these function values.
func SyncLoop(
	ctx context.Context,
	cfg SyncConfig,
	log Logger,
	embedFn func(string) ([]float32, error),
	upsertFn func(vec []float32, doc, id string, meta map[string]string, collection string) error,
	deleteFn func(id string) error,
	iterFn func(collection string, fn func(id string, meta map[string]string) bool),
) {
	// Validate vault path
	info, err := os.Stat(cfg.VaultPath)
	if err != nil || !info.IsDir() {
		log.Error("obsidian sync: vault path invalid", "path", cfg.VaultPath, "error", err)
		return
	}

	vaultName := filepath.Base(cfg.VaultPath)
	source := "obsidian"
	if _, err := os.Stat(filepath.Join(cfg.VaultPath, ".obsidian")); err != nil {
		source = "markdown"
	}

	if cfg.ExcludeDirs == nil {
		cfg.ExcludeDirs = DefaultExcludes()
	}
	if cfg.StateFile == "" {
		cfg.StateFile = filepath.Join(cfg.VaultPath, ".deepdata-sync")
	}

	state := LoadSyncState(cfg.StateFile)
	prevMtimes := state.Mtimes

	// Initial sync
	prevMtimes = syncOnce(ctx, cfg, log, vaultName, source, prevMtimes, embedFn, upsertFn, deleteFn, iterFn)

	ticker := time.NewTicker(cfg.Interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			log.Info("obsidian sync stopped")
			return
		case <-ticker.C:
			prevMtimes = syncOnce(ctx, cfg, log, vaultName, source, prevMtimes, embedFn, upsertFn, deleteFn, iterFn)
		}
	}
}

// syncOnce performs a single sync cycle: collect, diff, embed, upsert, prune.
func syncOnce(
	ctx context.Context,
	cfg SyncConfig,
	log Logger,
	vaultName, source string,
	prevMtimes map[string]time.Time,
	embedFn func(string) ([]float32, error),
	upsertFn func(vec []float32, doc, id string, meta map[string]string, collection string) error,
	deleteFn func(id string) error,
	iterFn func(collection string, fn func(id string, meta map[string]string) bool),
) map[string]time.Time {
	// Check context before starting
	if ctx.Err() != nil {
		return prevMtimes
	}

	log.Info("obsidian sync: starting cycle", "vault", cfg.VaultPath)
	notes, err := CollectDirect(cfg.VaultPath, vaultName, source, cfg.ExcludeDirs, cfg.StripFM)
	if err != nil {
		log.Error("obsidian sync: collect failed", "error", err)
		return prevMtimes
	}
	log.Info("obsidian sync: collected notes", "count", len(notes))

	if len(notes) == 0 {
		log.Info("obsidian sync: no notes found")
		return prevMtimes
	}

	// Build current mtime map
	currentMtimes := make(map[string]time.Time, len(notes))
	for _, note := range notes {
		currentMtimes[note.ID] = note.ModTime
	}

	// Find changed/new notes
	changedSet := make(map[string]bool)
	for _, note := range notes {
		prev, existed := prevMtimes[note.ID]
		if !existed || !note.ModTime.Equal(prev) {
			changedSet[note.ID] = true
		}
	}

	log.Info("obsidian sync: diff computed", "changed", len(changedSet), "prev_tracked", len(prevMtimes))
	if len(changedSet) == 0 && !cfg.Prune {
		log.Info("obsidian sync: no changes detected, skipping")
		return currentMtimes
	}

	if len(changedSet) == 0 && cfg.Prune {
		// Only prune, no notes changed
		pruned := pruneOrphans(notes, cfg.Collection, deleteFn, iterFn)
		if pruned > 0 {
			log.Info("obsidian sync: pruned orphans", "count", pruned)
		}
		SaveSyncState(cfg.StateFile, SyncState{Mtimes: currentMtimes})
		return currentMtimes
	}

	// Compute backlinks on full set
	ComputeBacklinks(notes)

	// Pre-compute changed note names
	changedNames := make(map[string]bool, len(changedSet))
	for cid := range changedSet {
		changedNames[NoteName(cid)] = true
	}

	// Collect changed notes + their backlink-affected neighbors
	var toUpsert []Note
	for _, note := range notes {
		if changedSet[note.ID] {
			toUpsert = append(toUpsert, note)
			continue
		}
		// If any of this note's backlinks are from a changed note,
		// its backlink metadata needs re-upserting
		if bl := note.Meta["backlinks"]; bl != "" {
			for _, link := range strings.Split(bl, ",") {
				if changedNames[link] {
					toUpsert = append(toUpsert, note)
					break
				}
			}
		}
	}

	// Embed and upsert in batches
	upserted := 0
	embedErrors := 0
	failedNotes := make(map[string]bool)
	for i, note := range toUpsert {
		if ctx.Err() != nil {
			log.Warn("obsidian sync: cancelled during embed", "completed", upserted)
			return prevMtimes
		}

		// Inject absolute vault path for UI file-open actions
		note.Meta["vault_path"] = filepath.Join(cfg.VaultPath, note.ID)

		vec, err := embedFn(note.Content)
		if err != nil {
			failedNotes[note.ID] = true
			embedErrors++
			if embedErrors <= 3 {
				log.Warn("obsidian sync: embed failed", "note", note.ID, "error", err)
			}
			continue
		}

		if err := upsertFn(vec, note.Content, note.ID, note.Meta, cfg.Collection); err != nil {
			failedNotes[note.ID] = true
			log.Warn("obsidian sync: upsert failed", "note", note.ID, "error", err)
			continue
		}
		upserted++

		// Log progress every BatchSize notes
		if cfg.BatchSize > 0 && (i+1)%cfg.BatchSize == 0 {
			log.Info("obsidian sync: progress", "upserted", upserted, "total", len(toUpsert))
		}
	}

	// Prune orphans
	pruned := 0
	if cfg.Prune {
		pruned = pruneOrphans(notes, cfg.Collection, deleteFn, iterFn)
	}

	nextMtimes := make(map[string]time.Time, len(currentMtimes))
	for id, mtime := range currentMtimes {
		if failedNotes[id] {
			if prev, ok := prevMtimes[id]; ok {
				nextMtimes[id] = prev
			}
			continue
		}
		nextMtimes[id] = mtime
	}

	SaveSyncState(cfg.StateFile, SyncState{Mtimes: nextMtimes})

	log.Info("obsidian sync: complete",
		"changed", len(changedSet), "upserted", upserted,
		"pruned", pruned, "total_notes", len(notes))
	if embedErrors > 0 {
		log.Warn("obsidian sync: embed errors", "count", embedErrors)
	}

	return nextMtimes
}

// pruneOrphans deletes docs from the collection that no longer have a matching
// file in the vault. Returns the number of pruned docs.
func pruneOrphans(
	currentNotes []Note,
	collection string,
	deleteFn func(id string) error,
	iterFn func(collection string, fn func(id string, meta map[string]string) bool),
) int {
	currentIDs := make(map[string]bool, len(currentNotes))
	for _, note := range currentNotes {
		currentIDs[note.ID] = true
	}

	pruned := 0
	iterFn(collection, func(id string, meta map[string]string) bool {
		// Only prune docs from obsidian/markdown sources
		src := meta["source"]
		if src != "obsidian" && src != "markdown" {
			return true // continue
		}
		if !currentIDs[id] {
			if err := deleteFn(id); err == nil {
				pruned++
			}
		}
		return true // continue
	})

	return pruned
}
