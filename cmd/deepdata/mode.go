package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ======================================================================================
// VectorDB Mode System
// ======================================================================================
// Historical compatibility profiles. The canonical RC forces local mode and
// accepts caller-supplied vectors; server-managed embedding modes are outside
// the supported production surface.
// ======================================================================================

// VectorDBMode represents the operational mode of the server
type VectorDBMode string

const (
	ModeLocal VectorDBMode = "local" // ONNX BGE-small, 384d, FREE
)

// ModeConfig contains all configuration for a specific mode
type ModeConfig struct {
	Mode           VectorDBMode
	Dimension      int
	EmbedderType   string  // "onnx", "openai", "ollama", "hash"
	EmbedderModel  string  // Model name/path
	CostPer1MToken float64 // USD per 1M tokens (0 for local)
	DataDirectory  string  // Where to store data
	Description    string  // Human-readable description
}

// Predefined mode configurations
// LOCAL mode dimension is set dynamically based on available embedder:
// - ONNX (bge-small): 384d
// - Ollama (nomic-embed-text): 768d
// - Hash fallback: configurable (default 384)
var ModeConfigs = map[VectorDBMode]ModeConfig{
	ModeLocal: {
		Mode:           ModeLocal,
		Dimension:      768, // Default to Ollama dimension (most common fallback)
		EmbedderType:   "ollama",
		EmbedderModel:  "nomic-embed-text",
		CostPer1MToken: 0.0,
		DataDirectory:  "local",
		Description:    "Local embeddings (Ollama/ONNX) - FREE, offline",
	},
}

// CurrentMode holds the active mode configuration
var CurrentMode *ModeConfig

// LoadModeFromEnv loads the mode configuration from environment variables
// Defaults to LOCAL mode
func LoadModeFromEnv() (*ModeConfig, error) {
	modeStr := strings.ToLower(os.Getenv("VECTORDB_MODE"))

	// Default to LOCAL mode (only supported mode)
	if modeStr == "" {
		modeStr = string(ModeLocal)
	}

	mode := VectorDBMode(modeStr)
	config, exists := ModeConfigs[mode]
	if !exists {
		return nil, fmt.Errorf("unknown mode: %s (valid: local)", modeStr)
	}

	// Allow dimension override (advanced use)
	if dimStr := os.Getenv("EMBED_DIM"); dimStr != "" {
		var dim int
		if _, err := fmt.Sscanf(dimStr, "%d", &dim); err == nil && dim > 0 {
			config.Dimension = dim
		}
	}

	// Allow custom data directory
	if dataDir := os.Getenv("VECTORDB_DATA_DIR"); dataDir != "" {
		config.DataDirectory = dataDir
	}

	CurrentMode = &config
	return &config, nil
}

// GetDataDirectory returns the full path to the data directory for the current mode
func GetDataDirectory(mode VectorDBMode) string {
	baseDir := os.Getenv("VECTORDB_BASE_DIR")
	if baseDir == "" {
		// Default: ~/.vectordb/
		home, err := os.UserHomeDir()
		if err != nil {
			home = "."
		}
		baseDir = filepath.Join(home, ".vectordb")
	}
	baseDir = filepath.Clean(baseDir)

	// VECTORDB_DATA_DIR is the exact primary state directory. Relative values
	// are resolved below VECTORDB_BASE_DIR; absolute values are used directly.
	// This makes --data-dir and container volume configuration authoritative.
	if dataDir := strings.TrimSpace(os.Getenv("VECTORDB_DATA_DIR")); dataDir != "" {
		if filepath.IsAbs(dataDir) {
			return filepath.Clean(dataDir)
		}
		return filepath.Join(baseDir, dataDir)
	}

	config, exists := ModeConfigs[mode]
	if !exists {
		return filepath.Join(baseDir, string(mode))
	}

	return filepath.Join(baseDir, config.DataDirectory)
}

// EnsureDataDirectory creates the data directory if it doesn't exist
func EnsureDataDirectory(mode VectorDBMode) (string, error) {
	dir := GetDataDirectory(mode)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", fmt.Errorf("failed to create data directory %s: %w", dir, err)
	}
	return dir, nil
}

// GetIndexPath returns the path to the index file for the current mode
func GetIndexPath(mode VectorDBMode) string {
	return filepath.Join(GetDataDirectory(mode), "index.gob")
}

// ModeInfo returns a struct suitable for JSON serialization in API responses
type ModeInfo struct {
	Mode           string  `json:"mode"`
	Dimension      int     `json:"dimension"`
	EmbedderType   string  `json:"embedder_type"`
	EmbedderModel  string  `json:"embedder_model"`
	CostPer1MToken float64 `json:"cost_per_1m_tokens"`
	Description    string  `json:"description"`
	DataDirectory  string  `json:"data_directory"`
	IsPro          bool    `json:"is_pro"`
	IsFree         bool    `json:"is_free"`
}
