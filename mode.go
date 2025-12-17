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
// Two deployment profiles:
// - LOCAL: ONNX BGE-small embeddings, 384d, FREE, offline
// - PRO: OpenAI text-embedding-3-small, 1536d, ~$0.02/1M tokens
// ======================================================================================

// VectorDBMode represents the operational mode of the server
type VectorDBMode string

const (
	ModeLocal VectorDBMode = "local" // ONNX BGE-small, 384d, FREE
	ModePro   VectorDBMode = "pro"   // OpenAI, 1536d, paid
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
// Note: LOCAL mode dimension is set dynamically based on available embedder
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
	ModePro: {
		Mode:           ModePro,
		Dimension:      1536,
		EmbedderType:   "openai",
		EmbedderModel:  "text-embedding-3-small",
		CostPer1MToken: 0.02, // $0.02 per 1M tokens
		DataDirectory:  "pro",
		Description:    "OpenAI embeddings (text-embedding-3-small, 1536d) - Best quality",
	},
}

// CurrentMode holds the active mode configuration
var CurrentMode *ModeConfig

// LoadModeFromEnv loads the mode configuration from environment variables
// Defaults to LOCAL mode if not specified
func LoadModeFromEnv() (*ModeConfig, error) {
	modeStr := strings.ToLower(os.Getenv("VECTORDB_MODE"))
	
	// Default to LOCAL mode (safe, free)
	if modeStr == "" {
		modeStr = string(ModeLocal)
	}

	mode := VectorDBMode(modeStr)
	config, exists := ModeConfigs[mode]
	if !exists {
		return nil, fmt.Errorf("unknown mode: %s (valid: local, pro)", modeStr)
	}

	// PRO mode requires OpenAI API key
	if mode == ModePro {
		if os.Getenv("OPENAI_API_KEY") == "" {
			return nil, fmt.Errorf("PRO mode requires OPENAI_API_KEY environment variable")
		}
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

// GetWALPath returns the path to the WAL file for the current mode
func GetWALPath(mode VectorDBMode) string {
	return filepath.Join(GetDataDirectory(mode), "index.gob.wal")
}

// GetCostDBPath returns the path to the SQLite cost tracking database
func GetCostDBPath(mode VectorDBMode) string {
	return filepath.Join(GetDataDirectory(mode), "costs.db")
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

// GetModeInfo returns mode information for API responses
func GetModeInfo(config *ModeConfig) ModeInfo {
	return ModeInfo{
		Mode:           string(config.Mode),
		Dimension:      config.Dimension,
		EmbedderType:   config.EmbedderType,
		EmbedderModel:  config.EmbedderModel,
		CostPer1MToken: config.CostPer1MToken,
		Description:    config.Description,
		DataDirectory:  GetDataDirectory(config.Mode),
		IsPro:          config.Mode == ModePro,
		IsFree:         config.CostPer1MToken == 0,
	}
}

// PrintModeBanner prints a startup banner showing the current mode
func PrintModeBanner(config *ModeConfig) {
	var modeIcon, modeColor string
	if config.Mode == ModeLocal {
		modeIcon = "🏠"
		modeColor = "\033[32m" // Green
	} else {
		modeIcon = "⚡"
		modeColor = "\033[33m" // Yellow/Gold
	}
	reset := "\033[0m"

	fmt.Println()
	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Printf("║  %s%s VectorDB %s Mode%s                                            ║\n", 
		modeColor, modeIcon, strings.ToUpper(string(config.Mode)), reset)
	fmt.Println("╠════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║  Embedder:  %-50s ║\n", config.EmbedderType+"/"+config.EmbedderModel)
	fmt.Printf("║  Dimension: %-50d ║\n", config.Dimension)
	if config.CostPer1MToken > 0 {
		fmt.Printf("║  Cost:      $%.2f per 1M tokens                               ║\n", config.CostPer1MToken)
	} else {
		fmt.Printf("║  Cost:      %-50s ║\n", "FREE")
	}
	fmt.Printf("║  Data:      %-50s ║\n", GetDataDirectory(config.Mode))
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")
	fmt.Println()
}

// ValidateDimension checks if a vector dimension matches the current mode
func ValidateDimension(dim int) error {
	if CurrentMode == nil {
		return fmt.Errorf("mode not initialized")
	}
	if dim != CurrentMode.Dimension {
		return fmt.Errorf("dimension mismatch: expected %d (%s mode), got %d", 
			CurrentMode.Dimension, CurrentMode.Mode, dim)
	}
	return nil
}
