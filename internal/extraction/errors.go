package extraction

import "errors"

// Schema validation errors
var (
	ErrEmptyNodeID       = errors.New("node ID cannot be empty")
	ErrEmptyNodeName     = errors.New("node name cannot be empty")
	ErrEmptyNodeType     = errors.New("node type cannot be empty")
	ErrEmptySourceNode   = errors.New("edge source node ID cannot be empty")
	ErrEmptyTargetNode   = errors.New("edge target node ID cannot be empty")
	ErrEmptyRelationship = errors.New("edge relationship name cannot be empty")
	ErrMissingSourceNode = errors.New("edge references non-existent source node")
	ErrMissingTargetNode = errors.New("edge references non-existent target node")
)

// Extraction errors
var (
	ErrEmptyContent        = errors.New("content cannot be empty")
	ErrExtractionFailed    = errors.New("extraction failed")
	ErrInvalidJSON         = errors.New("invalid JSON response from LLM")
	ErrNoExtractorConfig   = errors.New("no extractor configured")
	ErrUnsupportedProvider = errors.New("unsupported LLM provider")
	ErrTimeout             = errors.New("extraction timed out")
	ErrRateLimited         = errors.New("rate limited by LLM provider")
)

// Configuration errors
var (
	ErrInvalidChunkSize   = errors.New("chunk size must be positive")
	ErrInvalidOverlap     = errors.New("overlap must be less than chunk size")
	ErrInvalidConcurrency = errors.New("concurrency must be positive")
	ErrMissingOllamaURL   = errors.New("Ollama URL not configured")
	ErrMissingModel       = errors.New("model name not configured")
)
