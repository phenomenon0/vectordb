package extraction

import (
	"context"
)

// Extractor defines the interface for entity/relationship extraction.
type Extractor interface {
	// Extract extracts a knowledge graph from text content.
	Extract(ctx context.Context, content string) (*KnowledgeGraph, error)

	// ExtractTemporal extracts a knowledge graph with temporal events.
	ExtractTemporal(ctx context.Context, content string) (*TemporalKnowledgeGraph, error)

	// ExtractBatch extracts from multiple text chunks concurrently.
	ExtractBatch(ctx context.Context, chunks []string) ([]*KnowledgeGraph, error)

	// Provider returns the name of the LLM provider.
	Provider() string

	// Model returns the model being used.
	Model() string
}

// ExtractorConfig holds configuration for an extractor.
type ExtractorConfig struct {
	// Provider is the LLM provider (e.g., "ollama", "openai")
	Provider string `json:"provider"`

	// Model is the model name (e.g., "llama3.2", "gpt-4")
	Model string `json:"model"`

	// BaseURL is the API endpoint (e.g., "http://localhost:11434")
	BaseURL string `json:"base_url"`

	// APIKey for authenticated providers (optional for Ollama)
	APIKey string `json:"api_key,omitempty"`

	// Temperature for generation (0.0 - 1.0)
	Temperature float32 `json:"temperature"`

	// MaxTokens limits response length
	MaxTokens int `json:"max_tokens"`

	// Timeout in seconds for extraction
	TimeoutSecs int `json:"timeout_secs"`

	// CustomPrompt overrides the default extraction prompt
	CustomPrompt string `json:"custom_prompt,omitempty"`

	// TemporalPrompt for temporal extraction mode
	TemporalPrompt string `json:"temporal_prompt,omitempty"`

	// Concurrency for batch extraction
	Concurrency int `json:"concurrency"`

	// RetryAttempts for failed extractions
	RetryAttempts int `json:"retry_attempts"`

	// RetryDelayMs between retries
	RetryDelayMs int `json:"retry_delay_ms"`

	// GlyphMode enables Glyph format for extraction prompts (saves ~30% tokens)
	GlyphMode bool `json:"glyph_mode,omitempty"`
}

// DefaultConfig returns sensible defaults for Ollama.
func DefaultConfig() ExtractorConfig {
	return ExtractorConfig{
		Provider:      "ollama",
		Model:         "llama3.2",
		BaseURL:       "http://localhost:11434",
		Temperature:   0.1, // Low temp for consistent extraction
		MaxTokens:     4096,
		TimeoutSecs:   60,
		Concurrency:   4,
		RetryAttempts: 2,
		RetryDelayMs:  1000,
	}
}

// NewExtractor creates an extractor based on the configuration.
func NewExtractor(cfg ExtractorConfig) (Extractor, error) {
	switch cfg.Provider {
	case "ollama", "":
		return NewOllamaExtractor(cfg)
	case "openai":
		return NewOpenAIExtractor(cfg)
	default:
		return nil, ErrUnsupportedProvider
	}
}

// DefaultExtractionPrompt is the system prompt for entity extraction.
const DefaultExtractionPrompt = `You are an expert at extracting structured knowledge from text.
Extract entities (nodes) and relationships (edges) from the given text.

For each entity, provide:
- id: A unique lowercase identifier (use underscores for spaces)
- name: The display name
- type: One of: PERSON, ORGANIZATION, LOCATION, CONCEPT, TECHNOLOGY, EVENT, DATE, PRODUCT, CODE, FUNCTION, CLASS, MODULE
- description: A brief description (optional)

For each relationship, provide:
- source_node_id: The ID of the source entity
- target_node_id: The ID of the target entity
- relationship_name: One of: RELATED_TO, PART_OF, CREATED_BY, USED_BY, DEPENDS_ON, IMPLEMENTS, EXTENDS, CALLS, CONTAINS, LOCATED_IN, OCCURRED_AT, WORKS_FOR, COLLABORATES_WITH

Return ONLY valid JSON in this exact format:
{
  "nodes": [
    {"id": "entity_id", "name": "Entity Name", "type": "TYPE", "description": "Brief description"}
  ],
  "edges": [
    {"source_node_id": "source_id", "target_node_id": "target_id", "relationship_name": "RELATIONSHIP"}
  ]
}

Rules:
1. Extract ALL meaningful entities and relationships
2. Use consistent IDs across the graph
3. Prefer specific relationship types over RELATED_TO when possible
4. Include implicit relationships (e.g., if A contains B and B uses C, include A->B and B->C)
5. Do not include entities that are just values (numbers, URLs, etc.)
6. For code, extract functions, classes, modules, and their relationships`

// DefaultTemporalPrompt extends extraction with temporal events.
const DefaultTemporalPrompt = `You are an expert at extracting structured knowledge and temporal events from text.
Extract entities, relationships, AND time-based events.

For entities and relationships, follow the standard extraction rules.

Additionally, extract temporal events:
- id: Unique event identifier
- description: What happened
- date: ISO date if exact date is known (YYYY-MM-DD)
- year: Year as integer if only year is known
- date_text: Original text representation of the date/time
- entities: List of entity IDs involved in this event

Return ONLY valid JSON in this exact format:
{
  "nodes": [...],
  "edges": [...],
  "events": [
    {
      "id": "event_id",
      "description": "What happened",
      "year": 2024,
      "date_text": "in 2024",
      "entities": ["entity1_id", "entity2_id"]
    }
  ]
}

Rules:
1. Extract all events with temporal information (dates, years, time periods)
2. Connect events to relevant entities
3. Preserve original date text even if you extract structured date
4. Events can overlap with entities (e.g., "The 2020 Conference" is both an event and entity)`

// GlyphExtractionPrompt is the system prompt for Glyph-format entity extraction.
// Uses tabular Glyph format instead of JSON, saving ~30% tokens in prompt and response.
const GlyphExtractionPrompt = `You are an expert at extracting structured knowledge from text.
Extract entities (nodes) and relationships (edges) from the given text.

Return ONLY in this exact Glyph format:

@tab Node [id name type desc]
entity_id "Entity Name" TYPE "Brief description"
@end

@tab Edge [src tgt rel w]
source_id target_id RELATIONSHIP 1.0
@end

Entity types: PERSON ORGANIZATION LOCATION CONCEPT TECHNOLOGY EVENT DATE PRODUCT CODE FUNCTION CLASS MODULE
Relationships: RELATED_TO PART_OF CREATED_BY USED_BY DEPENDS_ON IMPLEMENTS EXTENDS CALLS CONTAINS LOCATED_IN OCCURRED_AT WORKS_FOR COLLABORATES_WITH

Rules:
1. Extract ALL meaningful entities and relationships
2. Use consistent IDs (lowercase, underscores for spaces)
3. Prefer specific relationship types over RELATED_TO
4. Quote strings containing spaces
5. Weight defaults to 1.0 if unsure`

// GlyphTemporalPrompt is the system prompt for Glyph-format temporal extraction.
const GlyphTemporalPrompt = `You are an expert at extracting structured knowledge and temporal events from text.

Return ONLY in this exact Glyph format:

@tab Node [id name type desc]
entity_id "Entity Name" TYPE "Brief description"
@end

@tab Edge [src tgt rel w]
source_id target_id RELATIONSHIP 1.0
@end

@tab Event [id desc year date_text entities]
event_id "What happened" 2024 "in 2024" [entity1 entity2]
@end

Rules:
1. Extract all entities, relationships, and temporal events
2. Connect events to relevant entities via the entities list
3. Use consistent lowercase IDs with underscores`
