package extraction

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

// OllamaExtractor implements entity extraction using Ollama.
type OllamaExtractor struct {
	cfg    ExtractorConfig
	client *http.Client
}

// OllamaRequest is the request format for Ollama API.
type OllamaRequest struct {
	Model   string         `json:"model"`
	Prompt  string         `json:"prompt"`
	System  string         `json:"system,omitempty"`
	Stream  bool           `json:"stream"`
	Format  string         `json:"format,omitempty"`
	Options *OllamaOptions `json:"options,omitempty"`
}

// OllamaOptions contains generation options.
type OllamaOptions struct {
	Temperature float32 `json:"temperature,omitempty"`
	NumPredict  int     `json:"num_predict,omitempty"`
}

// OllamaResponse is the response format from Ollama API.
type OllamaResponse struct {
	Model              string `json:"model"`
	CreatedAt          string `json:"created_at"`
	Response           string `json:"response"`
	Done               bool   `json:"done"`
	TotalDuration      int64  `json:"total_duration,omitempty"`
	LoadDuration       int64  `json:"load_duration,omitempty"`
	PromptEvalCount    int    `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int64  `json:"prompt_eval_duration,omitempty"`
	EvalCount          int    `json:"eval_count,omitempty"`
	EvalDuration       int64  `json:"eval_duration,omitempty"`
}

// NewOllamaExtractor creates a new Ollama-based extractor.
func NewOllamaExtractor(cfg ExtractorConfig) (*OllamaExtractor, error) {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "http://localhost:11434"
	}
	if cfg.Model == "" {
		cfg.Model = "llama3.2"
	}
	if cfg.TimeoutSecs <= 0 {
		cfg.TimeoutSecs = 60
	}
	if cfg.Concurrency <= 0 {
		cfg.Concurrency = 4
	}
	if cfg.Temperature <= 0 {
		cfg.Temperature = 0.1
	}
	if cfg.MaxTokens <= 0 {
		cfg.MaxTokens = 4096
	}

	return &OllamaExtractor{
		cfg: cfg,
		client: &http.Client{
			Timeout: time.Duration(cfg.TimeoutSecs) * time.Second,
		},
	}, nil
}

// Provider returns "ollama".
func (e *OllamaExtractor) Provider() string {
	return "ollama"
}

// Model returns the configured model name.
func (e *OllamaExtractor) Model() string {
	return e.cfg.Model
}

// Extract extracts a knowledge graph from text content.
func (e *OllamaExtractor) Extract(ctx context.Context, content string) (*KnowledgeGraph, error) {
	if content == "" {
		return nil, ErrEmptyContent
	}

	prompt := e.cfg.CustomPrompt
	if prompt == "" {
		prompt = DefaultExtractionPrompt
	}

	response, err := e.callOllama(ctx, prompt, content)
	if err != nil {
		return nil, err
	}

	return e.parseKnowledgeGraph(response)
}

// ExtractTemporal extracts a knowledge graph with temporal events.
func (e *OllamaExtractor) ExtractTemporal(ctx context.Context, content string) (*TemporalKnowledgeGraph, error) {
	if content == "" {
		return nil, ErrEmptyContent
	}

	prompt := e.cfg.TemporalPrompt
	if prompt == "" {
		prompt = DefaultTemporalPrompt
	}

	response, err := e.callOllama(ctx, prompt, content)
	if err != nil {
		return nil, err
	}

	return e.parseTemporalKnowledgeGraph(response)
}

// ExtractBatch extracts from multiple text chunks concurrently.
func (e *OllamaExtractor) ExtractBatch(ctx context.Context, chunks []string) ([]*KnowledgeGraph, error) {
	results := make([]*KnowledgeGraph, len(chunks))
	errors := make([]error, len(chunks))

	// Use semaphore for concurrency control
	sem := make(chan struct{}, e.cfg.Concurrency)
	var wg sync.WaitGroup

	for i, chunk := range chunks {
		wg.Add(1)
		go func(idx int, text string) {
			defer wg.Done()

			// Acquire semaphore
			select {
			case sem <- struct{}{}:
				defer func() { <-sem }()
			case <-ctx.Done():
				errors[idx] = ctx.Err()
				return
			}

			// Extract with retries
			var lastErr error
			for attempt := 0; attempt <= e.cfg.RetryAttempts; attempt++ {
				if attempt > 0 {
					time.Sleep(time.Duration(e.cfg.RetryDelayMs) * time.Millisecond)
				}

				kg, err := e.Extract(ctx, text)
				if err == nil {
					results[idx] = kg
					return
				}
				lastErr = err
			}
			errors[idx] = lastErr
		}(i, chunk)
	}

	wg.Wait()

	// Check for any errors
	for _, err := range errors {
		if err != nil {
			return results, fmt.Errorf("batch extraction had failures: %w", err)
		}
	}

	return results, nil
}

// callOllama sends a request to the Ollama API.
func (e *OllamaExtractor) callOllama(ctx context.Context, systemPrompt, userContent string) (string, error) {
	url := strings.TrimSuffix(e.cfg.BaseURL, "/") + "/api/generate"

	req := OllamaRequest{
		Model:  e.cfg.Model,
		System: systemPrompt,
		Prompt: userContent,
		Stream: false,
		Format: "json",
		Options: &OllamaOptions{
			Temperature: e.cfg.Temperature,
			NumPredict:  e.cfg.MaxTokens,
		},
	}

	body, err := json.Marshal(req)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := e.client.Do(httpReq)
	if err != nil {
		if ctx.Err() != nil {
			return "", ErrTimeout
		}
		return "", fmt.Errorf("http request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == 429 {
		return "", ErrRateLimited
	}
	if resp.StatusCode != 200 {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("ollama returned %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var ollamaResp OllamaResponse
	if err := json.NewDecoder(resp.Body).Decode(&ollamaResp); err != nil {
		return "", fmt.Errorf("decode response: %w", err)
	}

	return ollamaResp.Response, nil
}

// parseKnowledgeGraph parses a KnowledgeGraph from JSON response.
func (e *OllamaExtractor) parseKnowledgeGraph(response string) (*KnowledgeGraph, error) {
	// Clean up response - remove markdown code blocks if present
	response = cleanJSONResponse(response)

	var kg KnowledgeGraph
	if err := json.Unmarshal([]byte(response), &kg); err != nil {
		return nil, fmt.Errorf("%w: %v (response: %s)", ErrInvalidJSON, err, truncate(response, 200))
	}

	// Filter out invalid edges (referencing non-existent nodes)
	kg = filterInvalidEdges(kg)

	return &kg, nil
}

// parseTemporalKnowledgeGraph parses a TemporalKnowledgeGraph from JSON response.
func (e *OllamaExtractor) parseTemporalKnowledgeGraph(response string) (*TemporalKnowledgeGraph, error) {
	response = cleanJSONResponse(response)

	var tkg TemporalKnowledgeGraph
	if err := json.Unmarshal([]byte(response), &tkg); err != nil {
		return nil, fmt.Errorf("%w: %v (response: %s)", ErrInvalidJSON, err, truncate(response, 200))
	}

	// Filter invalid edges
	tkg.KnowledgeGraph = filterInvalidEdges(tkg.KnowledgeGraph)

	return &tkg, nil
}

// cleanJSONResponse removes markdown code blocks and trims whitespace.
func cleanJSONResponse(s string) string {
	s = strings.TrimSpace(s)

	// Remove markdown code blocks
	if strings.HasPrefix(s, "```json") {
		s = strings.TrimPrefix(s, "```json")
	} else if strings.HasPrefix(s, "```") {
		s = strings.TrimPrefix(s, "```")
	}

	if strings.HasSuffix(s, "```") {
		s = strings.TrimSuffix(s, "```")
	}

	return strings.TrimSpace(s)
}

// filterInvalidEdges removes edges that reference non-existent nodes.
func filterInvalidEdges(kg KnowledgeGraph) KnowledgeGraph {
	nodeIDs := make(map[string]bool)
	for _, node := range kg.Nodes {
		nodeIDs[node.ID] = true
	}

	validEdges := make([]Edge, 0, len(kg.Edges))
	for _, edge := range kg.Edges {
		if nodeIDs[edge.SourceNodeID] && nodeIDs[edge.TargetNodeID] {
			validEdges = append(validEdges, edge)
		}
	}

	kg.Edges = validEdges
	return kg
}

// truncate truncates a string to maxLen characters.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
