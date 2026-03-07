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

// OpenAIExtractor implements entity extraction using OpenAI API.
type OpenAIExtractor struct {
	cfg    ExtractorConfig
	client *http.Client
}

// OpenAIChatRequest is the request format for OpenAI Chat API.
type OpenAIChatRequest struct {
	Model          string          `json:"model"`
	Messages       []OpenAIMessage `json:"messages"`
	Temperature    float32         `json:"temperature,omitempty"`
	MaxTokens      int             `json:"max_tokens,omitempty"`
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
}

// OpenAIMessage represents a chat message.
type OpenAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ResponseFormat specifies the output format.
type ResponseFormat struct {
	Type string `json:"type"` // "json_object" for JSON mode
}

// OpenAIChatResponse is the response from OpenAI Chat API.
type OpenAIChatResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index   int `json:"index"`
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error,omitempty"`
}

// NewOpenAIExtractor creates a new OpenAI-based extractor.
func NewOpenAIExtractor(cfg ExtractorConfig) (*OpenAIExtractor, error) {
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("OpenAI API key required")
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.openai.com/v1"
	}
	if cfg.Model == "" {
		cfg.Model = "gpt-4o-mini"
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

	return &OpenAIExtractor{
		cfg: cfg,
		client: &http.Client{
			Timeout: time.Duration(cfg.TimeoutSecs) * time.Second,
		},
	}, nil
}

// Provider returns "openai".
func (e *OpenAIExtractor) Provider() string {
	return "openai"
}

// Model returns the configured model name.
func (e *OpenAIExtractor) Model() string {
	return e.cfg.Model
}

// Extract extracts a knowledge graph from text content.
func (e *OpenAIExtractor) Extract(ctx context.Context, content string) (*KnowledgeGraph, error) {
	if content == "" {
		return nil, ErrEmptyContent
	}

	prompt := e.cfg.CustomPrompt
	if prompt == "" {
		prompt = DefaultExtractionPrompt
	}

	response, err := e.callOpenAI(ctx, prompt, content)
	if err != nil {
		return nil, err
	}

	return e.parseKnowledgeGraph(response)
}

// ExtractTemporal extracts a knowledge graph with temporal events.
func (e *OpenAIExtractor) ExtractTemporal(ctx context.Context, content string) (*TemporalKnowledgeGraph, error) {
	if content == "" {
		return nil, ErrEmptyContent
	}

	prompt := e.cfg.TemporalPrompt
	if prompt == "" {
		prompt = DefaultTemporalPrompt
	}

	response, err := e.callOpenAI(ctx, prompt, content)
	if err != nil {
		return nil, err
	}

	return e.parseTemporalKnowledgeGraph(response)
}

// ExtractBatch extracts from multiple text chunks concurrently.
func (e *OpenAIExtractor) ExtractBatch(ctx context.Context, chunks []string) ([]*KnowledgeGraph, error) {
	results := make([]*KnowledgeGraph, len(chunks))
	errors := make([]error, len(chunks))

	sem := make(chan struct{}, e.cfg.Concurrency)
	var wg sync.WaitGroup

	for i, chunk := range chunks {
		wg.Add(1)
		go func(idx int, text string) {
			defer wg.Done()

			select {
			case sem <- struct{}{}:
				defer func() { <-sem }()
			case <-ctx.Done():
				errors[idx] = ctx.Err()
				return
			}

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

	for _, err := range errors {
		if err != nil {
			return results, fmt.Errorf("batch extraction had failures: %w", err)
		}
	}

	return results, nil
}

// callOpenAI sends a request to the OpenAI API.
func (e *OpenAIExtractor) callOpenAI(ctx context.Context, systemPrompt, userContent string) (string, error) {
	url := strings.TrimSuffix(e.cfg.BaseURL, "/") + "/chat/completions"

	req := OpenAIChatRequest{
		Model: e.cfg.Model,
		Messages: []OpenAIMessage{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: userContent},
		},
		Temperature:    e.cfg.Temperature,
		MaxTokens:      e.cfg.MaxTokens,
		ResponseFormat: &ResponseFormat{Type: "json_object"},
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
	httpReq.Header.Set("Authorization", "Bearer "+e.cfg.APIKey)

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

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read response: %w", err)
	}

	var openAIResp OpenAIChatResponse
	if err := json.Unmarshal(bodyBytes, &openAIResp); err != nil {
		return "", fmt.Errorf("decode response: %w", err)
	}

	if openAIResp.Error != nil {
		return "", fmt.Errorf("openai error: %s", openAIResp.Error.Message)
	}

	if len(openAIResp.Choices) == 0 {
		return "", fmt.Errorf("no choices in response")
	}

	return openAIResp.Choices[0].Message.Content, nil
}

// parseKnowledgeGraph parses a KnowledgeGraph from JSON response.
func (e *OpenAIExtractor) parseKnowledgeGraph(response string) (*KnowledgeGraph, error) {
	response = cleanJSONResponse(response)

	var kg KnowledgeGraph
	if err := json.Unmarshal([]byte(response), &kg); err != nil {
		return nil, fmt.Errorf("%w: %v (response: %s)", ErrInvalidJSON, err, truncate(response, 200))
	}

	kg = filterInvalidEdges(kg)
	return &kg, nil
}

// parseTemporalKnowledgeGraph parses a TemporalKnowledgeGraph from JSON response.
func (e *OpenAIExtractor) parseTemporalKnowledgeGraph(response string) (*TemporalKnowledgeGraph, error) {
	response = cleanJSONResponse(response)

	var tkg TemporalKnowledgeGraph
	if err := json.Unmarshal([]byte(response), &tkg); err != nil {
		return nil, fmt.Errorf("%w: %v (response: %s)", ErrInvalidJSON, err, truncate(response, 200))
	}

	tkg.KnowledgeGraph = filterInvalidEdges(tkg.KnowledgeGraph)
	return &tkg, nil
}
