package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"time"
)

// ======================================================================================
// Google Gemini Embedder (gemini-embedding-2-preview, 3072d default, MRL: 128-3072)
// Asymmetric: taskType RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY
// ======================================================================================

type GeminiEmbedder struct {
	apiKey string
	model  string
	dim    int
	client *http.Client
}

func NewGeminiEmbedder(apiKey string, dim int) *GeminiEmbedder {
	if dim <= 0 {
		dim = 3072 // gemini-embedding-2 default
	}
	return &GeminiEmbedder{
		apiKey: apiKey,
		model:  "gemini-embedding-2-preview",
		dim:    dim,
		client: &http.Client{Timeout: 30 * time.Second},
	}
}

func (e *GeminiEmbedder) Dim() int { return e.dim }

func (e *GeminiEmbedder) Embed(text string) ([]float32, error) {
	return e.embed(text, "RETRIEVAL_DOCUMENT")
}

func (e *GeminiEmbedder) EmbedQuery(text string) ([]float32, error) {
	return e.embed(text, "RETRIEVAL_QUERY")
}

func (e *GeminiEmbedder) embed(text, taskType string) ([]float32, error) {
	if text == "" {
		text = "empty"
	}

	reqBody := map[string]interface{}{
		"model": "models/" + e.model,
		"content": map[string]interface{}{
			"parts": []map[string]string{{"text": text}},
		},
		"taskType":             taskType,
		"outputDimensionality": e.dim,
	}
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal error: %w", err)
	}

	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/%s:embedContent?key=%s", e.model, e.apiKey)
	req, err := http.NewRequest("POST", url, bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("request error: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("Gemini API error: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		var errResp struct {
			Error struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		json.NewDecoder(resp.Body).Decode(&errResp)
		return nil, fmt.Errorf("Gemini API error %d: %s", resp.StatusCode, errResp.Error.Message)
	}

	var result struct {
		Embedding struct {
			Values []float64 `json:"values"`
		} `json:"embedding"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode error: %w", err)
	}

	if len(result.Embedding.Values) == 0 {
		return nil, fmt.Errorf("no embedding returned")
	}

	vec := make([]float32, len(result.Embedding.Values))
	for i, v := range result.Embedding.Values {
		vec[i] = float32(v)
	}
	return vec, nil
}

// ======================================================================================
// Voyage AI Embedder (voyage-4-large, 1024d default, MRL: 256-2048)
// Asymmetric: input_type "document" vs "query"
// ======================================================================================

type VoyageEmbedder struct {
	apiKey string
	model  string
	dim    int
	client *http.Client
}

func NewVoyageEmbedder(apiKey, model string, dim int) *VoyageEmbedder {
	if model == "" {
		model = "voyage-4-large"
	}
	if dim <= 0 {
		dim = 1024
	}
	return &VoyageEmbedder{
		apiKey: apiKey,
		model:  model,
		dim:    dim,
		client: &http.Client{Timeout: 30 * time.Second},
	}
}

func (e *VoyageEmbedder) Dim() int { return e.dim }

func (e *VoyageEmbedder) Embed(text string) ([]float32, error) {
	return e.embed(text, "document")
}

func (e *VoyageEmbedder) EmbedQuery(text string) ([]float32, error) {
	return e.embed(text, "query")
}

func (e *VoyageEmbedder) embed(text, inputType string) ([]float32, error) {
	if text == "" {
		text = "empty"
	}

	reqBody := map[string]interface{}{
		"model":      e.model,
		"input":      []string{text},
		"input_type": inputType,
	}
	if e.dim > 0 {
		reqBody["output_dimension"] = e.dim
	}
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal error: %w", err)
	}

	req, err := http.NewRequest("POST", "https://api.voyageai.com/v1/embeddings", bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("request error: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.apiKey)

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("Voyage API error: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		var errResp struct {
			Detail string `json:"detail"`
		}
		json.NewDecoder(resp.Body).Decode(&errResp)
		return nil, fmt.Errorf("Voyage API error %d: %s", resp.StatusCode, errResp.Detail)
	}

	var result struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode error: %w", err)
	}

	if len(result.Data) == 0 || len(result.Data[0].Embedding) == 0 {
		return nil, fmt.Errorf("no embedding returned")
	}

	vec := make([]float32, len(result.Data[0].Embedding))
	for i, v := range result.Data[0].Embedding {
		vec[i] = float32(v)
	}
	return vec, nil
}

// ======================================================================================
// Jina AI Embedder (jina-embeddings-v3, 1024d default, MRL: 32-1024)
// Asymmetric: task "retrieval.passage" vs "retrieval.query"
// ======================================================================================

type JinaEmbedder struct {
	apiKey string
	model  string
	dim    int
	client *http.Client
}

func NewJinaEmbedder(apiKey, model string, dim int) *JinaEmbedder {
	if model == "" {
		model = "jina-embeddings-v3"
	}
	if dim <= 0 {
		dim = 1024
	}
	return &JinaEmbedder{
		apiKey: apiKey,
		model:  model,
		dim:    dim,
		client: &http.Client{Timeout: 30 * time.Second},
	}
}

func (e *JinaEmbedder) Dim() int { return e.dim }

func (e *JinaEmbedder) Embed(text string) ([]float32, error) {
	return e.embed(text, "retrieval.passage")
}

func (e *JinaEmbedder) EmbedQuery(text string) ([]float32, error) {
	return e.embed(text, "retrieval.query")
}

func (e *JinaEmbedder) embed(text, task string) ([]float32, error) {
	if text == "" {
		text = "empty"
	}

	reqBody := map[string]interface{}{
		"model":      e.model,
		"input":      []string{text},
		"task":       task,
		"dimensions": e.dim,
	}
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal error: %w", err)
	}

	req, err := http.NewRequest("POST", "https://api.jina.ai/v1/embeddings", bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("request error: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.apiKey)

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("Jina API error: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		var errResp struct {
			Detail string `json:"detail"`
		}
		json.NewDecoder(resp.Body).Decode(&errResp)
		return nil, fmt.Errorf("Jina API error %d: %s", resp.StatusCode, errResp.Detail)
	}

	var result struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode error: %w", err)
	}

	if len(result.Data) == 0 || len(result.Data[0].Embedding) == 0 {
		return nil, fmt.Errorf("no embedding returned")
	}

	vec := make([]float32, len(result.Data[0].Embedding))
	for i, v := range result.Data[0].Embedding {
		vec[i] = float32(v)
	}
	return vec, nil
}

// ======================================================================================
// Cohere Embedder (embed-english-v3.0, 1024d)
// Asymmetric: input_type "search_document" vs "search_query"
// ======================================================================================

type CohereEmbedder struct {
	apiKey string
	model  string
	dim    int
	client *http.Client
}

func NewCohereEmbedder(apiKey, model string) *CohereEmbedder {
	if model == "" {
		model = "embed-english-v3.0"
	}
	return &CohereEmbedder{
		apiKey: apiKey,
		model:  model,
		dim:    1024,
		client: &http.Client{Timeout: 30 * time.Second},
	}
}

func (e *CohereEmbedder) Dim() int { return e.dim }

func (e *CohereEmbedder) Embed(text string) ([]float32, error) {
	return e.embed(text, "search_document")
}

func (e *CohereEmbedder) EmbedQuery(text string) ([]float32, error) {
	return e.embed(text, "search_query")
}

func (e *CohereEmbedder) embed(text, inputType string) ([]float32, error) {
	if text == "" {
		text = "empty"
	}

	reqBody := map[string]interface{}{
		"texts":      []string{text},
		"model":      e.model,
		"input_type": inputType,
	}
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal error: %w", err)
	}

	req, err := http.NewRequest("POST", "https://api.cohere.com/v1/embed", bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("request error: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.apiKey)

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("Cohere API error: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		var errResp struct {
			Message string `json:"message"`
		}
		json.NewDecoder(resp.Body).Decode(&errResp)
		return nil, fmt.Errorf("Cohere API error %d: %s", resp.StatusCode, errResp.Message)
	}

	var result struct {
		Embeddings [][]float64 `json:"embeddings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode error: %w", err)
	}

	if len(result.Embeddings) == 0 || len(result.Embeddings[0]) == 0 {
		return nil, fmt.Errorf("no embedding returned")
	}

	// L2 normalize
	raw := result.Embeddings[0]
	vec := make([]float32, len(raw))
	var norm float64
	for i, v := range raw {
		vec[i] = float32(v)
		norm += v * v
	}
	norm = math.Sqrt(norm)
	if norm > 0 {
		for i := range vec {
			vec[i] /= float32(norm)
		}
	}
	return vec, nil
}

// ======================================================================================
// Mistral Embedder (mistral-embed, 1024d) — symmetric, no query/doc distinction
// ======================================================================================

type MistralEmbedder struct {
	apiKey string
	model  string
	dim    int
	client *http.Client
}

func NewMistralEmbedder(apiKey, model string) *MistralEmbedder {
	if model == "" {
		model = "mistral-embed"
	}
	return &MistralEmbedder{
		apiKey: apiKey,
		model:  model,
		dim:    1024,
		client: &http.Client{Timeout: 30 * time.Second},
	}
}

func (e *MistralEmbedder) Dim() int { return e.dim }

func (e *MistralEmbedder) EmbedQuery(text string) ([]float32, error) { return e.Embed(text) }

func (e *MistralEmbedder) Embed(text string) ([]float32, error) {
	if text == "" {
		text = "empty"
	}

	reqBody := map[string]interface{}{
		"model": e.model,
		"input": []string{text},
	}
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal error: %w", err)
	}

	req, err := http.NewRequest("POST", "https://api.mistral.ai/v1/embeddings", bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("request error: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.apiKey)

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("Mistral API error: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		var errResp struct {
			Message string `json:"message"`
		}
		json.NewDecoder(resp.Body).Decode(&errResp)
		return nil, fmt.Errorf("Mistral API error %d: %s", resp.StatusCode, errResp.Message)
	}

	// OpenAI-compatible response format
	var result struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode error: %w", err)
	}

	if len(result.Data) == 0 || len(result.Data[0].Embedding) == 0 {
		return nil, fmt.Errorf("no embedding returned")
	}

	vec := make([]float32, len(result.Data[0].Embedding))
	for i, v := range result.Data[0].Embedding {
		vec[i] = float32(v)
	}
	return vec, nil
}
