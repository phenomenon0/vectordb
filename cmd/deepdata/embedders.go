package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io"
	"math"
	"math/rand"
	"net/http"
	"sort"
	"strings"
	"time"
)

// ======================================================================================
// Embedder/Reranker interfaces and hash fallback
// ======================================================================================

type Embedder interface {
	Embed(text string) ([]float32, error)      // encode as document (for indexing)
	EmbedQuery(text string) ([]float32, error) // encode as query (for searching)
	Dim() int
}

type Reranker interface {
	Rerank(query string, docs []string, topK int) ([]string, []float32, string, error)
}

type HashEmbedder struct {
	dim int
}

func NewHashEmbedder(dim int) *HashEmbedder {
	return &HashEmbedder{dim: dim}
}

func (e *HashEmbedder) Dim() int { return e.dim }

func (e *HashEmbedder) EmbedQuery(text string) ([]float32, error) { return e.Embed(text) }

func (e *HashEmbedder) Embed(text string) ([]float32, error) {
	if text == "" {
		text = "empty"
	}
	h := fnv.New64a()
	_, _ = h.Write([]byte(text))
	seed := int64(h.Sum64())
	vec := make([]float32, e.dim)
	r := rand.New(rand.NewSource(seed))
	var norm float64
	for i := 0; i < e.dim; i++ {
		val := r.Float64()*2 - 1
		vec[i] = float32(val)
		norm += val * val
	}
	norm = math.Sqrt(norm)
	if norm == 0 {
		return vec, nil
	}
	for i := range vec {
		vec[i] /= float32(norm)
	}
	return vec, nil
}

// OpenAIEmbedder uses OpenAI's text-embedding-3-small model
type OpenAIEmbedder struct {
	apiKey string
	model  string
	dim    int
	client *http.Client
}

func NewOpenAIEmbedder(apiKey string) *OpenAIEmbedder {
	return &OpenAIEmbedder{
		apiKey: apiKey,
		model:  "text-embedding-3-small",
		dim:    1536,
		client: &http.Client{Timeout: 30 * time.Second},
	}
}

func (e *OpenAIEmbedder) Dim() int { return e.dim }

func (e *OpenAIEmbedder) EmbedQuery(text string) ([]float32, error) { return e.Embed(text) }

func (e *OpenAIEmbedder) Embed(text string) ([]float32, error) {
	if text == "" {
		text = "empty"
	}

	reqBody := map[string]interface{}{
		"input": text,
		"model": e.model,
	}
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal error: %w", err)
	}

	req, err := http.NewRequest("POST", "https://api.openai.com/v1/embeddings", bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("request error: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.apiKey)

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("API error: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		var errResp struct {
			Error struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		json.NewDecoder(resp.Body).Decode(&errResp)
		return nil, fmt.Errorf("OpenAI API error %d: %s", resp.StatusCode, errResp.Error.Message)
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

	// Convert float64 to float32
	vec := make([]float32, len(result.Data[0].Embedding))
	for i, v := range result.Data[0].Embedding {
		vec[i] = float32(v)
	}
	return vec, nil
}

// OllamaEmbedder uses Ollama's local embedding models.
// Supports prefix-based asymmetry for nomic-embed-text models.
type OllamaEmbedder struct {
	baseURL string
	model   string
	dim     int
	isNomic bool // nomic models use "search_query: " / "search_document: " prefixes
	client  *http.Client
}

func NewOllamaEmbedder(baseURL, model string) *OllamaEmbedder {
	// nomic-embed-text produces 768-dim vectors
	dim := 768
	if model == "granite-embedding" {
		dim = 384
	}
	isNomic := strings.Contains(model, "nomic")
	return &OllamaEmbedder{
		baseURL: baseURL,
		model:   model,
		dim:     dim,
		isNomic: isNomic,
		client:  &http.Client{Timeout: 60 * time.Second},
	}
}

func (e *OllamaEmbedder) Dim() int { return e.dim }

// MaxChunkChars is the max characters per chunk (~2000 tokens for safety)
const MaxChunkChars = 6000

func (e *OllamaEmbedder) Embed(text string) ([]float32, error) {
	if e.isNomic {
		return e.embedWithPrefix(text, "search_document: ")
	}
	return e.embedWithPrefix(text, "")
}

func (e *OllamaEmbedder) EmbedQuery(text string) ([]float32, error) {
	if e.isNomic {
		return e.embedWithPrefix(text, "search_query: ")
	}
	return e.embedWithPrefix(text, "")
}

func (e *OllamaEmbedder) embedWithPrefix(text, prefix string) ([]float32, error) {
	if text == "" {
		text = "empty"
	}

	// If text is short enough, embed directly
	if len(text) <= MaxChunkChars {
		return e.embedSingle(prefix + text)
	}

	// Long text: chunk and average embeddings (late chunking / pooling)
	chunks := smartChunk(text, MaxChunkChars, 200) // 200 char overlap
	if len(chunks) == 0 {
		return e.embedSingle(prefix + text[:MaxChunkChars])
	}

	// Embed each chunk and compute weighted average
	var allVecs [][]float32
	var weights []float32
	for _, chunk := range chunks {
		vec, err := e.embedSingle(prefix + chunk)
		if err != nil {
			continue // Skip failed chunks
		}
		allVecs = append(allVecs, vec)
		// Weight by chunk length (longer chunks = more important)
		weights = append(weights, float32(len(chunk)))
	}

	if len(allVecs) == 0 {
		return nil, fmt.Errorf("all chunks failed to embed")
	}

	// Weighted average pooling
	return weightedAverageVecs(allVecs, weights), nil
}

// embedSingle embeds a single chunk of text
func (e *OllamaEmbedder) embedSingle(text string) ([]float32, error) {
	reqBody := map[string]interface{}{
		"model":  e.model,
		"prompt": text,
	}
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal error: %w", err)
	}

	req, err := http.NewRequest("POST", e.baseURL+"/api/embeddings", bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("request error: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("Ollama API error: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Ollama API error %d: %s", resp.StatusCode, string(body))
	}

	var result struct {
		Embedding []float64 `json:"embedding"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode error: %w", err)
	}

	if len(result.Embedding) == 0 {
		return nil, fmt.Errorf("no embedding returned")
	}

	// Convert float64 to float32 and L2 normalize
	vec := make([]float32, len(result.Embedding))
	var norm float64
	for i, v := range result.Embedding {
		vec[i] = float32(v)
		norm += v * v
	}
	norm = math.Sqrt(norm)
	if norm > 0 {
		for i := range vec {
			vec[i] /= float32(norm)
		}
	}
	if e.dim == 0 {
		e.dim = len(vec)
	}
	return vec, nil
}

// smartChunk splits text into semantic chunks with overlap
// Uses sentence boundaries for cleaner splits
func smartChunk(text string, maxChars, overlap int) []string {
	if len(text) <= maxChars {
		return []string{text}
	}

	var chunks []string

	// Split into sentences first (approximate)
	sentences := splitSentences(text)

	var currentChunk strings.Builder
	var currentLen int

	for _, sentence := range sentences {
		sentLen := len(sentence)

		// If single sentence exceeds max, split it by words
		if sentLen > maxChars {
			// Flush current chunk
			if currentLen > 0 {
				chunks = append(chunks, strings.TrimSpace(currentChunk.String()))
				currentChunk.Reset()
				currentLen = 0
			}
			// Split long sentence by words
			wordChunks := splitByWords(sentence, maxChars, overlap)
			chunks = append(chunks, wordChunks...)
			continue
		}

		// Check if adding this sentence exceeds limit
		if currentLen+sentLen > maxChars && currentLen > 0 {
			// Save current chunk
			chunkText := strings.TrimSpace(currentChunk.String())
			chunks = append(chunks, chunkText)

			// Start new chunk with overlap from end of previous
			currentChunk.Reset()
			if overlap > 0 && len(chunkText) > overlap {
				// Get last N chars for overlap
				overlapText := chunkText[len(chunkText)-overlap:]
				// Try to start at word boundary
				if idx := strings.LastIndex(overlapText, " "); idx > 0 {
					overlapText = overlapText[idx+1:]
				}
				currentChunk.WriteString(overlapText)
				currentChunk.WriteString(" ")
				currentLen = len(overlapText) + 1
			} else {
				currentLen = 0
			}
		}

		currentChunk.WriteString(sentence)
		currentChunk.WriteString(" ")
		currentLen += sentLen + 1
	}

	// Don't forget the last chunk
	if currentLen > 0 {
		chunks = append(chunks, strings.TrimSpace(currentChunk.String()))
	}

	return chunks
}

// splitSentences splits text into sentences using common delimiters
func splitSentences(text string) []string {
	var sentences []string
	var current strings.Builder

	runes := []rune(text)
	for i, r := range runes {
		current.WriteRune(r)

		// Check for sentence end
		if r == '.' || r == '!' || r == '?' || r == '\n' {
			// Look ahead to confirm (avoid splitting on abbreviations like "Dr.")
			isEnd := true
			if r == '.' && i+1 < len(runes) {
				next := runes[i+1]
				// Not a sentence end if followed by lowercase or digit
				if (next >= 'a' && next <= 'z') || (next >= '0' && next <= '9') {
					isEnd = false
				}
			}
			if isEnd {
				s := strings.TrimSpace(current.String())
				if len(s) > 0 {
					sentences = append(sentences, s)
				}
				current.Reset()
			}
		}
	}

	// Remaining text
	if current.Len() > 0 {
		s := strings.TrimSpace(current.String())
		if len(s) > 0 {
			sentences = append(sentences, s)
		}
	}

	return sentences
}

// splitByWords splits text by words when sentences are too long
func splitByWords(text string, maxChars, overlap int) []string {
	words := strings.Fields(text)
	if len(words) == 0 {
		return nil
	}

	var chunks []string
	var current strings.Builder
	currentLen := 0

	for _, word := range words {
		wordLen := len(word)
		if currentLen+wordLen+1 > maxChars && currentLen > 0 {
			chunks = append(chunks, strings.TrimSpace(current.String()))
			current.Reset()
			currentLen = 0
		}
		if currentLen > 0 {
			current.WriteString(" ")
			currentLen++
		}
		current.WriteString(word)
		currentLen += wordLen
	}

	if currentLen > 0 {
		chunks = append(chunks, strings.TrimSpace(current.String()))
	}

	return chunks
}

// weightedAverageVecs computes weighted average of vectors and normalizes
func weightedAverageVecs(vecs [][]float32, weights []float32) []float32 {
	if len(vecs) == 0 {
		return nil
	}
	if len(vecs) == 1 {
		return vecs[0]
	}

	dim := len(vecs[0])
	result := make([]float32, dim)

	// Normalize weights
	var totalWeight float32
	for _, w := range weights {
		totalWeight += w
	}
	if totalWeight == 0 {
		totalWeight = 1
	}

	// Weighted sum
	for i, vec := range vecs {
		w := weights[i] / totalWeight
		for j, v := range vec {
			result[j] += v * w
		}
	}

	// L2 normalize the result
	var norm float32
	for _, v := range result {
		norm += v * v
	}
	norm = float32(math.Sqrt(float64(norm)))
	if norm > 0 {
		for i := range result {
			result[i] /= norm
		}
	}

	return result
}

type SimpleReranker struct {
	Embedder Embedder
}

func (r *SimpleReranker) Rerank(query string, docs []string, topK int) ([]string, []float32, string, error) {
	qVec, err := r.Embedder.EmbedQuery(query)
	if err != nil {
		return nil, nil, "", err
	}
	if topK <= 0 || topK > len(docs) {
		topK = len(docs)
	}
	bestDocs := make([]string, 0, topK)
	bestScores := make([]float32, 0, topK)

	for _, doc := range docs {
		dVec, err := r.Embedder.Embed(doc)
		if err != nil {
			continue
		}
		score := DotProduct(qVec, dVec)
		if len(bestDocs) < topK {
			bestDocs = append(bestDocs, doc)
			bestScores = append(bestScores, score)
			continue
		}
		minIdx := 0
		for i := 1; i < len(bestScores); i++ {
			if bestScores[i] < bestScores[minIdx] {
				minIdx = i
			}
		}
		if score > bestScores[minIdx] {
			bestScores[minIdx] = score
			bestDocs[minIdx] = doc
		}
	}

	order := make([]int, len(bestScores))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(i, j int) bool {
		return bestScores[order[i]] > bestScores[order[j]]
	})

	sortedDocs := make([]string, 0, len(order))
	sortedScores := make([]float32, 0, len(order))
	for _, idx := range order {
		sortedDocs = append(sortedDocs, bestDocs[idx])
		sortedScores = append(sortedScores, bestScores[idx])
	}

	return sortedDocs, sortedScores, "Simple rerank", nil
}

// ======================================================================================
// Utility
// ======================================================================================

func DotProduct(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}
