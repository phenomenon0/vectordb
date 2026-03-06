package feedback

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Processor processes feedback and extracts signals.
type Processor struct {
	store  *Store
	cfg    FeedbackConfig
	client *http.Client
}

// NewProcessor creates a new feedback processor.
func NewProcessor(store *Store, cfg FeedbackConfig) *Processor {
	return &Processor{
		store: store,
		cfg:   cfg,
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// ProcessNaturalFeedback processes natural language feedback.
func (p *Processor) ProcessNaturalFeedback(ctx context.Context, interactionID, text string) (*Feedback, error) {
	fb := NewFeedback(interactionID, FeedbackTypeNatural)
	fb.Text = text

	// Analyze sentiment
	sentiment, err := p.analyzeSentiment(ctx, text)
	if err != nil {
		// Fall back to rule-based
		sentiment = p.ruleBasedSentiment(text)
	}
	fb.Sentiment = sentiment

	// Record the feedback
	if err := p.store.RecordFeedback(ctx, fb); err != nil {
		return nil, err
	}

	return fb, nil
}

// ProcessExplicitFeedback processes explicit rating/thumbs feedback.
func (p *Processor) ProcessExplicitFeedback(ctx context.Context, interactionID string, rating int, clickedIDs, rejectedIDs []string) (*Feedback, error) {
	// Determine feedback type and sentiment from rating
	var fbType FeedbackType
	var sentiment float32

	switch {
	case rating >= 4:
		fbType = FeedbackTypeThumbsUp
		sentiment = float32(rating-3) / 2.0 // 4->0.5, 5->1.0
	case rating <= 2:
		fbType = FeedbackTypeThumbsDown
		sentiment = float32(rating-3) / 2.0 // 2->-0.5, 1->-1.0
	default:
		fbType = FeedbackTypeExplicit
		sentiment = 0 // Neutral
	}

	fb := NewFeedback(interactionID, fbType)
	fb.Rating = rating
	fb.Sentiment = sentiment
	fb.ClickedIDs = clickedIDs
	fb.RejectedIDs = rejectedIDs

	if err := p.store.RecordFeedback(ctx, fb); err != nil {
		return nil, err
	}

	return fb, nil
}

// ProcessImplicitFeedback processes implicit signals (clicks, dwell time, etc).
func (p *Processor) ProcessImplicitFeedback(ctx context.Context, interactionID string, signal ImplicitSignal) (*Feedback, error) {
	var fbType FeedbackType
	var sentiment float32

	switch signal.Type {
	case "click":
		fbType = FeedbackTypeClick
		sentiment = p.cfg.ClickPositiveWeight
	case "dwell":
		if signal.DurationMs >= p.cfg.DwellTimeThresholdMs {
			fbType = FeedbackTypeDwell
			sentiment = p.cfg.DwellPositiveWeight
		} else {
			fbType = FeedbackTypeIgnore
			sentiment = -p.cfg.IgnoreNegativeWeight
		}
	case "ignore":
		fbType = FeedbackTypeIgnore
		sentiment = -p.cfg.IgnoreNegativeWeight
	case "requery":
		fbType = FeedbackTypeRequery
		sentiment = -0.5 // Strong negative signal
	default:
		return nil, fmt.Errorf("unknown signal type: %s", signal.Type)
	}

	fb := NewFeedback(interactionID, fbType)
	fb.Sentiment = sentiment
	fb.ClickedIDs = signal.TargetIDs

	if err := p.store.RecordFeedback(ctx, fb); err != nil {
		return nil, err
	}

	return fb, nil
}

// ImplicitSignal represents an implicit feedback signal.
type ImplicitSignal struct {
	Type       string   `json:"type"`        // click, dwell, ignore, requery
	TargetIDs  []string `json:"target_ids"`  // Which results
	DurationMs int      `json:"duration_ms"` // For dwell time
}

// analyzeSentiment uses LLM to analyze sentiment of text.
func (p *Processor) analyzeSentiment(ctx context.Context, text string) (float32, error) {
	if !p.cfg.EnableLLMSentiment {
		return p.ruleBasedSentiment(text), nil
	}

	switch p.cfg.LLMProvider {
	case "ollama", "":
		return p.ollamaSentiment(ctx, text)
	case "openai":
		return p.openaiSentiment(ctx, text)
	default:
		return p.ruleBasedSentiment(text), nil
	}
}

// ollamaSentiment uses Ollama for sentiment analysis.
func (p *Processor) ollamaSentiment(ctx context.Context, text string) (float32, error) {
	url := strings.TrimSuffix(p.cfg.LLMURL, "/") + "/api/generate"

	reqBody := map[string]any{
		"model":  p.cfg.LLMModel,
		"prompt": text,
		"system": sentimentPrompt,
		"stream": false,
		"format": "json",
		"options": map[string]any{
			"temperature": 0.1,
			"num_predict": 50,
		},
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return 0, err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return 0, fmt.Errorf("ollama returned %d", resp.StatusCode)
	}

	var ollamaResp struct {
		Response string `json:"response"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&ollamaResp); err != nil {
		return 0, err
	}

	return parseSentimentResponse(ollamaResp.Response)
}

// openaiSentiment uses OpenAI for sentiment analysis.
func (p *Processor) openaiSentiment(ctx context.Context, text string) (float32, error) {
	url := strings.TrimSuffix(p.cfg.LLMURL, "/") + "/chat/completions"
	if p.cfg.LLMURL == "" {
		url = "https://api.openai.com/v1/chat/completions"
	}

	reqBody := map[string]any{
		"model": p.cfg.LLMModel,
		"messages": []map[string]string{
			{"role": "system", "content": sentimentPrompt},
			{"role": "user", "content": text},
		},
		"temperature":     0.1,
		"max_tokens":      50,
		"response_format": map[string]string{"type": "json_object"},
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return 0, err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.cfg.LLMAPIKey)

	resp, err := p.client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return 0, fmt.Errorf("openai returned %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var openaiResp struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&openaiResp); err != nil {
		return 0, err
	}

	if len(openaiResp.Choices) == 0 {
		return 0, fmt.Errorf("no choices in response")
	}

	return parseSentimentResponse(openaiResp.Choices[0].Message.Content)
}

// ruleBasedSentiment uses simple rules to estimate sentiment.
func (p *Processor) ruleBasedSentiment(text string) float32 {
	text = strings.ToLower(text)

	var score float32 = 0

	// Positive indicators
	positiveWords := []string{
		"great", "good", "excellent", "perfect", "helpful", "useful", "accurate",
		"correct", "thanks", "thank", "awesome", "amazing", "love", "best",
		"well done", "exactly", "right", "yes", "nice", "fantastic", "wonderful",
	}

	// Negative indicators
	negativeWords := []string{
		"bad", "wrong", "incorrect", "useless", "unhelpful", "terrible", "awful",
		"missed", "missing", "didn't", "not", "no", "poor", "worse", "worst",
		"fail", "failed", "error", "broken", "confused", "confusing", "irrelevant",
	}

	for _, word := range positiveWords {
		if strings.Contains(text, word) {
			score += 0.2
		}
	}

	for _, word := range negativeWords {
		if strings.Contains(text, word) {
			score -= 0.25
		}
	}

	// Clamp to [-1, 1]
	if score > 1 {
		score = 1
	}
	if score < -1 {
		score = -1
	}

	return score
}

// parseSentimentResponse parses the LLM response for sentiment.
func parseSentimentResponse(response string) (float32, error) {
	response = strings.TrimSpace(response)

	// Try to parse as JSON
	var result struct {
		Sentiment float32 `json:"sentiment"`
		Score     float32 `json:"score"`
	}

	if err := json.Unmarshal([]byte(response), &result); err == nil {
		if result.Sentiment != 0 {
			return result.Sentiment, nil
		}
		if result.Score != 0 {
			return result.Score, nil
		}
	}

	// Try to extract number
	response = strings.ReplaceAll(response, "```json", "")
	response = strings.ReplaceAll(response, "```", "")
	response = strings.TrimSpace(response)

	// Try parsing again
	if err := json.Unmarshal([]byte(response), &result); err == nil {
		if result.Sentiment != 0 {
			return result.Sentiment, nil
		}
		return result.Score, nil
	}

	return 0, fmt.Errorf("could not parse sentiment from: %s", response)
}

const sentimentPrompt = `Analyze the sentiment of the user's feedback about search results.
Return ONLY a JSON object with a "sentiment" field containing a number from -1 to 1:
- -1 = very negative (frustrated, unhelpful results)
- 0 = neutral
- 1 = very positive (satisfied, helpful results)

Example responses:
{"sentiment": 0.8}  - for "Great results, exactly what I needed!"
{"sentiment": -0.6} - for "These results aren't relevant to my question"
{"sentiment": 0.0}  - for "OK I guess"

Respond ONLY with the JSON object, no explanation.`

// ProcessBatch processes multiple feedback items.
func (p *Processor) ProcessBatch(ctx context.Context, items []FeedbackItem) ([]*Feedback, error) {
	results := make([]*Feedback, len(items))

	for i, item := range items {
		var fb *Feedback
		var err error

		switch item.Type {
		case "natural":
			fb, err = p.ProcessNaturalFeedback(ctx, item.InteractionID, item.Text)
		case "explicit":
			fb, err = p.ProcessExplicitFeedback(ctx, item.InteractionID, item.Rating, item.ClickedIDs, item.RejectedIDs)
		case "implicit":
			fb, err = p.ProcessImplicitFeedback(ctx, item.InteractionID, ImplicitSignal{
				Type:       item.SignalType,
				TargetIDs:  item.ClickedIDs,
				DurationMs: item.DurationMs,
			})
		}

		if err != nil {
			// Continue processing others
			continue
		}
		results[i] = fb
	}

	return results, nil
}

// FeedbackItem represents a feedback item for batch processing.
type FeedbackItem struct {
	Type          string   `json:"type"` // natural, explicit, implicit
	InteractionID string   `json:"interaction_id"`
	Text          string   `json:"text,omitempty"`
	Rating        int      `json:"rating,omitempty"`
	ClickedIDs    []string `json:"clicked_ids,omitempty"`
	RejectedIDs   []string `json:"rejected_ids,omitempty"`
	SignalType    string   `json:"signal_type,omitempty"` // click, dwell, ignore, requery
	DurationMs    int      `json:"duration_ms,omitempty"`
}
