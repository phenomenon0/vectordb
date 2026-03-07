package encoding

import (
	"strings"
	"testing"

	vcollection "github.com/phenomenon0/vectordb/internal/collection"
)

func TestEncodeSearchResults_Basic(t *testing.T) {
	docs := []vcollection.Document{
		{ID: 1, Metadata: map[string]interface{}{"title": "Hello World"}},
		{ID: 2, Metadata: map[string]interface{}{"title": "Go Programming", "score": 42}},
	}
	scores := []float32{0.95, 0.87}

	result := EncodeSearchResults(docs, scores)

	if !strings.HasPrefix(result, "@tab SearchResult [id score metadata]") {
		t.Errorf("expected @tab header, got: %s", result[:50])
	}
	if !strings.HasSuffix(result, "@end") {
		t.Errorf("expected @end footer")
	}
	if !strings.Contains(result, "1 0.9500") {
		t.Errorf("expected doc 1 with score 0.95, got: %s", result)
	}
	if !strings.Contains(result, "2 0.8700") {
		t.Errorf("expected doc 2 with score 0.87, got: %s", result)
	}
}

func TestEncodeSearchResults_Empty(t *testing.T) {
	result := EncodeSearchResults(nil, nil)
	expected := "@tab SearchResult [id score metadata]\n@end"
	if result != expected {
		t.Errorf("expected %q, got %q", expected, result)
	}
}

func TestEncodeSearchResults_MetadataTypes(t *testing.T) {
	docs := []vcollection.Document{
		{ID: 1, Metadata: map[string]interface{}{
			"name":   "test doc",
			"count":  42,
			"active": true,
			"weight": 0.5,
		}},
	}
	scores := []float32{0.99}

	result := EncodeSearchResults(docs, scores)

	if !strings.Contains(result, "1 0.9900") {
		t.Errorf("missing doc entry")
	}
	// Verify it contains the metadata values (order may vary)
	if !strings.Contains(result, "42") {
		t.Errorf("missing int metadata")
	}
	if !strings.Contains(result, "0.5") {
		t.Errorf("missing float metadata")
	}
}

func TestEncodeSearchResults_EmptyMetadata(t *testing.T) {
	docs := []vcollection.Document{
		{ID: 5, Metadata: nil},
	}
	scores := []float32{0.5}

	result := EncodeSearchResults(docs, scores)
	if !strings.Contains(result, "5 0.5000 {}") {
		t.Errorf("expected empty metadata {}, got: %s", result)
	}
}

func TestEncodeSearchResults_QuotedStrings(t *testing.T) {
	docs := []vcollection.Document{
		{ID: 1, Metadata: map[string]interface{}{
			"title": "has spaces here",
		}},
	}
	scores := []float32{0.8}

	result := EncodeSearchResults(docs, scores)
	if !strings.Contains(result, `"has spaces here"`) {
		t.Errorf("expected quoted string with spaces, got: %s", result)
	}
}

func TestQuoteIfNeeded(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"simple", "simple"},
		{"has space", `"has space"`},
		{"", `""`},
		{"with:colon", `"with:colon"`},
		{"[brackets]", `"[brackets]"`},
	}

	for _, tc := range tests {
		got := quoteIfNeeded(tc.input)
		if got != tc.want {
			t.Errorf("quoteIfNeeded(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}
