// Package encoding provides alternative response encodings for DeepData HTTP APIs.
//
// The Glyph tabular encoder produces token-efficient output for RAG pipelines,
// saving 50-62% tokens compared to JSON when returning top-K search results
// to an LLM. Field names appear once in the header, not repeated per result.
package encoding

import (
	"fmt"
	"strings"

	vcollection "github.com/phenomenon0/vectordb/internal/collection"
)

// EncodeSearchResults encodes search results in Glyph tabular format.
//
// Output format:
//
//	@tab SearchResult [id score metadata]
//	1 0.95 {key:"val"}
//	2 0.87 {key2:"val2"}
//	@end
//
// This saves 50-62% tokens compared to JSON by stating field names only once.
func EncodeSearchResults(docs []vcollection.Document, scores []float32) string {
	if len(docs) == 0 {
		return "@tab SearchResult [id score metadata]\n@end"
	}

	var b strings.Builder
	b.Grow(len(docs) * 80) // Estimate ~80 chars per row

	b.WriteString("@tab SearchResult [id score metadata]\n")

	for i, doc := range docs {
		// ID
		fmt.Fprintf(&b, "%d", doc.ID)

		// Score
		if i < len(scores) {
			fmt.Fprintf(&b, " %.4f", scores[i])
		} else {
			b.WriteString(" 0")
		}

		// Metadata
		b.WriteByte(' ')
		encodeGlyphMap(&b, doc.Metadata)

		b.WriteByte('\n')
	}

	b.WriteString("@end")
	return b.String()
}

// encodeGlyphMap encodes a map as {key:"val" key2:42}.
func encodeGlyphMap(b *strings.Builder, m map[string]interface{}) {
	if len(m) == 0 {
		b.WriteString("{}")
		return
	}

	b.WriteByte('{')
	first := true
	for k, v := range m {
		if !first {
			b.WriteByte(' ')
		}
		first = false

		b.WriteString(quoteIfNeeded(k))
		b.WriteByte(':')
		encodeGlyphValue(b, v)
	}
	b.WriteByte('}')
}

// encodeGlyphValue encodes a single value in Glyph format.
func encodeGlyphValue(b *strings.Builder, v interface{}) {
	switch val := v.(type) {
	case nil:
		b.WriteString("null")
	case bool:
		if val {
			b.WriteString("t")
		} else {
			b.WriteString("f")
		}
	case int:
		fmt.Fprintf(b, "%d", val)
	case int64:
		fmt.Fprintf(b, "%d", val)
	case float64:
		fmt.Fprintf(b, "%g", val)
	case float32:
		fmt.Fprintf(b, "%g", val)
	case string:
		b.WriteString(quoteIfNeeded(val))
	case map[string]interface{}:
		encodeGlyphMap(b, val)
	case []interface{}:
		b.WriteByte('[')
		for i, elem := range val {
			if i > 0 {
				b.WriteByte(' ')
			}
			encodeGlyphValue(b, elem)
		}
		b.WriteByte(']')
	default:
		fmt.Fprintf(b, "%v", val)
	}
}

// quoteIfNeeded quotes a string if it contains spaces, special chars, or is empty.
func quoteIfNeeded(s string) string {
	if s == "" {
		return `""`
	}
	for _, c := range s {
		if c == ' ' || c == '"' || c == '{' || c == '}' || c == '[' || c == ']' ||
			c == ':' || c == '\n' || c == '\t' {
			return `"` + strings.ReplaceAll(s, `"`, `\"`) + `"`
		}
	}
	return s
}
