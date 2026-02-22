package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/phenomenon0/Agent-GO/cowrie/codec"
)

// generateQueryResponseData creates realistic query response data
func generateQueryResponseData(n int) map[string]any {
	ids := make([]any, n)
	docs := make([]any, n)
	scores := make([]float32, n)
	meta := make([]map[string]string, n)

	for i := 0; i < n; i++ {
		ids[i] = fmt.Sprintf("doc_%d", i)
		docs[i] = fmt.Sprintf("This is document %d with content.", i)
		scores[i] = float32(n-i) / float32(n)
		meta[i] = map[string]string{
			"author":   fmt.Sprintf("author_%d", i%10),
			"category": fmt.Sprintf("cat_%d", i%5),
		}
	}

	return map[string]any{
		"ids":    ids,
		"docs":   docs,
		"scores": scores,
		"meta":   meta,
		"stats":  fmt.Sprintf("%d results", n),
		"next":   "",
	}
}

var benchSizes = []int{10, 100, 500, 1000}

// BenchmarkQueryResponseJSON benchmarks JSON decoding into QueryResponse
func BenchmarkQueryResponseJSON(b *testing.B) {
	for _, size := range benchSizes {
		data := generateQueryResponseData(size)
		jsonBytes, _ := json.Marshal(data)

		b.Run(fmt.Sprintf("n=%d", size), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(len(jsonBytes)))
			for i := 0; i < b.N; i++ {
				var resp QueryResponse
				if err := json.Unmarshal(jsonBytes, &resp); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkQueryResponseCowrie benchmarks Cowrie decoding into QueryResponse
// This uses the registered fast unmarshaler
func BenchmarkQueryResponseCowrie(b *testing.B) {
	cowrieCodec := codec.CowrieCodec{}

	for _, size := range benchSizes {
		data := generateQueryResponseData(size)
		var buf bytes.Buffer
		if err := cowrieCodec.Encode(&buf, data); err != nil {
			b.Fatal(err)
		}
		cowrieBytes := buf.Bytes()

		b.Run(fmt.Sprintf("n=%d", size), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(len(cowrieBytes)))
			for i := 0; i < b.N; i++ {
				var resp QueryResponse
				if err := cowrieCodec.Decode(bytes.NewReader(cowrieBytes), &resp); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkQueryResponseCowrieMap benchmarks Cowrie decoding into map[string]any
// This shows the generic decode path for comparison
func BenchmarkQueryResponseCowrieMap(b *testing.B) {
	cowrieCodec := codec.CowrieCodec{}

	for _, size := range benchSizes {
		data := generateQueryResponseData(size)
		var buf bytes.Buffer
		if err := cowrieCodec.Encode(&buf, data); err != nil {
			b.Fatal(err)
		}
		cowrieBytes := buf.Bytes()

		b.Run(fmt.Sprintf("n=%d", size), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(len(cowrieBytes)))
			for i := 0; i < b.N; i++ {
				var resp map[string]any
				if err := cowrieCodec.Decode(bytes.NewReader(cowrieBytes), &resp); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// TestQueryResponseCowrieRoundTrip verifies QueryResponse Cowrie round-trip works
func TestQueryResponseCowrieRoundTrip(t *testing.T) {
	cowrieCodec := codec.CowrieCodec{}

	for _, size := range []int{10, 100} {
		t.Run(fmt.Sprintf("n=%d", size), func(t *testing.T) {
			// Create test data
			data := generateQueryResponseData(size)

			// Encode
			var buf bytes.Buffer
			if err := cowrieCodec.Encode(&buf, data); err != nil {
				t.Fatalf("encode error: %v", err)
			}

			// Decode into QueryResponse (uses registered unmarshaler)
			var resp QueryResponse
			if err := cowrieCodec.Decode(bytes.NewReader(buf.Bytes()), &resp); err != nil {
				t.Fatalf("decode error: %v", err)
			}

			// Verify
			if len(resp.IDs) != size {
				t.Errorf("IDs length mismatch: got %d, want %d", len(resp.IDs), size)
			}
			if len(resp.Docs) != size {
				t.Errorf("Docs length mismatch: got %d, want %d", len(resp.Docs), size)
			}
			if len(resp.Scores) != size {
				t.Errorf("Scores length mismatch: got %d, want %d", len(resp.Scores), size)
			}
			if len(resp.Meta) != size {
				t.Errorf("Meta length mismatch: got %d, want %d", len(resp.Meta), size)
			}
			if resp.Stats != fmt.Sprintf("%d results", size) {
				t.Errorf("Stats mismatch: got %q", resp.Stats)
			}

			// Verify first score
			expectedScore := float32(size-0) / float32(size)
			if resp.Scores[0] != expectedScore {
				t.Errorf("Score[0] mismatch: got %v, want %v", resp.Scores[0], expectedScore)
			}
		})
	}
}
