package scenarios

import (
	"context"
	"math/rand"
	"testing"
	"time"

	"github.com/phenomenon0/vectordb/benchmarks/testdata"
	"github.com/phenomenon0/vectordb/internal/sparse"
)

// Sparse vector benchmarks: BM25-scored inverted index with realistic text corpora.

func BenchmarkSparse_BM25_Insert(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	vocabSize := 50_000
	docs := testdata.GenerateSparseDocuments(b.N+100, vocabSize, 50, 200, rng)

	idx := sparse.NewInvertedIndex(vocabSize)
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		doc := docs[i%len(docs)]
		if err := idx.Add(ctx, uint64(i), doc); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()

	b.ReportMetric(float64(b.N)/b.Elapsed().Seconds(), "insert_qps")
}

func BenchmarkSparse_BM25_Search(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	vocabSize := 50_000
	scale := 50_000
	if testing.Short() {
		scale = 10_000
	}

	docs := testdata.GenerateSparseDocuments(scale, vocabSize, 50, 200, rng)
	queries := testdata.GenerateSparseQueries(100, vocabSize, 5, 15, rng)

	idx := sparse.NewInvertedIndex(vocabSize)
	ctx := context.Background()
	for i, doc := range docs {
		if err := idx.Add(ctx, uint64(i), doc); err != nil {
			b.Fatal(err)
		}
	}

	b.ResetTimer()
	latencies := make([]time.Duration, 0, b.N)
	for i := 0; i < b.N; i++ {
		q := queries[i%len(queries)]
		start := time.Now()
		_, err := idx.Search(ctx, q, 10)
		latencies = append(latencies, time.Since(start))
		if err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()

	if len(latencies) > 0 {
		var total time.Duration
		for _, l := range latencies {
			total += l
		}
		qps := float64(len(latencies)) / (float64(total) / float64(time.Second))
		b.ReportMetric(qps, "qps")
	}
}

func BenchmarkSparse_Corpus(b *testing.B) {
	rng := rand.New(rand.NewSource(42))
	vocabSize := 50_000

	corpora := []struct {
		Name    string
		MinToks int
		MaxToks int
	}{
		{"short_queries", 3, 8},
		{"medium_docs", 50, 150},
		{"long_docs", 200, 500},
	}

	for _, corpus := range corpora {
		b.Run(corpus.Name, func(b *testing.B) {
			scale := 20_000
			if testing.Short() {
				scale = 5_000
			}
			docs := testdata.GenerateSparseDocuments(scale, vocabSize, corpus.MinToks, corpus.MaxToks, rng)
			queries := testdata.GenerateSparseQueries(100, vocabSize, 5, 15, rng)

			idx := sparse.NewInvertedIndex(vocabSize)
			ctx := context.Background()
			for i, doc := range docs {
				if err := idx.Add(ctx, uint64(i), doc); err != nil {
					b.Fatal(err)
				}
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				q := queries[i%len(queries)]
				if _, err := idx.Search(ctx, q, 10); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
