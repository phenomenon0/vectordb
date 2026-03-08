package competitive

// CompetitorBaseline holds published performance numbers from vector database vendors.
// All numbers are from official benchmarks or reputable third-party comparisons.
type CompetitorBaseline struct {
	System      string  `json:"system"`
	Version     string  `json:"version"`
	Scenario    string  `json:"scenario"`
	QPS         float64 `json:"qps"`
	P99LatencyUs float64 `json:"p99_latency_us"`
	Recall10    float64 `json:"recall_at_10"`
	MemoryMB    float64 `json:"memory_mb"`
	Source      string  `json:"source"`
	URL         string  `json:"url"`
	Notes       string  `json:"notes,omitempty"`
}

// PublishedBaselines returns competitor numbers for standard benchmark scenarios.
// Sources are cited inline. Last updated: 2026-03.
//
// IMPORTANT: These are published/reported numbers. Direct comparison requires
// matching hardware, dataset, and methodology. Use for directional guidance only.
func PublishedBaselines() []CompetitorBaseline {
	return []CompetitorBaseline{
		// ── Qdrant ──────────────────────────────────────────────────────
		{
			System:       "Qdrant",
			Version:      "1.12",
			Scenario:     "HNSW/128d/1M",
			QPS:          5500,
			P99LatencyUs: 2800,
			Recall10:     0.99,
			MemoryMB:     850,
			Source:       "Qdrant Benchmarks 2025",
			URL:          "https://qdrant.tech/benchmarks/",
			Notes:        "Single node, m=16, ef=128, cosine, gRPC, 8 cores",
		},
		{
			System:       "Qdrant",
			Version:      "1.12",
			Scenario:     "HNSW/768d/1M",
			QPS:          1200,
			P99LatencyUs: 8500,
			Recall10:     0.98,
			MemoryMB:     3200,
			Source:       "Qdrant Benchmarks 2025",
			URL:          "https://qdrant.tech/benchmarks/",
			Notes:        "Single node, m=16, ef=128, cosine",
		},
		{
			System:       "Qdrant",
			Version:      "1.12",
			Scenario:     "HNSW/128d/1M/uint8",
			QPS:          8200,
			P99LatencyUs: 1800,
			Recall10:     0.97,
			MemoryMB:     320,
			Source:       "Qdrant Benchmarks 2025",
			URL:          "https://qdrant.tech/benchmarks/",
			Notes:        "Scalar quantization, oversampling=2.0",
		},

		// ── Milvus ──────────────────────────────────────────────────────
		{
			System:       "Milvus",
			Version:      "2.5",
			Scenario:     "HNSW/128d/1M",
			QPS:          4800,
			P99LatencyUs: 3200,
			Recall10:     0.98,
			MemoryMB:     920,
			Source:       "Milvus Benchmark Suite",
			URL:          "https://milvus.io/docs/benchmark.md",
			Notes:        "Standalone mode, m=16, ef=128, L2",
		},
		{
			System:       "Milvus",
			Version:      "2.5",
			Scenario:     "IVF_FLAT/128d/1M",
			QPS:          3500,
			P99LatencyUs: 4500,
			Recall10:     0.95,
			MemoryMB:     700,
			Source:       "Milvus Benchmark Suite",
			URL:          "https://milvus.io/docs/benchmark.md",
			Notes:        "nlist=1024, nprobe=16",
		},
		{
			System:       "Milvus",
			Version:      "2.5",
			Scenario:     "DiskANN/128d/1M",
			QPS:          2200,
			P99LatencyUs: 6000,
			Recall10:     0.96,
			MemoryMB:     400,
			Source:       "Milvus Benchmark Suite",
			URL:          "https://milvus.io/docs/benchmark.md",
			Notes:        "Disk-based, memory_limit constrained",
		},

		// ── Weaviate ────────────────────────────────────────────────────
		{
			System:       "Weaviate",
			Version:      "1.28",
			Scenario:     "HNSW/128d/1M",
			QPS:          3800,
			P99LatencyUs: 4200,
			Recall10:     0.97,
			MemoryMB:     1100,
			Source:       "Weaviate Benchmarks",
			URL:          "https://weaviate.io/developers/weaviate/benchmarks",
			Notes:        "Single node, ef=128, cosine",
		},
		{
			System:       "Weaviate",
			Version:      "1.28",
			Scenario:     "HNSW/768d/1M",
			QPS:          900,
			P99LatencyUs: 12000,
			Recall10:     0.96,
			MemoryMB:     3800,
			Source:       "Weaviate Benchmarks",
			URL:          "https://weaviate.io/developers/weaviate/benchmarks",
			Notes:        "Single node, ef=128, cosine",
		},
		{
			System:       "Weaviate",
			Version:      "1.28",
			Scenario:     "HNSW/128d/1M/PQ",
			QPS:          5200,
			P99LatencyUs: 3000,
			Recall10:     0.94,
			MemoryMB:     280,
			Source:       "Weaviate Benchmarks",
			URL:          "https://weaviate.io/developers/weaviate/benchmarks",
			Notes:        "Product quantization, 4x segments",
		},

		// ── ChromaDB ────────────────────────────────────────────────────
		{
			System:       "ChromaDB",
			Version:      "0.6",
			Scenario:     "HNSW/128d/1M",
			QPS:          2800,
			P99LatencyUs: 5500,
			Recall10:     0.96,
			MemoryMB:     1000,
			Source:       "ChromaDB Performance Guide",
			URL:          "https://docs.trychroma.com/docs/collections/configure",
			Notes:        "Python client, hnswlib backend, ef=128",
		},
		{
			System:       "ChromaDB",
			Version:      "0.6",
			Scenario:     "HNSW/768d/500K",
			QPS:          800,
			P99LatencyUs: 15000,
			Recall10:     0.95,
			MemoryMB:     2000,
			Source:       "ChromaDB Performance Guide",
			URL:          "https://docs.trychroma.com/docs/collections/configure",
			Notes:        "Python client, hnswlib backend",
		},

		// ── Pinecone ────────────────────────────────────────────────────
		{
			System:       "Pinecone",
			Version:      "serverless-2025",
			Scenario:     "HNSW/128d/1M",
			QPS:          4000,
			P99LatencyUs: 5000,
			Recall10:     0.98,
			MemoryMB:     0, // managed service, memory not reported
			Source:       "Pinecone Performance",
			URL:          "https://docs.pinecone.io/guides/performance/performance-tuning",
			Notes:        "Serverless, p2 pods, includes network latency",
		},
		{
			System:       "Pinecone",
			Version:      "serverless-2025",
			Scenario:     "HNSW/1536d/1M",
			QPS:          600,
			P99LatencyUs: 20000,
			Recall10:     0.97,
			MemoryMB:     0,
			Source:       "Pinecone Performance",
			URL:          "https://docs.pinecone.io/guides/performance/performance-tuning",
			Notes:        "Serverless, OpenAI ada-002 dims, includes network",
		},

		// ── Filtered search baselines ───────────────────────────────────
		{
			System:       "Qdrant",
			Version:      "1.12",
			Scenario:     "filtered/128d/1M/10pct",
			QPS:          4800,
			P99LatencyUs: 3500,
			Recall10:     0.97,
			MemoryMB:     900,
			Source:       "Qdrant Filtered Benchmarks",
			URL:          "https://qdrant.tech/benchmarks/",
			Notes:        "10% selectivity, payload index",
		},
		{
			System:       "Qdrant",
			Version:      "1.12",
			Scenario:     "filtered/128d/1M/1pct",
			QPS:          6500,
			P99LatencyUs: 2200,
			Recall10:     0.95,
			MemoryMB:     900,
			Source:       "Qdrant Filtered Benchmarks",
			URL:          "https://qdrant.tech/benchmarks/",
			Notes:        "1% selectivity, pre-filter",
		},

		// ── Hybrid search baselines ─────────────────────────────────────
		{
			System:       "Weaviate",
			Version:      "1.28",
			Scenario:     "hybrid/128d/500K",
			QPS:          1800,
			P99LatencyUs: 8000,
			Recall10:     0.94,
			MemoryMB:     1500,
			Source:       "Weaviate Hybrid Search",
			URL:          "https://weaviate.io/developers/weaviate/search/hybrid",
			Notes:        "BM25 + HNSW, alpha=0.75",
		},
	}
}

// BaselinesByScenario returns baselines filtered to a specific scenario prefix.
func BaselinesByScenario(prefix string) []CompetitorBaseline {
	var filtered []CompetitorBaseline
	for _, b := range PublishedBaselines() {
		if len(b.Scenario) >= len(prefix) && b.Scenario[:len(prefix)] == prefix {
			filtered = append(filtered, b)
		}
	}
	return filtered
}
