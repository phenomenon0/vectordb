package competitive

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"time"
)

// BenchmarkResult captures all metrics from a single benchmark run.
type BenchmarkResult struct {
	Scenario   string            `json:"scenario"`
	Index      string            `json:"index"`
	Dimension  int               `json:"dimension"`
	Scale      int               `json:"scale"`
	Quantizer  string            `json:"quantizer,omitempty"`
	Config     map[string]interface{} `json:"config,omitempty"`
	Timestamp  time.Time         `json:"timestamp"`

	// Latency
	MeanLatencyUs  float64 `json:"mean_latency_us"`
	P50LatencyUs   float64 `json:"p50_latency_us"`
	P95LatencyUs   float64 `json:"p95_latency_us"`
	P99LatencyUs   float64 `json:"p99_latency_us"`
	MaxLatencyUs   float64 `json:"max_latency_us"`

	// Throughput
	QPS float64 `json:"qps"`

	// Recall
	Recall1   float64 `json:"recall_at_1,omitempty"`
	Recall10  float64 `json:"recall_at_10,omitempty"`
	Recall100 float64 `json:"recall_at_100,omitempty"`

	// Memory
	MemoryMB     float64 `json:"memory_mb,omitempty"`
	BytesPerVec  float64 `json:"bytes_per_vector,omitempty"`

	// Insert
	InsertQPS float64 `json:"insert_qps,omitempty"`
}

// Percentile computes the p-th percentile from a sorted slice of durations.
func Percentile(latencies []time.Duration, p float64) time.Duration {
	if len(latencies) == 0 {
		return 0
	}
	sorted := make([]time.Duration, len(latencies))
	copy(sorted, latencies)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	idx := int(float64(len(sorted)-1) * p / 100.0)
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

// MeanDuration computes the arithmetic mean of durations.
func MeanDuration(latencies []time.Duration) time.Duration {
	if len(latencies) == 0 {
		return 0
	}
	var total time.Duration
	for _, d := range latencies {
		total += d
	}
	return total / time.Duration(len(latencies))
}

// LatencyStats computes standard latency stats from raw durations.
func LatencyStats(latencies []time.Duration) (mean, p50, p95, p99, max float64) {
	mean = float64(MeanDuration(latencies).Microseconds())
	p50 = float64(Percentile(latencies, 50).Microseconds())
	p95 = float64(Percentile(latencies, 95).Microseconds())
	p99 = float64(Percentile(latencies, 99).Microseconds())
	if len(latencies) > 0 {
		sorted := make([]time.Duration, len(latencies))
		copy(sorted, latencies)
		sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
		max = float64(sorted[len(sorted)-1].Microseconds())
	}
	return
}

// WriteJSON writes results to a JSON file.
func WriteJSON(results []BenchmarkResult, path string) error {
	data, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		return fmt.Errorf("marshaling results: %w", err)
	}
	return os.WriteFile(path, data, 0644)
}

// ToMarkdownTable converts results to a markdown table string.
func ToMarkdownTable(results []BenchmarkResult) string {
	if len(results) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("| Scenario | Index | Dim | Scale | Quant | QPS | P50 (us) | P99 (us) | Recall@10 | Memory (MB) |\n")
	sb.WriteString("|----------|-------|-----|-------|-------|-----|----------|----------|-----------|-------------|\n")

	for _, r := range results {
		quant := r.Quantizer
		if quant == "" {
			quant = "none"
		}
		sb.WriteString(fmt.Sprintf("| %s | %s | %d | %d | %s | %.0f | %.0f | %.0f | %.3f | %.1f |\n",
			r.Scenario, r.Index, r.Dimension, r.Scale, quant,
			r.QPS, r.P50LatencyUs, r.P99LatencyUs, r.Recall10, r.MemoryMB))
	}
	return sb.String()
}

// ComparisonTable generates a markdown table comparing DeepData against competitor baselines.
func ComparisonTable(deepdata []BenchmarkResult, baselines []CompetitorBaseline) string {
	var sb strings.Builder
	sb.WriteString("| System | Scenario | QPS | P99 (us) | Recall@10 | Memory (MB) | Source |\n")
	sb.WriteString("|--------|----------|-----|----------|-----------|-------------|--------|\n")

	for _, r := range deepdata {
		sb.WriteString(fmt.Sprintf("| DeepData | %s/%s/%dd/%dk | %.0f | %.0f | %.3f | %.1f | measured |\n",
			r.Scenario, r.Index, r.Dimension, r.Scale/1000,
			r.QPS, r.P99LatencyUs, r.Recall10, r.MemoryMB))
	}
	for _, b := range baselines {
		sb.WriteString(fmt.Sprintf("| %s | %s | %.0f | %.0f | %.3f | %.1f | [%s](%s) |\n",
			b.System, b.Scenario, b.QPS, b.P99LatencyUs, b.Recall10, b.MemoryMB,
			b.Source, b.URL))
	}
	return sb.String()
}
