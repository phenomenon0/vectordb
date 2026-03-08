package telemetry

import (
	"github.com/prometheus/client_golang/prometheus"
)

// Prometheus metrics for DeepData vector database.
// All metrics use the "deepdata_" prefix and are registered on the default
// prometheus registry, so they are automatically served by promhttp.Handler().

var (
	// VectorsTotal tracks the total number of vectors stored per collection.
	VectorsTotal = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "deepdata_vectors_total",
			Help: "Total number of vectors stored",
		},
		[]string{"collection"},
	)

	// SearchRequestsTotal counts search requests by index type.
	SearchRequestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "deepdata_search_requests_total",
			Help: "Total number of search requests",
		},
		[]string{"index_type"},
	)

	// SearchDurationSeconds records search latency by index type.
	SearchDurationSeconds = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "deepdata_search_duration_seconds",
			Help:    "Search request latency in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"index_type"},
	)

	// InsertRequestsTotal counts insert requests by index type.
	InsertRequestsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "deepdata_insert_requests_total",
			Help: "Total number of insert requests",
		},
		[]string{"index_type"},
	)

	// InsertDurationSeconds records insert latency by index type.
	InsertDurationSeconds = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "deepdata_insert_duration_seconds",
			Help:    "Insert request latency in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"index_type"},
	)

	// MemoryBytes tracks index memory usage.
	MemoryBytes = prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "deepdata_memory_bytes",
			Help: "Index memory usage in bytes",
		},
	)

	// IndexBuildDurationSeconds records index build time.
	IndexBuildDurationSeconds = prometheus.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "deepdata_index_build_duration_seconds",
			Help:    "Index build duration in seconds",
			Buckets: prometheus.ExponentialBuckets(0.01, 2, 15), // 10ms to ~160s
		},
	)
)

func init() {
	prometheus.MustRegister(
		VectorsTotal,
		SearchRequestsTotal,
		SearchDurationSeconds,
		InsertRequestsTotal,
		InsertDurationSeconds,
		MemoryBytes,
		IndexBuildDurationSeconds,
	)
}
