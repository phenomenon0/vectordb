package main

import (
	"net/http"
	"strings"
	"sync"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// ===========================================================================================
// PROMETHEUS METRICS
// Comprehensive observability for distributed vectordb
// ===========================================================================================

// MetricsCollector collects and exposes Prometheus metrics
type MetricsCollector struct {
	mu       sync.RWMutex
	registry *prometheus.Registry

	// Vector operations
	vectorsTotal        *prometheus.GaugeVec
	vectorsDeleted      *prometheus.GaugeVec
	operationsTotal     *prometheus.CounterVec
	operationDuration   *prometheus.HistogramVec
	operationErrors     *prometheus.CounterVec

	// Query performance
	queryLatency        *prometheus.HistogramVec
	queryResultsTotal   *prometheus.HistogramVec
	queryShardsFanout   *prometheus.HistogramVec

	// Shard health
	shardHealthStatus   *prometheus.GaugeVec
	shardReplicationLag *prometheus.GaugeVec
	shardNodeCount      *prometheus.GaugeVec

	// Failover
	failoverTotal       *prometheus.CounterVec
	failoverDuration    *prometheus.HistogramVec

	// HTTP
	httpRequestsTotal   *prometheus.CounterVec
	httpRequestDuration *prometheus.HistogramVec
}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector() *MetricsCollector {
	registry := prometheus.NewRegistry()

	mc := &MetricsCollector{
		registry: registry,

		// Vector metrics
		vectorsTotal: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "vectordb_vectors_total",
				Help: "Total number of vectors stored",
			},
			[]string{"shard_id", "collection", "node_id"},
		),

		vectorsDeleted: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "vectordb_vectors_deleted",
				Help: "Number of deleted vectors (tombstones)",
			},
			[]string{"shard_id", "collection", "node_id"},
		),

		operationsTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "vectordb_operations_total",
				Help: "Total number of operations",
			},
			[]string{"operation", "shard_id", "status"},
		),

		operationDuration: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "vectordb_operation_duration_seconds",
				Help:    "Duration of vectordb operations",
				Buckets: prometheus.ExponentialBuckets(0.001, 2, 15), // 1ms to ~32s
			},
			[]string{"operation", "shard_id"},
		),

		operationErrors: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "vectordb_operation_errors_total",
				Help: "Total number of operation errors",
			},
			[]string{"operation", "shard_id", "error_type"},
		),

		// Query metrics
		queryLatency: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "vectordb_query_duration_seconds",
				Help:    "Query latency in seconds",
				Buckets: prometheus.ExponentialBuckets(0.01, 2, 12), // 10ms to ~40s
			},
			[]string{"mode", "collections"},
		),

		queryResultsTotal: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "vectordb_query_results",
				Help:    "Number of results returned per query",
				Buckets: []float64{1, 5, 10, 20, 50, 100, 500, 1000},
			},
			[]string{"mode"},
		),

		queryShardsFanout: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "vectordb_query_shards_fanout",
				Help:    "Number of shards queried per request",
				Buckets: []float64{1, 2, 3, 5, 10, 20, 50, 100},
			},
			[]string{"mode"},
		),

		// Shard health metrics
		shardHealthStatus: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "vectordb_shard_health_status",
				Help: "Shard node health status (1=healthy, 0=unhealthy)",
			},
			[]string{"shard_id", "node_id", "role"},
		),

		shardReplicationLag: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "vectordb_shard_replication_lag_operations",
				Help: "Number of operations replica is behind primary",
			},
			[]string{"shard_id", "node_id"},
		),

		shardNodeCount: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "vectordb_shard_nodes",
				Help: "Number of nodes per shard",
			},
			[]string{"shard_id", "role"},
		),

		// Failover metrics
		failoverTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "vectordb_failover_total",
				Help: "Total number of failovers",
			},
			[]string{"shard_id", "status"},
		),

		failoverDuration: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "vectordb_failover_duration_seconds",
				Help:    "Duration of failover operations",
				Buckets: []float64{1, 5, 10, 30, 60, 120, 300},
			},
			[]string{"shard_id"},
		),

		// HTTP metrics
		httpRequestsTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "vectordb_http_requests_total",
				Help: "Total HTTP requests",
			},
			[]string{"method", "endpoint", "status"},
		),

		httpRequestDuration: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "vectordb_http_request_duration_seconds",
				Help:    "HTTP request duration",
				Buckets: prometheus.DefBuckets,
			},
			[]string{"method", "endpoint"},
		),
	}

	// Register all metrics
	registry.MustRegister(
		mc.vectorsTotal,
		mc.vectorsDeleted,
		mc.operationsTotal,
		mc.operationDuration,
		mc.operationErrors,
		mc.queryLatency,
		mc.queryResultsTotal,
		mc.queryShardsFanout,
		mc.shardHealthStatus,
		mc.shardReplicationLag,
		mc.shardNodeCount,
		mc.failoverTotal,
		mc.failoverDuration,
		mc.httpRequestsTotal,
		mc.httpRequestDuration,
	)

	return mc
}

// Handler returns the HTTP handler for exposing metrics
func (mc *MetricsCollector) Handler() http.Handler {
	return promhttp.HandlerFor(mc.registry, promhttp.HandlerOpts{})
}

// responseWriter wraps http.ResponseWriter to capture status code
type responseWriter struct {
	http.ResponseWriter
	statusCode int
}

// ===========================================================================================
// GLOBAL METRICS INSTANCE
// ===========================================================================================

var globalMetrics *MetricsCollector

// initMetrics initializes the global metrics collector
func initMetrics() {
	globalMetrics = NewMetricsCollector()
}

// normalizeMetricsPath is retained for contract_test.go coverage.
func normalizeMetricsPath(path string) string {
	// Fast path for v1 endpoints (no dynamic segments)
	if !strings.Contains(path, "/v2/") && !strings.Contains(path, "/v3/") {
		return path
	}

	parts := strings.Split(path, "/")
	for i, part := range parts {
		if part == "" {
			continue
		}
		// /v2/collections/{name} → /v2/collections/:name
		if i >= 3 && len(parts) > 2 && parts[i-1] == "collections" {
			parts[i] = ":name"
		}
		// /v3/tenants/{id}/... → /v3/tenants/:id/...
		if i >= 3 && len(parts) > 2 && parts[i-1] == "tenants" {
			parts[i] = ":id"
		}
	}
	return strings.Join(parts, "/")
}
