package main

import (
	"fmt"
	"net/http"
	"sync"
	"time"

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

// RecordOperation records a completed operation
func (mc *MetricsCollector) RecordOperation(operation string, shardID int, duration time.Duration, err error) {
	status := "success"
	if err != nil {
		status = "error"
		mc.operationErrors.WithLabelValues(operation, fmt.Sprintf("%d", shardID), "unknown").Inc()
	}

	mc.operationsTotal.WithLabelValues(operation, fmt.Sprintf("%d", shardID), status).Inc()
	mc.operationDuration.WithLabelValues(operation, fmt.Sprintf("%d", shardID)).Observe(duration.Seconds())
}

// RecordQuery records query metrics
func (mc *MetricsCollector) RecordQuery(mode string, collections []string, duration time.Duration, resultCount int, shardsFanout int) {
	collectionsLabel := fmt.Sprintf("%d", len(collections))
	if len(collections) == 0 {
		collectionsLabel = "all"
	}

	mc.queryLatency.WithLabelValues(mode, collectionsLabel).Observe(duration.Seconds())
	mc.queryResultsTotal.WithLabelValues(mode).Observe(float64(resultCount))
	mc.queryShardsFanout.WithLabelValues(mode).Observe(float64(shardsFanout))
}

// UpdateShardHealth updates shard health metrics
func (mc *MetricsCollector) UpdateShardHealth(shardID int, nodeID string, role string, healthy bool, replicationLag int) {
	healthValue := 0.0
	if healthy {
		healthValue = 1.0
	}

	mc.shardHealthStatus.WithLabelValues(
		fmt.Sprintf("%d", shardID),
		nodeID,
		role,
	).Set(healthValue)

	if role == "replica" {
		mc.shardReplicationLag.WithLabelValues(
			fmt.Sprintf("%d", shardID),
			nodeID,
		).Set(float64(replicationLag))
	}
}

// HTTPMiddleware returns middleware that instruments HTTP requests
func (mc *MetricsCollector) HTTPMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		// Wrap response writer to capture status code
		wrapped := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}

		next.ServeHTTP(wrapped, r)

		duration := time.Since(start)
		mc.RecordHTTPRequest(r.Method, r.URL.Path, wrapped.statusCode, duration)
	})
}

// responseWriter wraps http.ResponseWriter to capture status code
type responseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

// RecordHTTPRequest records an HTTP request
func (mc *MetricsCollector) RecordHTTPRequest(method, endpoint string, status int, duration time.Duration) {
	mc.httpRequestsTotal.WithLabelValues(
		method,
		endpoint,
		fmt.Sprintf("%d", status),
	).Inc()

	mc.httpRequestDuration.WithLabelValues(method, endpoint).Observe(duration.Seconds())
}

// ===========================================================================================
// GLOBAL METRICS INSTANCE
// ===========================================================================================

var globalMetrics *MetricsCollector

// initMetrics initializes the global metrics collector
func initMetrics() {
	globalMetrics = NewMetricsCollector()
}

// withMetrics wraps an HTTP handler with metrics collection
func withMetrics(endpoint string, handler http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rw := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}

		handler(rw, r)

		duration := time.Since(start)
		if globalMetrics != nil {
			globalMetrics.RecordHTTPRequest(r.Method, endpoint, rw.statusCode, duration)
		}
	}
}
