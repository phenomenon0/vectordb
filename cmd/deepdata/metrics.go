package main

import (
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	vcollection "github.com/phenomenon0/vectordb/internal/collection"
)

// ===========================================================================================
// PROMETHEUS METRICS
// Comprehensive observability for distributed vectordb
// ===========================================================================================

// MetricsCollector collects and exposes Prometheus metrics
type MetricsCollector struct {
	mu       sync.RWMutex
	registry *prometheus.Registry

	// Per-tenant request accounting (HTTP + gRPC)
	tenantRequestsTotal   *prometheus.CounterVec
	tenantRequestDuration *prometheus.HistogramVec
	tenantDocuments       *prometheus.GaugeVec
	tenantBytes           *prometheus.GaugeVec
}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector() *MetricsCollector {
	registry := prometheus.NewRegistry()

	mc := &MetricsCollector{
		registry: registry,

		// ponytail: label cardinality = tenants x operations x codes; add a tenant allow-list if scrapes get slow
		tenantRequestsTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "vectordb_tenant_requests_total",
				Help: "Total requests per tenant",
			},
			[]string{"transport", "tenant", "operation", "code"},
		),

		tenantRequestDuration: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "vectordb_tenant_request_duration_seconds",
				Help:    "Request duration per tenant",
				Buckets: prometheus.DefBuckets,
			},
			[]string{"transport", "tenant", "operation"},
		),

		tenantDocuments: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "vectordb_tenant_documents",
				Help: "Document count per tenant",
			},
			[]string{"tenant"},
		),

		tenantBytes: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "vectordb_tenant_bytes",
				Help: "Storage bytes per tenant",
			},
			[]string{"tenant"},
		),
	}

	// Register all metrics
	registry.MustRegister(
		mc.tenantRequestsTotal,
		mc.tenantRequestDuration,
		mc.tenantDocuments,
		mc.tenantBytes,
	)

	return mc
}

// Handler returns the HTTP handler for exposing metrics
func (mc *MetricsCollector) Handler() http.Handler {
	return promhttp.HandlerFor(mc.registry, promhttp.HandlerOpts{})
}

// RecordTenantRequest records one completed request's outcome and latency
// for a tenant. Safe to call on a nil receiver (metrics are best-effort).
func (mc *MetricsCollector) RecordTenantRequest(transport, tenant, operation, code string, elapsed time.Duration) {
	if mc == nil {
		return
	}
	mc.tenantRequestsTotal.WithLabelValues(transport, tenant, operation, code).Inc()
	mc.tenantRequestDuration.WithLabelValues(transport, tenant, operation).Observe(elapsed.Seconds())
}

// RefreshTenantUsage replaces the tenant usage gauges with the current
// snapshot, so a tenant deleted since the last scrape doesn't linger.
func (mc *MetricsCollector) RefreshTenantUsage(infos []vcollection.TenantInfo) {
	if mc == nil {
		return
	}
	mc.tenantDocuments.Reset()
	mc.tenantBytes.Reset()
	for _, info := range infos {
		mc.tenantDocuments.WithLabelValues(info.TenantID).Set(float64(info.Usage.Documents))
		mc.tenantBytes.WithLabelValues(info.TenantID).Set(float64(info.Usage.Bytes))
	}
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

// ===========================================================================================
// GLOBAL METRICS INSTANCE
// ===========================================================================================

var globalMetrics *MetricsCollector

// initMetrics initializes the global metrics collector
func initMetrics() {
	globalMetrics = NewMetricsCollector()
}

// normalizeMetricsPath replaces dynamic path segments with placeholders to
// prevent unbounded Prometheus label cardinality.
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
		// /v3/tenants/{id}/collections/{name}/docs/{doc_id} → .../docs/:doc_id
		// "batch" is a literal sub-route (fixed set of operations), not a document ID.
		if i >= 3 && len(parts) > 2 && parts[i-1] == "docs" && part != "batch" {
			parts[i] = ":doc_id"
		}
	}
	return strings.Join(parts, "/")
}
