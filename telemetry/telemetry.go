// Package telemetry provides OpenTelemetry-compatible tracing for VectorDB
package telemetry

import (
	"context"
	"net/http"
)

// Span represents a tracing span
type Span struct {
	name string
}

// End ends the span (no-op in stub)
func (s *Span) End() {}

// SetAttribute sets an attribute on the span (no-op in stub)
func (s *Span) SetAttribute(key string, value interface{}) {}

// RecordError records an error on the span (no-op in stub)
func RecordError(span *Span, err error) {}

// RecordOK marks the span as successful (no-op in stub)
func RecordOK(span *Span) {}

// RecordSearchResults records search result metrics (no-op in stub)
func RecordSearchResults(span *Span, count int, latencyMs int64) {}

// StartInsert starts a span for an insert operation
func StartInsert(ctx context.Context, collection string, id string) (context.Context, *Span) {
	return ctx, &Span{name: "insert"}
}

// StartSearch starts a span for a search operation
func StartSearch(ctx context.Context, collection string, topK int, mode string) (context.Context, *Span) {
	return ctx, &Span{name: "search"}
}

// StartDelete starts a span for a delete operation
func StartDelete(ctx context.Context, collection string, id string) (context.Context, *Span) {
	return ctx, &Span{name: "delete"}
}

// SetupSimple initializes telemetry with simple/default configuration
func SetupSimple() error {
	return nil
}

// Shutdown gracefully shuts down the telemetry system
func Shutdown(ctx context.Context) error {
	return nil
}

// HTTPMiddleware returns HTTP middleware for request tracing
func HTTPMiddleware() func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return next
	}
}
