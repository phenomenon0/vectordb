// Package telemetry provides OpenTelemetry-compatible tracing for VectorDB.
//
// By default, telemetry is disabled (no-op). To enable, set OTEL_EXPORTER_OTLP_ENDPOINT
// environment variable to your collector endpoint (e.g., "http://jaeger:4318").
//
// Tower layer: L3 transports — the HTTP middleware, spans and Prometheus
// metrics the server surfaces are wrapped in.
//
// Example usage:
//
//	// At startup
//	if err := telemetry.SetupSimple(); err != nil {
//	    log.Printf("Warning: telemetry setup failed: %v", err)
//	}
//	defer telemetry.Shutdown(context.Background())
//
//	// In handlers
//	ctx, span := telemetry.StartSearch(ctx, collection, topK, "vector")
//	defer span.End()
package telemetry
