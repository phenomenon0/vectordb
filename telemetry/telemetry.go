// Package telemetry provides OpenTelemetry-compatible tracing for VectorDB.
//
// By default, telemetry is disabled (no-op). To enable, set OTEL_EXPORTER_OTLP_ENDPOINT
// environment variable to your collector endpoint (e.g., "http://jaeger:4318").
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

import (
	"context"
	"net/http"
	"os"
	"sync"
	"time"

	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/exporters/stdout/stdouttrace"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"
	"go.opentelemetry.io/otel/trace"
)

var (
	tracer         trace.Tracer
	tracerProvider *sdktrace.TracerProvider
	initialized    bool
	initMu         sync.Mutex
)

// Span wraps trace.Span with convenience methods
type Span struct {
	span trace.Span
}

// End ends the span
func (s *Span) End() {
	if s.span != nil {
		s.span.End()
	}
}

// SetAttribute sets an attribute on the span
func (s *Span) SetAttribute(key string, value interface{}) {
	if s.span == nil {
		return
	}
	switch v := value.(type) {
	case string:
		s.span.SetAttributes(attribute.String(key, v))
	case int:
		s.span.SetAttributes(attribute.Int(key, v))
	case int64:
		s.span.SetAttributes(attribute.Int64(key, v))
	case float64:
		s.span.SetAttributes(attribute.Float64(key, v))
	case bool:
		s.span.SetAttributes(attribute.Bool(key, v))
	}
}

// RecordError records an error on the span
func RecordError(span *Span, err error) {
	if span != nil && span.span != nil && err != nil {
		span.span.RecordError(err)
	}
}

// RecordOK marks the span as successful
func RecordOK(span *Span) {
	if span != nil && span.span != nil {
		span.span.SetAttributes(attribute.Bool("success", true))
	}
}

// RecordSearchResults records search result metrics
func RecordSearchResults(span *Span, count int, latencyMs int64) {
	if span != nil && span.span != nil {
		span.span.SetAttributes(
			attribute.Int("result_count", count),
			attribute.Int64("latency_ms", latencyMs),
		)
	}
}

// SetupSimple initializes telemetry with environment-based configuration.
//
// Environment variables:
//   - OTEL_EXPORTER_OTLP_ENDPOINT: OTLP collector endpoint (e.g., "http://jaeger:4318")
//   - OTEL_SERVICE_NAME: Service name (default: "vectordb")
//   - OTEL_TRACES_SAMPLER_ARG: Sampling ratio 0.0-1.0 (default: 1.0)
//   - OTEL_TRACE_STDOUT: If "1", also export to stdout (for debugging)
//
// If no OTLP endpoint is configured, telemetry is effectively disabled (no-op).
func SetupSimple() error {
	initMu.Lock()
	defer initMu.Unlock()

	if initialized {
		return nil
	}

	ctx := context.Background()

	// Get service name
	serviceName := os.Getenv("OTEL_SERVICE_NAME")
	if serviceName == "" {
		serviceName = "vectordb"
	}

	// Create resource with service info
	res, err := resource.New(ctx,
		resource.WithAttributes(
			semconv.ServiceName(serviceName),
			semconv.ServiceVersion("1.0.0"),
		),
		resource.WithHost(),
		resource.WithProcess(),
	)
	if err != nil {
		return err
	}

	// Collect exporters
	var exporters []sdktrace.SpanExporter

	// Check for OTLP endpoint
	otlpEndpoint := os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
	if otlpEndpoint != "" {
		// Configure OTLP exporter
		opts := []otlptracehttp.Option{
			otlptracehttp.WithEndpoint(stripProtocol(otlpEndpoint)),
		}

		// Check if insecure (http://)
		if len(otlpEndpoint) > 7 && otlpEndpoint[:7] == "http://" {
			opts = append(opts, otlptracehttp.WithInsecure())
		}

		exporter, err := otlptracehttp.New(ctx, opts...)
		if err != nil {
			return err
		}
		exporters = append(exporters, exporter)
	}

	// Check for stdout debugging
	if os.Getenv("OTEL_TRACE_STDOUT") == "1" {
		stdoutExporter, err := stdouttrace.New(stdouttrace.WithPrettyPrint())
		if err != nil {
			return err
		}
		exporters = append(exporters, stdoutExporter)
	}

	// If no exporters, use no-op provider
	if len(exporters) == 0 {
		tracer = otel.Tracer(serviceName)
		initialized = true
		return nil
	}

	// Create trace provider with all exporters
	var batchOpts []sdktrace.TracerProviderOption
	batchOpts = append(batchOpts, sdktrace.WithResource(res))

	for _, exp := range exporters {
		batchOpts = append(batchOpts, sdktrace.WithBatcher(exp,
			sdktrace.WithBatchTimeout(5*time.Second),
			sdktrace.WithMaxExportBatchSize(512),
		))
	}

	// Add sampler if configured
	if samplerArg := os.Getenv("OTEL_TRACES_SAMPLER_ARG"); samplerArg != "" {
		var ratio float64
		if _, err := parseFloat(samplerArg, &ratio); err == nil && ratio >= 0 && ratio <= 1 {
			batchOpts = append(batchOpts, sdktrace.WithSampler(
				sdktrace.ParentBased(sdktrace.TraceIDRatioBased(ratio)),
			))
		}
	}

	tracerProvider = sdktrace.NewTracerProvider(batchOpts...)

	// Register as global provider
	otel.SetTracerProvider(tracerProvider)
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{},
		propagation.Baggage{},
	))

	tracer = tracerProvider.Tracer(serviceName)
	initialized = true

	return nil
}

// Shutdown gracefully shuts down the telemetry system
func Shutdown(ctx context.Context) error {
	initMu.Lock()
	defer initMu.Unlock()

	if tracerProvider != nil {
		ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
		defer cancel()
		return tracerProvider.Shutdown(ctx)
	}
	return nil
}

// StartInsert starts a span for an insert operation
func StartInsert(ctx context.Context, collection string, id string) (context.Context, *Span) {
	if tracer == nil {
		return ctx, &Span{}
	}

	ctx, span := tracer.Start(ctx, "vectordb.insert",
		trace.WithSpanKind(trace.SpanKindServer),
		trace.WithAttributes(
			attribute.String("db.system", "vectordb"),
			attribute.String("db.operation", "insert"),
			attribute.String("db.collection", collection),
			attribute.String("db.document.id", id),
		),
	)

	return ctx, &Span{span: span}
}

// StartSearch starts a span for a search operation
func StartSearch(ctx context.Context, collection string, topK int, mode string) (context.Context, *Span) {
	if tracer == nil {
		return ctx, &Span{}
	}

	ctx, span := tracer.Start(ctx, "vectordb.search",
		trace.WithSpanKind(trace.SpanKindServer),
		trace.WithAttributes(
			attribute.String("db.system", "vectordb"),
			attribute.String("db.operation", "search"),
			attribute.String("db.collection", collection),
			attribute.Int("db.search.top_k", topK),
			attribute.String("db.search.mode", mode),
		),
	)

	return ctx, &Span{span: span}
}

// StartDelete starts a span for a delete operation
func StartDelete(ctx context.Context, collection string, id string) (context.Context, *Span) {
	if tracer == nil {
		return ctx, &Span{}
	}

	ctx, span := tracer.Start(ctx, "vectordb.delete",
		trace.WithSpanKind(trace.SpanKindServer),
		trace.WithAttributes(
			attribute.String("db.system", "vectordb"),
			attribute.String("db.operation", "delete"),
			attribute.String("db.collection", collection),
			attribute.String("db.document.id", id),
		),
	)

	return ctx, &Span{span: span}
}

// StartBatchInsert starts a span for a batch insert operation
func StartBatchInsert(ctx context.Context, collection string, count int) (context.Context, *Span) {
	if tracer == nil {
		return ctx, &Span{}
	}

	ctx, span := tracer.Start(ctx, "vectordb.batch_insert",
		trace.WithSpanKind(trace.SpanKindServer),
		trace.WithAttributes(
			attribute.String("db.system", "vectordb"),
			attribute.String("db.operation", "batch_insert"),
			attribute.String("db.collection", collection),
			attribute.Int("db.batch.size", count),
		),
	)

	return ctx, &Span{span: span}
}

// StartEmbed starts a span for an embedding operation
func StartEmbed(ctx context.Context, provider string, textLen int) (context.Context, *Span) {
	if tracer == nil {
		return ctx, &Span{}
	}

	ctx, span := tracer.Start(ctx, "vectordb.embed",
		trace.WithSpanKind(trace.SpanKindClient),
		trace.WithAttributes(
			attribute.String("db.system", "vectordb"),
			attribute.String("db.operation", "embed"),
			attribute.String("embedding.provider", provider),
			attribute.Int("embedding.text_length", textLen),
		),
	)

	return ctx, &Span{span: span}
}

// StartWALWrite starts a span for a WAL write operation
func StartWALWrite(ctx context.Context, entryType string, size int) (context.Context, *Span) {
	if tracer == nil {
		return ctx, &Span{}
	}

	ctx, span := tracer.Start(ctx, "vectordb.wal.write",
		trace.WithSpanKind(trace.SpanKindInternal),
		trace.WithAttributes(
			attribute.String("db.system", "vectordb"),
			attribute.String("wal.entry_type", entryType),
			attribute.Int("wal.entry_size", size),
		),
	)

	return ctx, &Span{span: span}
}

// StartIndexOperation starts a span for an index operation
func StartIndexOperation(ctx context.Context, indexType string, operation string) (context.Context, *Span) {
	if tracer == nil {
		return ctx, &Span{}
	}

	ctx, span := tracer.Start(ctx, "vectordb.index."+operation,
		trace.WithSpanKind(trace.SpanKindInternal),
		trace.WithAttributes(
			attribute.String("db.system", "vectordb"),
			attribute.String("index.type", indexType),
			attribute.String("index.operation", operation),
		),
	)

	return ctx, &Span{span: span}
}

// HTTPMiddleware returns HTTP middleware for request tracing
func HTTPMiddleware() func(http.Handler) http.Handler {
	// If telemetry not initialized, return pass-through
	if !initialized || os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT") == "" {
		return func(next http.Handler) http.Handler {
			return next
		}
	}

	return otelhttp.NewMiddleware("vectordb-http",
		otelhttp.WithSpanNameFormatter(func(operation string, r *http.Request) string {
			return r.Method + " " + r.URL.Path
		}),
	)
}

// stripProtocol removes http:// or https:// from endpoint
func stripProtocol(endpoint string) string {
	if len(endpoint) > 8 && endpoint[:8] == "https://" {
		return endpoint[8:]
	}
	if len(endpoint) > 7 && endpoint[:7] == "http://" {
		return endpoint[7:]
	}
	return endpoint
}

// parseFloat parses a string to float64
func parseFloat(s string, out *float64) (bool, error) {
	var f float64
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= '0' && c <= '9' {
			f = f*10 + float64(c-'0')
		} else if c == '.' {
			// Handle decimal
			decimal := 0.1
			for i++; i < len(s); i++ {
				c = s[i]
				if c >= '0' && c <= '9' {
					f += float64(c-'0') * decimal
					decimal *= 0.1
				}
			}
		}
	}
	*out = f
	return true, nil
}
