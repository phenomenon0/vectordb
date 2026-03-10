module github.com/phenomenon0/vectordb

go 1.24.0

require (
	github.com/Neumenon/cowrie/go v0.0.0-20260306181650-7d62141ec1de
	github.com/coder/hnsw v0.6.1
	github.com/golang-jwt/jwt/v5 v5.3.0
	github.com/mattn/go-sqlite3 v1.14.32
	github.com/prometheus/client_golang v1.19.1
	github.com/sugarme/tokenizer v0.3.0
	github.com/yalue/onnxruntime_go v1.27.0
	go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp v0.66.0
	go.opentelemetry.io/otel v1.41.0
	go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp v1.41.0
	go.opentelemetry.io/otel/exporters/stdout/stdouttrace v1.41.0
	go.opentelemetry.io/otel/sdk v1.41.0
	go.opentelemetry.io/otel/trace v1.41.0
	golang.org/x/crypto v0.48.0
	golang.org/x/sys v0.41.0
)

require (
	github.com/Neumenon/shard/go v0.0.0-20260307220650-31a1f09ee1fc // indirect
	github.com/beorn7/perks v1.0.1 // indirect
	github.com/cenkalti/backoff/v5 v5.0.3 // indirect
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/chewxy/math32 v1.10.1 // indirect
	github.com/edsrzf/mmap-go v1.2.0 // indirect
	github.com/emirpasic/gods v1.18.1 // indirect
	github.com/felixge/httpsnoop v1.0.4 // indirect
	github.com/go-logr/logr v1.4.3 // indirect
	github.com/go-logr/stdr v1.2.2 // indirect
	github.com/google/uuid v1.6.0 // indirect
	github.com/grpc-ecosystem/grpc-gateway/v2 v2.28.0 // indirect
	github.com/klauspost/compress v1.18.4 // indirect
	github.com/mitchellh/colorstring v0.0.0-20190213212951-d06e56a500db // indirect
	github.com/patrickmn/go-cache v2.1.0+incompatible // indirect
	github.com/pierrec/lz4/v4 v4.1.26 // indirect
	github.com/prometheus/client_model v0.5.0 // indirect
	github.com/prometheus/common v0.48.0 // indirect
	github.com/prometheus/procfs v0.12.0 // indirect
	github.com/rivo/uniseg v0.4.7 // indirect
	github.com/schollz/progressbar/v2 v2.15.0 // indirect
	github.com/sugarme/regexpset v0.0.0-20200920021344-4d4ec8eaf93c // indirect
	github.com/viterin/partial v1.1.0 // indirect
	github.com/viterin/vek v0.4.2 // indirect
	go.opentelemetry.io/auto/sdk v1.2.1 // indirect
	go.opentelemetry.io/otel/exporters/otlp/otlptrace v1.41.0 // indirect
	go.opentelemetry.io/otel/metric v1.41.0 // indirect
	go.opentelemetry.io/proto/otlp v1.9.0 // indirect
	golang.org/x/exp v0.0.0-20240506185415-9bf2ced13842 // indirect
	golang.org/x/net v0.50.0
	golang.org/x/text v0.34.0 // indirect
	google.golang.org/genproto/googleapis/api v0.0.0-20260209200024-4cfbd4190f57 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20260209200024-4cfbd4190f57 // indirect
	google.golang.org/grpc v1.79.1 // indirect
	google.golang.org/protobuf v1.36.11 // indirect
)

replace github.com/coder/hnsw => ./internal/index/hnsw
