# ============================================================================
# DeepData Vector Database — Multi-stage Docker Build
# ============================================================================

# --- Build stage ---
FROM golang:1.24-bookworm AS builder

WORKDIR /src

# Copy go.mod/sum first for layer caching
COPY go.mod go.sum ./
# go.mod replaces github.com/coder/hnsw with this repository-local module, so
# its module metadata must exist before `go mod download` can resolve the graph.
COPY internal/index/hnsw/go.mod internal/index/hnsw/go.sum ./internal/index/hnsw/
RUN go mod download

# Copy only packages needed by the production server. Keeping this allowlist
# prevents local caches, SDK build trees, and test harnesses from entering a
# build layer if a future ignore rule regresses.
COPY api ./api
COPY cmd ./cmd
COPY internal ./internal

# Build with CGO enabled (required for go-sqlite3 cost tracker)
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && \
    CGO_ENABLED=1 GOOS=linux go build -trimpath -ldflags="-s -w" \
    -o /out/deepdata ./cmd/deepdata/

# --- Runtime stage ---
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Keep this identity stable: Kubernetes security contexts and volume ownership
# use the same numeric UID/GID. Numeric IDs also avoid name-service ambiguity.
RUN groupadd --gid 10001 deepdata && \
    useradd --uid 10001 --gid 10001 --no-create-home \
      --home-dir /data --shell /usr/sbin/nologin deepdata && \
    mkdir -p /data && chown 10001:10001 /data

COPY --from=builder /out/deepdata /usr/local/bin/deepdata

USER 10001:10001
WORKDIR /data

# Default environment
ENV PORT=8080
ENV GRPC_PORT=50051
ENV VECTORDB_BASE_DIR=/data
ENV VECTORDB_DATA_DIR=local
ENV VECTORDB_MODE=local
ENV LOG_FORMAT=json

EXPOSE 8080 50051

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:${PORT}/livez || exit 1

ENTRYPOINT ["deepdata"]
