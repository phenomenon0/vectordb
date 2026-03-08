# ============================================================================
# DeepData Vector Database — Multi-stage Docker Build
# ============================================================================

# --- Build stage ---
FROM golang:1.24-bookworm AS builder

WORKDIR /src

# Copy go.mod/sum first for layer caching
COPY go.mod go.sum ./
RUN go mod download

# Copy source
COPY . .

# Build with CGO enabled (required for go-sqlite3 cost tracker)
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && \
    CGO_ENABLED=1 GOOS=linux go build -trimpath -ldflags="-s -w" \
    -o /out/deepdata ./cmd/deepdata/

# --- Runtime stage ---
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -r -s /bin/false -d /data deepdata && \
    mkdir -p /data && chown deepdata:deepdata /data

COPY --from=builder /out/deepdata /usr/local/bin/deepdata

USER deepdata
WORKDIR /data

# Default environment
ENV PORT=8080
ENV VECTORDB_BASE_DIR=/data
ENV VECTORDB_MODE=local
ENV LOG_FORMAT=json
ENV USE_HASH_EMBEDDER=1
ENV VECTOR_CAPACITY=1000
ENV HYDRATION_COUNT=0

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:${PORT}/health || exit 1

ENTRYPOINT ["deepdata"]
