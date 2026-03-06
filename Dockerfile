FROM golang:1.24-bookworm AS builder

WORKDIR /src

# Copy go.mod/sum first for layer caching
COPY go.mod go.sum ./
RUN go mod download

# Copy source
COPY . .

# Build vectordb binary (static, no CGO for portability)
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" \
    -o /out/vectordb ./vectordb/

# --- Runtime ---
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -r -s /bin/false -d /data vectordb && \
    mkdir -p /data && chown vectordb:vectordb /data

COPY --from=builder /out/vectordb /usr/local/bin/vectordb

USER vectordb
WORKDIR /data

# Default environment
ENV PORT=8080
ENV DATA_DIR=/data
ENV LOG_LEVEL=info

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

ENTRYPOINT ["vectordb"]
