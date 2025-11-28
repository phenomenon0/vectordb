#!/bin/bash

# VectorDB Startup Script
# Usage: ./start.sh [options]

set -e

# Default values
PORT="${PORT:-8080}"
COMPACT_INTERVAL_MIN="${COMPACT_INTERVAL_MIN:-60}"
SNAPSHOT_INTERVAL_MIN="${SNAPSHOT_INTERVAL_MIN:-30}"
RATE_LIMIT_PER_SEC="${RATE_LIMIT_PER_SEC:-100}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if binary exists
if [ ! -f "./vectordb" ]; then
    print_error "vectordb binary not found in current directory"
    print_info "Build it with: go build -ldflags=\"-s -w\" -o vectordb ."
    exit 1
fi

# Make binary executable
chmod +x ./vectordb

# Create data directory
mkdir -p vectordb

print_info "Starting VectorDB..."
print_info "Configuration:"
echo "  - Port: $PORT"
echo "  - Compaction interval: $COMPACT_INTERVAL_MIN minutes"
echo "  - Snapshot interval: $SNAPSHOT_INTERVAL_MIN minutes"
echo "  - Rate limit: $RATE_LIMIT_PER_SEC req/sec per IP"
echo ""

# Export environment variables
export PORT
export COMPACT_INTERVAL_MIN
export SNAPSHOT_INTERVAL_MIN
export RATE_LIMIT_PER_SEC

# Start the server
print_info "Server starting on http://localhost:$PORT"
print_info "Press Ctrl+C to shutdown gracefully"
echo ""

./vectordb
