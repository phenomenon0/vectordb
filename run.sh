#!/bin/bash
set -a
source "$(dirname "$0")/.env"
set +a
exec go run ./cmd/deepdata/ "$@"
