#!/usr/bin/env bash
# Simple follower pull script: fetch snapshot from a leader and import locally.
set -euo pipefail

LEADER_URL="${LEADER_URL:-http://localhost:8080}"
SNAPSHOT_PATH="${SNAPSHOT_PATH:-/tmp/vectordb-snapshot.gob}"
FOLLOWER_URL="${FOLLOWER_URL:-http://localhost:8080}"

echo "Fetching snapshot from ${LEADER_URL}/export ..."
curl -sSf "${LEADER_URL}/export" -o "${SNAPSHOT_PATH}"
echo "Importing snapshot into ${FOLLOWER_URL}/import ..."
curl -sSf -X POST --data-binary @"${SNAPSHOT_PATH}" "${FOLLOWER_URL}/import"
echo "Done."
