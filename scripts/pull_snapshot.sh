#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'EOF'
pull_snapshot.sh is unavailable in the single-node release candidate.

There is no supported use for this script or any online state-transfer
workflow. For backup or restore, stop DeepData and copy the whole configured
state root, or capture its whole storage volume. Never merge a backup into live
or non-empty state.
See docs/cookbook.md and docs/kubernetes.md.
EOF

exit 64
