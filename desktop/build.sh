#!/bin/bash
set -e
cd "$(dirname "$0")"

# Determine Rust target triple for sidecar naming
TARGET=$(rustc -vV | grep host | cut -d' ' -f2)
echo "==> Building Go server for target: $TARGET"

# Build Go binary into the sidecar binaries dir
cd ..
go build -o "desktop/src-tauri/binaries/deepdata-$TARGET" ./cmd/deepdata
cd desktop

# Install npm deps if needed
if [ ! -d node_modules ]; then
    echo "==> Installing npm dependencies..."
    npm install
fi

# Build Tauri app
echo "==> Building Tauri desktop app..."
npm run tauri build

echo "==> Done! Binary at: src-tauri/target/release/deepdata-desktop"
echo "==> Bundle at:  src-tauri/target/release/bundle/"
