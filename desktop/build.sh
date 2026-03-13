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

# macOS: ad-hoc sign if no signing identity configured
if [[ "$OSTYPE" == "darwin"* ]]; then
    APP_BUNDLE=$(find src-tauri/target/release/bundle -name "*.app" -maxdepth 3 2>/dev/null | head -1)
    if [ -n "$APP_BUNDLE" ]; then
        echo "==> Ad-hoc signing macOS app bundle..."
        codesign --force --deep --sign - --entitlements src-tauri/entitlements.plist "$APP_BUNDLE"
        echo "==> Signed: $APP_BUNDLE"
        echo ""
        echo "    NOTE: This is an ad-hoc signature. macOS Gatekeeper may still"
        echo "    block it on first launch. To open: right-click → Open, or run:"
        echo "      xattr -cr \"$APP_BUNDLE\""
    fi

    DMG=$(find src-tauri/target/release/bundle/dmg -name "*.dmg" 2>/dev/null | head -1)
    if [ -n "$DMG" ]; then
        echo "==> Ad-hoc signing DMG..."
        codesign --force --sign - "$DMG"
        echo "==> Signed: $DMG"
        echo ""
        echo "    To allow opening the DMG on a fresh Mac:"
        echo "      xattr -cr \"$DMG\""
    fi
fi

echo ""
echo "==> Done! Binary at: src-tauri/target/release/deepdata-desktop"
echo "==> Bundle at:  src-tauri/target/release/bundle/"
