# DeepData Fix Plan

- [x] Fix `cmd/deepdata/server.go` query result integrity and pagination bugs; add targeted tests.
- [x] Fix `cmd/deepdata/main.go` write atomicity and collection-move correctness; add targeted tests.
- [x] Fix `internal/cluster` snapshot auth, WAL streaming/replay correctness, and follower lag accounting; add targeted tests.
- [x] Fix `internal/index` and `internal/hybrid` persistence/quantization defects called out in review; add targeted tests.
- [x] Fix `desktop/` build and API payload mismatches against the current Go server.
- [x] Run targeted verification for each area and summarize residual risk.

## Review Notes

- Highest risk is silent corruption or divergence: partial writes, broken pagination/result mapping, replica WAL loss, and persisted index state not matching runtime behavior.
- Desktop is untracked and isolated, so it can be fixed in parallel without touching server code.

## Verification

- `GOCACHE=/tmp/gocache go test -count=1 ./cmd/deepdata -run "TestHTTPQueryPaginationUsesPageToken|TestHTTPQueryRerankKeepsResponseFieldsAligned|TestAddRollsBackStateOnIndexFailure|TestAddRollsBackStateOnWALError|TestUpsertNewIDRollsBackStateOnWALError|TestUpsertMovingCollectionRemovesOldIndexEntry"` passed.
- `GOCACHE=/tmp/gocache go test -count=1 ./internal/index -run "Test(HNSWExportImport|IVFExportImport|IVFIndexWithFloat16Quantization|IVFIndexWithUint8Quantization|DiskANNExportImportSelfContainedSearchAndMetadata|DiskANNExportImportFloat16PreservesQuantizedVectors)"` passed.
- `GOCACHE=/tmp/gocache go test -count=1 ./internal/index -run "Test(FLATIndexWithFloat16Quantization|FLATIndexWithUint8Quantization|HNSWIndexWithFloat16Quantization|HNSWIndexWithUint8Quantization|HNSWExportImport|Float16Quantization|Uint8Quantization|ProductQuantization)"` passed.
- `GOCACHE=/tmp/gocache go test -count=1 ./internal/hybrid -run "TestFuseWeighted_(Normalization|SingletonResultSetContributes)"` passed.
- `bash -n desktop/build.sh`, `node --check desktop/src-tauri/src/inject.js`, and `cargo check --manifest-path desktop/src-tauri/Cargo.toml` passed.
- Added `internal/testutil.NewLoopbackServer` and switched listener-based tests to use it so restricted environments skip cleanly instead of panicking on `httptest.NewServer`.
- `GOCACHE=/tmp/gocache go test -count=1 -short ./internal/cluster` passed.
- `GOCACHE=/tmp/gocache go test -count=1 -short -run 'TestDiskANNCompactionBasic' ./internal/index` passed after fixing DiskANN compaction offset rebuilding.
- `GOCACHE=/tmp/gocache go test -count=1 -short -p 1 ./...` passed.
