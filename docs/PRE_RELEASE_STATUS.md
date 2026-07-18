# DeepData Pre-Release Status

**Last audited:** 2026-07-17
**Audited codebase:** `fa521cc`
**Verdict:** **Not release-ready.** The responsible near-term target is a **single-node release candidate**. Distributed/HA mode is still experimental and should be a separate milestone.

This is the overall product checklist. The completed [mega benchmark report](../benchmarks/results/mega/REPORT.md) covers only benchmark repair, not the full release.

## What Is Done

- Seventeen production-hardening commits added server timeouts, panic recovery, safe background shutdown, startup configuration validation, structured logging, working Prometheus metrics, smaller default search payloads, gRPC authentication, request IDs, sanitized server errors, and multiple concurrency/atomicity fixes.
- The legacy store received substantial WAL and snapshot repair: fsync before rename, WAL error surfacing, WAL rotation, restart coverage, range-index reconstruction, and safer final shutdown.
- CI definitions now cover Go tests/vet, a short race suite, web UI build/Playwright, four server cross-compiles, and desktop builds on Linux, macOS, and Windows.
- Docker, Compose, Helm, a Python SDK, a desktop wrapper, operational docs, smoke tests, and VDB correctness/soak tooling exist.
- The corrected VDB benchmark contains **108/108 cells**, zero failed/pending cells, and **7/7** passing failure-mode probes.
- Current local verification: the short Go race suite passes; Python has **75 passing tests and 5 integration skips**.

## Release Blockers

| Priority | Gate | Current gap | Exit condition |
|---|---|---|---|
| P0 | Data durability and recovery | Default snapshots omit ownership/index state; V2/V3/gRPC writes have no crash WAL/checkpoint; corrupt snapshots can boot empty and later overwrite recoverable state. | Every advertised API survives SIGKILL/power loss; all fields and index types round-trip; corrupt state fails closed or enters an explicit recovery path; upgrade/crash tests pass. |
| P0 | Authorization and security surface | Authentication exists, but V2/V3/gRPC do not consistently enforce permissions and collection scopes. Feedback/extraction routes bypass auth. Documented TLS, encryption-at-rest, and audit logging are not wired into server startup. | One policy layer covers HTTP and gRPC; cross-tenant/read-only tests pass; all sensitive routes are guarded; security features are wired and tested or removed from the release claims. |
| P0 | Green, trustworthy CI | The latest GitHub main run failed Go, race, and Playwright jobs. CGO-disabled vet is locally reproducible as broken; normal Go CI includes 10M/50M/100M scale tests; Playwright does not use the binary CI builds. | A new run on the release SHA is green, with scale tests separated from PR CI and the UI starting the intended freshly built binary. |
| P0 | Legal and release identity | No `LICENSE` file exists; component versions disagree; historical tags mix DeepData with Atlas Runtime; the current hardening is newer than the published v1.1.0 release. | Add the intended license, choose one version/source of truth, namespace or clean future tags, update changelog/upgrade notes, and release only from the reviewed SHA. |
| P0 | Python distribution | `pip install deepdata` currently resolves to an unrelated 2020 package on PyPI, while the README advertises that command. Python tests are not in CI, and strict SDK mypy reports 16 `no-any-return` errors. | Choose an available package name or obtain the existing name; fix the strict typing gate; update imports/docs as needed; build, install, test, and publish through trusted CI. |
| P0 | Deployment parity | Docker/Compose/Helm do not expose advertised gRPC; Helm references a missing auth Secret; authenticated health checks target `/health` instead of the public probe; static CGO-off binaries lose SQLite cost tracking. | Docker and Helm smoke tests prove HTTP, gRPC, auth, probes, persistence, shutdown, and artifact feature parity using pinned images. |
| P0 | Honest supported scope | README claims production cluster scaling, replication, TLS, encryption, audit logging, and other behaviors that source/runtime label experimental or do not wire up. | Publish a precise support matrix. For the first RC, label distributed mode experimental and remove unsupported claims, or complete and prove those features. |
| P0 | Release-candidate proof | Existing smoke/soak artifacts predate the latest hardening; no exact-SHA release rehearsal covers upgrades and every shipped artifact. | Archive an exact-SHA matrix: unit/race, restart/crash, UI, Python integration, Docker, Helm, cross-platform binaries, upgrade from v1.1.0, soak, security scan, and fresh benchmark provenance. |

## Highest-Risk Technical Work

### 1. Fix persistence before adding features

- `storage.Default()` selects `cowrie-zstd`, but its codec does not serialize `TenantID` or named `Indexes`: [`internal/storage/format.go`](../internal/storage/format.go#L66), [`internal/storage/cowrie.go`](../internal/storage/cowrie.go#L52).
- V2/V3/gRPC collection state loads at startup and is only saved during graceful shutdown; there is no collection WAL/autosave path: [`cmd/deepdata/server.go`](../cmd/deepdata/server.go#L2090), [`cmd/deepdata/main.go`](../cmd/deepdata/main.go#L2588), [`internal/collection/manager.go`](../internal/collection/manager.go#L12).
- An unreadable legacy snapshot can fall back to a fresh store instead of stopping or quarantining state: [`cmd/deepdata/main.go`](../cmd/deepdata/main.go#L897).
- Legacy snapshots do not persist index type reliably, so non-HNSW configurations are not restart-safe.

### 2. Finish authorization, not just authentication

- The common guard authenticates and injects tenant context, while V2 handlers and gRPC methods operate without consistent permission/collection checks: [`cmd/deepdata/server.go`](../cmd/deepdata/server.go#L150), [`cmd/deepdata/collection_http.go`](../cmd/deepdata/collection_http.go#L125), [`cmd/deepdata/collection_grpc.go`](../cmd/deepdata/collection_grpc.go#L20).
- V3 checks tenant identity but does not enforce read versus write permission for each operation: [`cmd/deepdata/collection_http.go`](../cmd/deepdata/collection_http.go#L1016).
- Feedback and extraction endpoints are registered outside the guard: [`cmd/deepdata/server.go`](../cmd/deepdata/server.go#L2098).
- TLS, encryption, and audit packages exist, but the production startup path still uses plaintext listeners and direct storage.

### 3. Make CI describe the real product

- Latest remote run: [GitHub Actions run 24375610647](https://github.com/phenomenon0/vectordb/actions/runs/24375610647) — Go, race, and UI jobs failed; build/cross-compile/desktop jobs passed.
- `CGO_ENABLED=0 go vet ./...` fails because an unconditional SIMD test imports a CGO-only package: [`internal/index/simd/bench_cref_test.go`](../internal/index/simd/bench_cref_test.go#L1).
- The normal job runs all tests without `-short`, including scale tests intended for 10M, 50M, and 100M vectors: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml#L14), [`cmd/deepdata/benchmark_test.go`](../cmd/deepdata/benchmark_test.go#L354).
- CI builds `tests/ui/deepdata-test`, but Playwright starts `../../deepdata-server`: [`.github/workflows/ci.yml`](../.github/workflows/ci.yml#L76), [`tests/ui/playwright.config.ts`](../tests/ui/playwright.config.ts#L22).
- Python unit tests pass locally, but strict mypy reports 16 `no-any-return` errors; neither check has a workflow job.

### 4. Repair the release surface

- README claims MIT but links a missing license: [`README.md`](../README.md#L6).
- Server/web/Helm use `1.0.0` or `1.1.0` concepts while Python/desktop/chart use `0.1.0`; the changelog does not describe the current release candidate.
- The published [DeepData v1.1.0 release](https://github.com/phenomenon0/vectordb/releases/tag/v1.1.0) points to `add91ea`, before the April production-hardening series.
- The PyPI name [`deepdata`](https://pypi.org/project/deepdata/) belongs to an unrelated project; the advertised install command installs the wrong software.
- There is no tag-driven workflow for GitHub Release assets, container publishing, Python publishing, Helm packaging, checksums, SBOM, signatures, or provenance.

## Recommended Release Scope

Ship the first candidate as:

- **Single-node only**.
- HTTP and gRPC only after the same durability and authorization rules cover both.
- A clearly enumerated set of restart-safe index types.
- Distributed/HA, unsupported index variants, and unwired security features explicitly marked experimental or excluded.
- One pinned Docker image and signed/checksummed platform binaries built from the same commit.

Treat production cluster parity as a later milestone; current cluster code itself warns that quorum, snapshot catch-up, and leader-election safety are incomplete: [`internal/cluster/distributed.go`](../internal/cluster/distributed.go#L102).

## Release Exit Checklist

- [ ] Fix default snapshot round-trip and corrupt-state recovery.
- [ ] Add crash durability for V2/V3/gRPC mutations.
- [ ] Enforce permissions, collection scopes, and tenant isolation on every protocol and route.
- [ ] Wire or de-scope TLS, encryption-at-rest, and audit logging.
- [ ] Make all required CI jobs green on the candidate SHA.
- [ ] Add `LICENSE`, unify versions/names/tags, and resolve the PyPI name collision.
- [ ] Fix and smoke-test Docker/Compose/Helm with auth, gRPC, probes, persistence, and shutdown.
- [ ] Update README, security docs, changelog, supported-feature matrix, and upgrade notes.
- [ ] Run and archive the exact-SHA release evidence matrix.
- [ ] Publish the RC through a reproducible release workflow.

## Later Hardening

- Branch protection, `SECURITY.md`, dependency/secret/image scanning, SBOM, signatures, and provenance.
- Fresh 108-cell benchmark run with uniform timestamps and apples-to-apples DeepData transport settings.
- Longer mixed-load soak, restart-under-load, memory-drift, and chaos testing.
- Real macOS/Windows signing if the desktop app ships.
- Distributed quorum, snapshot catch-up, failover, fencing, and rebalancing proof.
