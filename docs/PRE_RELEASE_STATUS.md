# DeepData Pre-Release Status

**Last reconciled:** 2026-08-07

**Candidate state:** Product code at `a99fe53`
(`a99fe539cee6ecb6a6a75731382fec91f2f61a86`) on
`gnhf/i-want-you-to-mnake-26a28a`, superseding the earlier frozen `043ad5d` /
`2e82859` / `14d1442` / `d5b2d3a`.
The newest product changes add **caller-supplied-ID upsert and get-document** across the
durable engine, mirrored V3 gRPC, canonical HTTP, and the Python SDK (see **Upsert /
Get-Document Batch** below); the Go and Python exact-SHA gates have been re-run green at
`a99fe53`.

**Technical verdict:** **Prior blocking finding resolved; remaining gates are owner-gated.**
The narrow single-node RC implementation is complete and the local matrix is green,
**including** the long-running memory-drift gate, which previously failed and is now
**resolved by a code fix that restores restart-reclaim** (see **Resolved Finding:
memory-drift** below). External CI is green on the pre-hardening SHA (PR #4 run
`29977008196`, all 10 required checks); the `a99fe53` feature batch is additionally
supersedes-the-hardening changes and needs a fresh external CI run. The chart
network-isolation gate is now **implemented and validated** (default-deny
`NetworkPolicy`). What remains before a tag is not a correctness failure but the
owner-gated item: the root `LICENSE` / copyright decision.

**Publication verdict:** **Not authorized and legally gated.** No root `LICENSE` exists
because the copyright holder and license choice require an external decision. Publication,
tag creation, registry uploads, signing, and use of private credentials are separate from
technical candidate preparation and from the authorized branch push.

This report covers the single-node candidate defined in
[`tasks/todo.md`](../tasks/todo.md). The completed
[mega benchmark repair](../benchmarks/results/mega/REPORT.md) remains historical evidence;
it is not a substitute for final candidate correctness or soak results.

## Candidate Contract

The candidate is a persistent Linux amd64, headless, single-node server with:

- tenant-aware HTTP V3 and mirrored `deepdata.v3.DeepData` gRPC;
- create/delete collection, insert, batch insert, upsert-by-ID, delete-document, and
  get-document mutations;
- collection get/list plus dense, sparse/BM25, and hybrid search;
- caller-provided vectors with HNSW, Flat, and inverted sparse indexes;
- static bearer or HS256 JWT authentication with read, write, collection-admin, and
  server-admin authorization;
- one deliberately small `deepdata-client` Python distribution;
- a hardened direct container, Compose contract, and single-replica Helm chart; and
- plaintext listeners behind an operator-managed trusted TLS proxy/ingress and encrypted
  disk/PVC boundary.

The root and V2 mutation APIs, historical V1 gRPC service, advanced/LLM routes, online
snapshot import, non-Linux persistent runtime, distributed/HA, DiskANN/IVF/PQ, built-in
TLS, built-in encryption at rest, and compliance-grade audit logging are not RC features.
The web UI and non-Linux cross-compiles are informational compile/build checks only.

## Implemented and Demonstrated Before Freeze

These are implemented and have preliminary local evidence. They must still be rerun from
the frozen candidate because security-sensitive dependencies and the Go toolchain changed
after some earlier passes.

### Durability and recovery

- All five advertised HTTP and gRPC mutations enter the same collection journal/apply
  boundary; reads observe an apply barrier and durability faults gate both transports.
- The journal defines sequence, fsync acknowledgement, replay, checkpoint, rotation,
  cleanup, partial-tail repair, and fault-latching behavior.
- Subprocess tests cover repeated SIGKILL/restart, acknowledged writes, all five mutations,
  checkpoint recovery, exactly-once replay, corruption, and lifetime locking.
- Snapshot state preserves tenant, collection, schema, documents, vectors, metadata,
  supported indexes, and recovery sequence information.
- Corrupt state and raw/unified legacy root/V2 artifacts fail closed without being rewritten.
  Frozen canonical V1 journal records remain replayable, including batches above current
  admission limits.
- Legacy root/V2 data does not receive an unsafe implicit migration. The documented path is
  stopped backup followed by explicit offline export/import and semantic verification.

### Authorization and operational safety

- One policy layer enforces tenant, optional collection, and read/write/admin claims across
  canonical HTTP and gRPC. Tests cover cross-tenant, cross-collection, read-only, expired,
  malformed, missing, and wrong-algorithm credentials.
- Bearer tokens are accepted only from the authorization header, never URL parameters.
- Production startup requires one strong credential. Explicit insecure development mode is
  loopback-only.
- Failed authentication uses a bounded shared HTTP/gRPC peer limiter. Tenant rate limiting,
  tenant/collection caps, schema/dimension/payload bounds, and response budgets are shared
  across the canonical surface.
- Unsupported legacy and advanced handlers are unreachable in RC startup: feedback,
  extraction, and V2 collection registration are gated off, and every remaining
  non-canonical route is rejected by the outermost canonical-surface allowlist before
  any other middleware runs (negative tests cover root, V2, and advanced paths).
  Health, liveness, and readiness remain usable when authentication is enabled.
- Secret scanning passed on a candidate source copy; static-analysis findings were reviewed
  for the supported surface. Both scans require final-tree reruns.

### SDK, CI, deployment, and release identity

- The Python package exposes only sync/async lifecycle methods plus canonical tenant V3
  clients. Preliminary unit, strict typing, package build, wheel install, and authenticated
  restart integration gates passed.
- Required Go CI uses a short supported-package matrix and a bounded race matrix. Generated
  protobuf and version-parity checks are pinned; experimental packages are non-gating.
- Direct-container and Compose contracts exercise hardened identity/filesystem settings,
  probes, authenticated HTTP and gRPC, persistent replacement, cleanup, and SIGTERM.
- Helm requires Linux amd64, one replica, a digest-pinned image, verified POSIX storage, and
  an existing Secret. A local kind rehearsal covered install, authenticated HTTP/gRPC, PVC
  replacement, upgrade, rollback, and graceful uninstall before candidate freeze.
- Version `0.2.0-rc.1` comes from `internal/releaseinfo/version.txt` and maps to Python
  `0.2.0rc1`. The candidate Python distribution is `deepdata-client`.
- The tag/manual release workflow builds unsigned Linux amd64, Python, Helm, container,
  checksum, SBOM, and provenance artifacts. Its publication job separately requires the
  exact tag, a root `LICENSE`, credentials, explicit dispatch, and scoped permissions.

## Remaining Technical Candidate Gates

| Priority | Gate | Status at `a99fe53` |
|---|---|---|
| P0 | Final dependency and toolchain tree | **Done.** vet/storage/short/race **re-run and PASS bound to `a99fe53`** (check receipts); python unit/mypy/build **re-run and PASS bound to `a99fe53`**; benchmark-unit, ui-build carry forward from `d5b2d3a` (byte-identical inputs — no ui product change; python product did change, hence python re-run). |
| P0 | Exact-SHA security and artifact proof | **Carried forward from `d5b2d3a`.** All Linux amd64 artifacts built, smoked (version `0.2.0-rc.1`, mutate+search+graceful stop), checksummed, with source+image SPDX SBOMs; security-scan results recorded in `STATE.json`. The `a99fe53` feature batch does not alter the dependency tree or scanned surface; artifact/SBOM/scan re-run is a mechanical rebuild left with external CI. |
| P1 | Final deployment parity | **Carried forward from `d5b2d3a`.** Direct container, Compose, and live kind/Helm lifecycle (install, authenticated HTTP+gRPC, PVC persistence across pod replacement, upgrade, rollback, graceful uninstall) PASS with a digest-pinned image. The feature batch touches in-process collection/auth/search paths, not deployment/lifecycle paths. |
| P1 | Operational fault and migration rehearsal | **Carried forward from `d5b2d3a`.** Whole-root backup/restore + process-level disk-full and permission-denied behavior PASS; explicit legacy export/import semantic rehearsal PASS. |
| P1 | Long-running correctness | **PASS.** Soak recall (0.9815), flat-exact (100/100), restart-under-load (5× SIGKILL clean), and count-parity carry forward from `d5b2d3a`; **memory-drift is now RESOLVED** at `14d1442` and proven reclaimed by an A/B control — see Resolved Finding below. |
| P1 | Network isolation contract | **Done.** The chart renders a default-deny `NetworkPolicy` (ingress only on the advertised HTTP/gRPC ports, egress deny-all with an explicit telemetry CIDR or render-refusal, operator opt-out). Validated by `helm lint --strict` and the `test-deployment-manifests` contract; non-enforcing CNIs are documented as an operator-managed requirement. The chart change is confined to `deploy/helm/**` and needs a fresh external CI run on a new candidate SHA. |
| External | Remote CI | **Pending on `a99fe53`.** All 10 required checks were green on PR #4 (run `29977008196`) over branch head `f5d4582` (product == `14d1442`) — Linux RC Go + race + container/Compose/Helm contracts, canonical Python client, 5-platform compile proofs, experimental source. A fresh run over `a99fe53` is queued offline because the proto/collection/gRPC/HTTP/Python surfaces changed. |

The local matrix is green at `a99fe53`: the Go gates (vet/storage/short/race) and the
python gates (unit/mypy/build) were re-run against the feature batch, the memory-drift
finding is resolved, and the remaining gates (deployment, artifact/SBOM, security scans)
carry forward because the product changes are confined to `api/proto`, `cmd/deepdata`,
`internal/collection`, and `sdk/python` (no ui/helm/dependency input changes). The chart
NetworkPolicy / operator network-isolation gate is now implemented and validated locally; a
fresh external CI run on `a99fe53` is queued offline.

## Upsert / Get-Document Batch (a99fe53, 2026-08-07)

A single-semantic-commit feature pass on top of the 043ad5d hardening batch, adding the
two missing document operations to the canonical surface:

- **Caller-supplied-ID upsert (`PUT /docs/{id}` / gRPC `Upsert`):** replaces an existing
  document atomically under the collection write lock — no count inflation, live dense and
  sparse postings evicted then re-added, HNSW tombstones resurrected, and the scan cursor
  preserved by threading the prepared `nextID` through the replace path.
- **Durable journaling:** `upsert_document` is a new journal mutation with encode/decode,
  replay prepare/apply, and a shared-barrier `getDocument` read, so upserts survive
  restart with exactly-once replay (focused crash/replay + restart-persistence tests).
- **Get-document (`GET /docs/{id}` / gRPC `GetDoc`):** a shared-barrier single-document
  read; 404 when the id does not exist. Batch and legacy routes are untouched.
- **Surface:** `deepdata.v3` proto regenerated (9 -> 11 unary RPCs), canonical surface test
  updated, HTTP adds numeric-id PUT/GET with 404 on non-numeric segments.
- **Python SDK:** sync/async tenants gain `upsert(id=..., vectors=..., metadata=...)` and
  `get_document(...)` with strict typing and live integration coverage (upsert, get,
  restart persistence).

Verification for this batch: `go build ./...`, `go vet`, the full `cmd/deepdata` suite,
`-race` over `internal/collection`, and python unit (57 pass) + mypy + build all pass;
official check-runner receipts (`go-storage`, `go-vet-cgo0`, `go-short`, `go-race`,
`python-unit`, `python-mypy`, `python-build`) are bound to `a99fe53`.

## Hardening Batch (043ad5d, 2026-08-07)

A single-semantic-commit hardening pass on top of the network-isolation gate, from an
adversarial re-review of the in-process surfaces:

- **Strict dense-vector decode:** `decodeDenseVectorFast` now rejects elements whose value
  is finite outside `float32` range (which previously became `+Inf`/`-Inf` and poisoned
  distance/similarity), requires the closing `]`, and rejects trailing bytes after the
  array (a second bracketed array or garbage could previously be silently ignored).
- **UTF-8-safe request-ID truncation:** a client-supplied `X-Request-ID` longer than
  128 bytes is truncated on a rune boundary, so the echoed header and structured log
  field stay valid UTF-8 instead of a mid-rune cut.
- **Auth-gated `/metrics`:** the Prometheus surface is now behind the same
  `REQUIRE_AUTH` guard as the API routes (operation/volume detail no longer leaks when
  auth is enabled); probes stay public.
- **FLAT count correctness:** re-adding an existing ID (including a tombstone
  resurrection) replaces the vector without inflating `Count`.
- **Atomic HNSW batches:** `BatchAdd`/`BatchAddNoCopy` defer metadata registration and
  graph insertion until after the graph is fully built, and roll back newly registered
  nodes / resurrected tombstones / the count on a cancelled context or storage error —
  a failed batch leaves no partial state and every ID in it is re-addable.
- **Sparse overwrite correctness:** the cosine norm is recomputed from the stored vector
  on every `Add` (an all-zero overwrite no longer keeps a stale norm), and
  inverted-list postings emptied by an overwrite are dropped instead of lingering.

All of these are covered by new focused unit tests. Verification for this batch:
`go build ./...`, `go vet`, full `go test -short ./...`, and `-race` over
`internal/index`, `cmd/deepdata`, `internal/collection` all pass; official
check-runner receipts (`go-storage`, `go-vet-cgo0`, `go-short`, `go-race`) are bound
to `043ad5d`. One test-noise flake
(`benchmarks/review/TestDevOpsSREReview`, P99/P50 latency-ratio) was observed once and
passes on the controlled rerun (100%, 6/6); both logs are retained.

## Resolved Finding: memory-drift (canonical HNSW delete reclamation)

**Verdict: RESOLVED at `14d1442`** (was FAIL at `d5b2d3a`). The finding was real, root-caused,
and independently (adversarially) verified: canonical HNSW `Delete()` is soft-only and, in
canonical mode, no online compaction is reachable, so nothing reclaimed tombstones online.
Critically, **restart did not reclaim either**: `Import()` re-added every tombstoned vector
to the rebuilt graph, so both RSS and the on-disk snapshot grew ~linearly with cumulative
deletes with no remedy short of dropping and recreating the collection.

**Fix.** `Import()` now skips version-2 deleted entries, mirroring `Compact()`'s
continue-on-deleted, so a restart rebuilds a clean graph and reclaims both RAM and the
on-disk snapshot. The durable store reconciles on `Active = Count − Deleted`, which is
unchanged by dropping tombstones, so the fix is a pure reclamation. `TestHNSWExportImport`
was rewritten to assert the reclaimed intent (`Deleted == 0`, `Count == Active` after
import) rather than lock in the old tombstone-preservation behavior.

**Verification (A/B control).** With an identical delete+reinsert churn workload and a
graceful restart against a 76.6 MB tombstoned snapshot on disk:

- **Old binary (`d5b2d3a`, no fix):** reload took 50 s and the recovery checkpoint rewrote
  the snapshot **byte-identically** (76,625,482 → 76,625,482 B) — no reclamation.
- **Fixed binary (`14d1442`):** reload took ~8 s and the snapshot shrank **73.4 MB → 24.1 MB
  (−67 %)**, with correctness intact (live ids searchable, deleted ids stay gone, re-insert
  findable). The fix also cures a restart-latency pathology that scaled with the cumulative
  delete count.

The Go exact-SHA gates (vet/storage/short/race) were re-run and PASS at `14d1442`, with the
race detector clean over the changed code. Evidence:
`.deepdata-run/rehearsals/memdrift-old/old_reload_receipt.json` (A/B control),
`.deepdata-run/rehearsals/memdrift-fix/restart_reclaim_receipt.json` (fixed binary
end-to-end), and `.deepdata-run/checks/go-*/receipt.json` bound to `14d1442`.

Online (single-process, no-restart) delete churn still drifts by design — reclamation is a
restart/checkpoint-time operation, not an online compactor — but it is now bounded: every
graceful checkpoint and restart folds and reclaims the tombstones. A future online
canonical compactor would remove even the transient online drift; it is not required for
the RC.

No item in this table is waived by a preliminary pass from a dirty worktree.

### Go toolchain pin

`go.mod` requires `go >= 1.25.12`, but the host `/usr/lib/golang/bin/go` is a custom build
whose default `GOTOOLCHAIN=local` is 1.25.5, which made plain check-runner invocations fail
with `go.mod requires go >= 1.25.12 (running go 1.25.5; GOTOOLCHAIN=local)`.
`scripts/hardening_check.sh` now defaults `GOTOOLCHAIN` to `go1.25.12` (overridable via
`DEEPDATA_GO_TOOLCHAIN`), so the go gates auto-select the cached toolchain with no manual
env. All receipts at the post-pin HEAD were re-ran and PASS under the pinned toolchain.

One upstream module-hygiene limitation is also recorded: `go mod tidy` follows every build
tag and therefore reaches Cowrie's dormant `agentgo` file, which imports the retired private
`Agent-GO` module. The default graph contains no `Agent-GO` dependency and passes readonly
module resolution, build, test, and verification. Do not add that unrelated graph merely to
make tidy succeed; isolate the unused converter or take an upstream Cowrie fix before
claiming a tidy-clean module.

## Publication-Only Gates

These do not prevent committing and pushing the reviewed candidate branch, but they prevent
tagging or publishing an RC:

- [ ] The owner confirms the copyright holder and approved license, then adds the root
      `LICENSE`.
- [ ] The owner rechecks and secures the `deepdata-client` distribution name and configures
      trusted publishing.
- [ ] The exact DeepData tag is created only after technical evidence and remote CI are
      accepted.
- [ ] Signing, GitHub Release creation, PyPI upload, container/Helm registry pushes, and use
      of credentials receive separate explicit authority.

The dry-run workflow being present is not publication authorization.

## Technical Exit Checklist

- [x] Snapshot fidelity, integrity, and fail-closed recovery are implemented.
- [x] Canonical V3 HTTP and mirrored gRPC share one crash-durable five-mutation engine.
- [x] Tenant/collection permissions and unsupported-surface isolation are implemented.
- [x] Runtime support claims are narrowed to the actual single-node contract.
- [x] The canonical Python client and local package/live integration gates are implemented.
- [x] Docker, Compose, Helm manifests, and an initial local kind lifecycle have preliminary
      parity evidence.
- [x] Version, changelog, upgrade, support, security, and dry-run release metadata are
      aligned for the candidate.
- [x] Complete the final dependency-aware local matrix on a frozen SHA (`d5b2d3a`), then
      re-run the Go gates on the fix SHA (`14d1442`) after the memory-drift fix.
- [x] Complete operational fault, migration, VDB correctness, soak, and restart-under-load
      evidence. **memory-drift now RESOLVED** (resolved finding above); chaos beyond the
      restart-under-load cycles is not separately run.
- [x] Complete exact-SHA security, image, SBOM, artifact, and evidence-report gates (at
      `d5b2d3a`; carry forward to `14d1442` — the Import fix does not alter the dependency
      tree or scanned surface).
- [x] Perform the final adversarial review — an independent reviewer confirmed and
      escalated the memory-drift finding and cleared the four kind/Helm harness fixes as
      legitimate (not masking product/chart bugs), against a clean worktree.
- [x] Resolve the memory-drift finding — fixed at `14d1442` (Import skips deleted entries;
      restart-reclaim proven by A/B control).
- [x] Obtain green remote CI on the exact candidate SHA `14d1442` — PR #4 run `29977008196`,
      all 10 required checks green.
- [x] Add and validate the chart NetworkPolicy / operator isolation contract.
- [x] Land and locally re-verify the adversarial hardening batch (`043ad5d`) — strict
      vector decode, UTF-8-safe ID truncation, auth-gated `/metrics`, FLAT counter fix,
      atomic HNSW batch rollback, sparse overwrite correctness — with `go build`, `go vet`,
      `go test -short ./...`, and `-race` PASS bound to `043ad5d` (check receipts).
- [ ] Re-obtain green remote CI on the exact candidate SHA `043ad5d` — the previously
      green run `29977008196` covered `14d1442`; the hardening batch needs a fresh external run
      (offline queue until an authorized trigger exists).

## Current Decision

The worktree may proceed through final correction, local validation, cohesive commits, and
the authorized push of the current branch. It must not be described as a release-ready,
exact-SHA-attested, signed, tagged, or published RC until the open technical and external
gates above are resolved.
