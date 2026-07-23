# DeepData Pre-Release Status

**Last reconciled:** 2026-07-22

**Candidate state:** Product code frozen at `d5b2d3a`
(`d5b2d3a4b9e83e2111d93aa9f2cfa477dd779fda`, tree `982daf0`) on
`gnhf/i-want-you-to-mnake-26a28a`. The full exact-SHA local evidence matrix has been run
against this tree; an exact-tree evidence report is recorded at
`.deepdata-run/rehearsals/EVIDENCE-d5b2d3a.md`.

**Technical verdict:** **Not ready to tag — one open correctness finding.** The narrow
single-node RC implementation is complete and the exact-SHA local matrix is green
*except* for the long-running memory-drift gate, which **fails for a real, root-caused
reason**: canonical HNSW deletes are never reclaimed in canonical mode — not online, and
not on restart. This is a concrete ship/no-ship item for the owner, not merely "evidence
not yet gathered." See **Open Finding: memory-drift** below.

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
- create/delete collection, insert, batch insert, and delete-document mutations;
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

| Priority | Gate | Status at `d5b2d3a` |
|---|---|---|
| P0 | Final dependency and toolchain tree | **Done.** vet/storage/short/race, benchmark-unit, python unit/mypy/build, state-json, ui-build all PASS bound to `d5b2d3a` (check receipts). |
| P0 | Exact-SHA security and artifact proof | **Done.** All Linux amd64 artifacts built, smoked (version `0.2.0-rc.1`, mutate+search+graceful stop), checksummed, with source+image SPDX SBOMs; exact-tree evidence report generated against a clean worktree. Security-scan results recorded in `STATE.json`. |
| P1 | Final deployment parity | **Done.** Direct container, Compose, and live kind/Helm lifecycle (install, authenticated HTTP+gRPC, PVC persistence across pod replacement, upgrade, rollback, graceful uninstall) PASS with a digest-pinned image. |
| P1 | Operational fault and migration rehearsal | **Done.** Whole-root backup/restore + process-level disk-full and permission-denied behavior PASS; explicit legacy export/import semantic rehearsal PASS. |
| P1 | Long-running correctness | **FAIL (open finding).** Soak recall (0.9815), flat-exact (100/100), restart-under-load (5× SIGKILL clean), and count-parity PASS; **memory-drift FAILS** — see Open Finding below. |
| P1 | Network isolation contract | Open. Add and validate a chart NetworkPolicy or document and test a precise operator-managed isolation requirement. |
| External | Remote CI | Open. Obtain a green required workflow run on the exact candidate SHA. A branch push alone does not trigger the current main-push-or-PR workflow. |

The exact-SHA local matrix is complete; the only failing local gate is memory-drift, and
one operational gate (network isolation) plus external CI remain.

## Open Finding: memory-drift (canonical HNSW delete reclamation)

**Verdict: FAIL — real, root-caused, and independently (adversarially) verified against
source.** Canonical HNSW `Delete()` is soft-only (`internal/index/hnsw.go:629`): it sets a
tombstone and frees nothing. In canonical mode — the RC's only mode — no online compaction
is reachable (`cmd/deepdata/main.go:3458` closes the compaction channel; the `/compact`
handler and tombstone goroutine act only on the legacy store). Critically, **process
restart does not reclaim either**: `Export()` serializes every tombstoned vector with its
full data (`hnsw.go:748-763`) and `Import()` re-adds them all to the rebuilt graph
unconditionally (`hnsw.go:880`) before re-marking them deleted. The only code path that
drops tombstones is `HNSWIndex.Compact()` (`hnsw.go:911`), which nothing in canonical mode
ever calls.

**Impact.** Under delete-heavy or sustained delete+reinsert churn, both in-memory RSS and
the on-disk snapshot grow ~linearly with cumulative deletes (~1.9 KB/delete, ≈+14 MB/min
in the probe) with no online or restart remedy; the only way to reclaim is to drop and
recreate the collection. Insert-mostly / read-mostly workloads are unaffected. Full
evidence: `.deepdata-run/rehearsals/memdrift-d5b2d3a/FINDING.md`.

**Owner decision (ship/no-ship):**

1. **Ship RC with a hard documented limitation** — "canonical mode performs no delete
   reclamation; neither checkpointing nor restart reclaims deleted-vector memory;
   delete-heavy workloads must drop and recreate the collection to reclaim." Keeps the
   narrow-RC scope; the limitation is materially more severe than a transient drift.
2. **Block and fix** — either make a canonical compaction trigger reachable, or (smaller,
   ~few lines) make `Import` skip re-adding `Deleted` entries so restart reclaims. Either
   change edits product code and therefore **unfreezes `d5b2d3a`**, forcing a new candidate
   SHA and a full re-run of the exact-SHA evidence matrix.

This is deliberately left to the owner and is not silently resolved.

No item in this table is waived by a preliminary pass from a dirty worktree.

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
- [x] Complete the final dependency-aware local matrix on one frozen SHA (`d5b2d3a`).
- [x] Complete operational fault, migration, VDB correctness, soak, and restart-under-load
      evidence. **Exception: memory-drift FAILS** (open finding above); chaos beyond the
      restart-under-load cycles is not separately run.
- [x] Complete exact-SHA security, image, SBOM, artifact, and evidence-report gates.
- [x] Perform the final adversarial review — an independent reviewer confirmed and
      escalated the memory-drift finding and cleared the four kind/Helm harness fixes as
      legitimate (not masking product/chart bugs), against a clean worktree.
- [ ] Resolve the memory-drift finding (owner ship/no-ship decision).
- [ ] Add and validate the chart NetworkPolicy / operator isolation contract.
- [ ] Obtain green remote CI on the exact candidate SHA.

## Current Decision

The worktree may proceed through final correction, local validation, cohesive commits, and
the authorized push of the current branch. It must not be described as a release-ready,
exact-SHA-attested, signed, tagged, or published RC until the open technical and external
gates above are resolved.
