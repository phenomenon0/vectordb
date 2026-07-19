# DeepData Single-Node Production-Hardening Plan

Generated: 2026-07-17

Reconciled: 2026-07-19. Checkmarks in Phases 2-7 record implemented behavior and
local pre-freeze evidence. They are not an exact-SHA release attestation; every final
candidate rerun and report remains open in Phases 8-9.

Execution is pre-approved by the user for all in-repository implementation, tests,
local builds, and checkpoint commits required by this plan. The authoritative readiness
assessment is [`docs/PRE_RELEASE_STATUS.md`](../docs/PRE_RELEASE_STATUS.md). The completed
mega-benchmark repair remains documented in
[`benchmarks/results/mega/REPORT.md`](../benchmarks/results/mega/REPORT.md).

## Objective and Completion Definition

Deliver a defensible single-node DeepData release candidate. Work continues autonomously
until every required gate below passes on one exact commit, or only a genuine external
decision remains. Distributed/HA stays explicitly experimental and is not a hidden
dependency of the single-node release.

## Frozen RC Scope

The first RC is a Linux-only persistent, headless, single-node vector server. It has one
canonical tenant-aware HTTP API (V3), a gRPC API with the same tenant and operation model,
JWT/static-token authentication, tenant isolation, read/write/admin authorization, Docker,
Helm, and one deliberately small Python client contract.

The supported collection mutations are create collection, delete collection, insert one,
batch insert, and delete document. Supported retrieval is collection get/list plus dense,
sparse/BM25, and hybrid search. Dense indexes are HNSW and Flat; sparse fields use the
inverted index. Callers provide vectors. Server-managed embedding providers, runtime
embedder hot-swapping, bulk-specialized mutation paths, rename, metadata mutation, and
drop-all are outside the RC.

The older root mutation API and V2 collection mutation surface are not production RC paths.
They will be disabled by default rather than receiving separate durability and authorization
implementations. Existing code may remain for explicit migration/experimental tooling, but
it cannot be silently reachable or advertised as supported. Existing V2/root data requires
an explicit, tested migration decision; it must never be discarded or mistaken for an empty
canonical tenant store.

The RC also excludes Windows/macOS persistent runtime support, distributed/HA,
replication/follower restore/snapshot streaming, desktop installers, the web UI as a release
gate, GraphRAG, recommendation/discovery, extraction, feedback loops, DiskANN, IVF,
PQ/binary quantization, CUDA, built-in TLS/mTLS, built-in encryption-at-rest, and
compliance-grade audit logging. Unsupported advanced API handlers are unregistered, not
merely undocumented. Deployments terminate TLS at a trusted proxy/ingress and use encrypted
disks/PVCs. Structured security logs remain supported.

Production hardening is complete only when:

- Every advertised V3 HTTP and tenant-aware gRPC mutation is crash-durable and
  restart-correct through the same mutation engine.
- Corrupt or incompatible state fails closed without overwriting recoverable data.
- Tenant, collection, and read/write/admin authorization is enforced consistently.
- Required CI, minimal SDK, container, Helm, upgrade, security, and soak gates pass.
- Shipped artifacts share one version, commit, feature set, and support statement.
- The exact-SHA evidence report contains no unresolved P0/P1 finding.

External publication, use of private signing credentials, and legal ownership choices are
separate authority gates. The workflow and unsigned artifacts will be made ready without
waiting for those actions.

## Phase 0 — Durable Workflow and Baseline

- [x] Audit overall pre-release status and record prioritized blockers.
- [x] Create a persistent production-hardening goal.
- [x] Upgrade the global workflow skill with truthful outage and safe-edit behavior.
- [x] Commit the project-local resume protocol, state ledger, and this master plan.
- [x] Capture the clean baseline: HEAD, branch, tool versions, disk, ports, and fast tests.
- [x] Add a deterministic local check runner with logs, timeouts, and resumable receipts.

Exit gate: another agent can read the protocol/state, verify the last checkpoint, and name
the next exact action without relying on chat history.

## Phase 1 — Snapshot Fidelity and Fail-Closed Recovery

- [x] Add failing tests proving the default storage codec loses tenant/index semantics.
- [x] Define and version the persisted snapshot envelope and index descriptors.
- [x] Round-trip tenant ownership, named indexes, index types/configuration, vectors,
      documents, metadata, IDs, and all required recovery fields through the default codec.
- [x] Preserve backward reads or provide an explicit, tested migration path.
- [x] Compute integrity data from the state being written and cover meaningful content.
- [x] Make unreadable/corrupt/incompatible snapshots fail closed or enter an explicit
      quarantine/recovery mode; never start empty and overwrite the source.
- [x] Replay a valid WAL when snapshot recovery permits it, without duplicate mutation.
- [x] Add round-trip, truncation, bit-flip, stale-checksum, migration, and subprocess tests.

Exit gate: normal restart preserves every advertised semantic field/index, and fault
injection cannot silently replace recoverable state with an empty database.

## Phase 2 — One Canonical Crash-Durable Mutation Engine

- [x] Disable the legacy root and V2 collection mutation surfaces in normal RC startup.
- [x] Make V3 the canonical tenant-aware collection contract and make gRPC mirror it.
- [x] Inventory the five supported mutations and route them through one durability boundary.
- [x] Specify sequence numbers, WAL records, fsync acknowledgement, replay idempotency,
      checkpoint ordering, rotation, and partial-record handling.
- [x] Implement the collection WAL/checkpoint path without HTTP/gRPC bypasses.
- [x] Ensure acknowledged writes survive SIGKILL/power-loss simulation.
- [x] Ensure unacknowledged/partial writes are either absent or replayed exactly once.
- [x] Persist collection/index lifecycle operations, not only document mutations.
- [x] Add subprocess crash matrices for HTTP V3 and tenant-aware gRPC.
- [x] Preserve and test frozen canonical V1 journal replay, including acknowledged batches
      above current request-admission limits.
- [x] Refuse raw or unified legacy root/V2 state without modifying it, and document the
      explicit offline export/import boundary.
- [ ] Rehearse offline export/import from the last published legacy root/V2 format and
      retain semantic validation evidence; there is deliberately no automatic in-place
      migration.
- [x] Fail non-Linux persistent startup explicitly; cross-compilation is not a support claim.

Exit gate: all advertised protocols pass the same crash/replay invariants across repeated
kill points, checkpoint rotation, and restart.

## Phase 3 — Authorization and Tenant Isolation

- [x] Define one operation policy matrix for read, write, collection-admin, and server-admin.
- [x] Centralize policy evaluation for tenant and collection scope.
- [x] Apply it to every supported V3 HTTP route and every gRPC method.
- [x] Prove unsupported legacy/advanced handlers are unreachable in normal RC startup.
- [x] Remove bearer tokens from URL query parameters.
- [x] Make static-token behavior and administrative capabilities explicit and testable.
- [x] Add table-driven allow/deny tests for cross-tenant, cross-collection, read-only,
      expired, malformed, and missing credentials on both transports.

Exit gate: a shared authorization suite proves least privilege and isolation across all
exposed routes and protocols.

## Phase 4 — Runtime Security and Operational Safety

- [x] Enforce the frozen scope in configuration, startup warnings, support docs, and examples.
- [x] Unregister GraphRAG, recommendation/discovery, feedback, extraction, and server-managed
      embedding/configuration handlers from the RC runtime.
- [x] Emit useful structured security events without secrets or compliance-grade claims.
- [x] Ensure secrets and bearer tokens never enter logs, URLs, panic output, or metrics.
- [x] Make health/liveness/readiness probes work with authentication enabled.
- [x] Validate graceful shutdown in standalone, direct-container, and Compose paths.
- [x] Document a stopped whole-root backup/restore procedure with semantic verification and
      rollback rather than enabling unsafe online import.
- [ ] Run and retain an actual offline backup/restore rehearsal plus process-level disk-full
      and permission-denied evidence on the frozen candidate.

Exit gate: the runtime matches the support matrix, and enabled security features are proved
through the production startup path rather than library-only tests.

## Phase 5 — CI, UI, and Python SDK Gates

- [x] Fix the CGO-disabled SIMD test/build-constraint failure.
- [x] Keep required Go CI short and exclude experimental/large scale work from the required
      RC package matrix.
- [ ] Add a deliberate scheduled/manual 10M/50M/100M scale job if those scenarios remain
      part of the release evidence policy.
- [x] Make Playwright launch the binary produced by its CI job.
- [x] Fix the 16 strict SDK mypy errors from the SDK's own project directory.
- [x] Reduce the Python SDK to the canonical tenant-aware contract, then add unit, strict
      typing, package-build, install, and live-server integration CI.
- [x] Keep the web UI build healthy without making it an RC durability/evidence gate.
- [x] Run canonical subprocess crash/restart tests in the supported Go suite and add direct
      Docker, Compose, and Helm manifest/lint CI contracts.
- [ ] Add live Kubernetes/Helm and VDB correctness jobs at a suitable deliberate cadence.
- [x] Remove artifact uploads that silently ignore missing outputs.
- [x] Split or budget the long race scenario so timeout headroom is credible.

Exit gate: all required local equivalents pass twice and the release-branch workflow is
green on the exact candidate SHA when network execution becomes available.

## Phase 6 — Docker, Compose, and Helm Parity

- [x] Expose and smoke-test both HTTP and gRPC where both are advertised.
- [x] Pin image versions/digests and enforce the Linux-only support statement.
- [x] Fix authenticated probes to use the public readiness/liveness contract.
- [x] Provide a valid existing-Secret path for Helm authentication without retaining secret
      material in Helm values or release records.
- [x] Add security contexts, capability drops, termination grace, persistent volumes,
      resource defaults, a single-node Recreate strategy, and ClusterIP-only defaults.
- [ ] Provide and validate the intended network-isolation contract, either as a chart
      NetworkPolicy or as an explicit operator-managed requirement.
- [x] Prove direct-container and Compose first boot, authenticated HTTP/gRPC, replacement
      persistence, cleanup, and graceful SIGTERM locally.
- [x] Complete a local Kubernetes/Helm install, authenticated HTTP/gRPC, PVC replacement,
      upgrade, rollback, and graceful uninstall rehearsal.

Exit gate: packaged deployments preserve data and protocol/security behavior under the
same smoke contract as the standalone binary.

## Phase 7 — Release Identity, Documentation, and Reproducibility

- [x] Establish one version source and propagate it to server, Python, Helm, artifacts,
      and docs; UI/desktop metadata is non-gating.
- [x] Resolve the Python distribution name collision for the candidate by selecting
      `deepdata-client` and documenting the required ownership recheck before publication.
- [ ] Add the license once the legal copyright holder/choice is confirmed.
- [x] Correct repository/module/package URLs and namespace future DeepData tags.
- [x] Mark distributed/HA and incomplete index operations experimental or unsupported.
- [x] Correct security, environment-variable, dashboard, benchmark, and roadmap claims.
- [x] Write changelog, upgrade/migration notes, support matrix, and security policy.
- [x] Add a tag-driven dry-run-capable workflow for binaries, containers, Python, Helm,
      checksums, SBOM, provenance, and optional signatures/publication.

Exit gate: one command can build reproducible unsigned RC artifacts whose metadata and
documentation identify the same version and commit.

## Phase 8 — Exact-SHA Release Evidence

- [ ] Freeze the candidate SHA and regenerate dependency/tool manifests.
- [ ] Resolve or explicitly retain the recorded Cowrie dormant-build-tag limitation; never
      import the unrelated retired `Agent-GO` dependency solely to make `go mod tidy` pass.
- [ ] Run Go vet, short unit, race, fuzz/property targets, and focused persistence tests.
- [ ] Run Python unit/type/build/install/live integration; keep UI checks informational.
- [ ] Run V3/gRPC restart and crash matrices plus explicit unsupported-surface checks.
- [ ] Run Docker/Compose/Helm install, auth, probe, persistence, shutdown, and upgrade smoke.
- [ ] Run VDB correctness, mixed-load soak, restart-under-load, memory-drift, and chaos.
- [ ] Run dependency, secret, static, image, and SBOM checks with findings triaged.
- [ ] Refresh benchmark provenance where needed without hiding old or incomparable rows.
- [ ] Build and smoke every promised Linux headless-server artifact.
- [ ] Generate a signed-by-hash evidence report tied to the exact Git tree.

Exit gate: the evidence report is reproducible, all required gates pass, and no result came
from a different commit or stale generated asset.

## Phase 9 — Final Adversarial Review

- [ ] Re-audit persistence, authorization, API exposure, configuration, and deployments.
- [ ] Reconcile every completed task against its test evidence and containing commit.
- [ ] Verify a clean worktree and no untracked release-critical files.
- [ ] Update `docs/PRE_RELEASE_STATUS.md` to the final supported/unsupported truth.
- [ ] Mark the persistent goal complete only after every required invariant passes.

## Parallelism and Critical Path

- Critical path: Phase 1 → Phase 2 → Phase 8 crash evidence.
- Authorization may proceed after the canonical V3/gRPC mutation boundary is stable.
- CI/Python and deployment packaging may run in parallel with durability work when their
  file scopes do not overlap.
- Documentation follows behavior; claims are not finalized ahead of implementation.
- Mutation agents use bounded file scopes. The coordinator reviews, tests, and commits.

## Decision Rules

- Correctness and recoverability outrank performance and feature breadth.
- Do not introduce a persistence-format break without a tested dual-read or migration path.
- Never treat corrupt state as an empty new database; preserve originals for recovery.
- A deterministic failure is investigated, not retried unchanged.
- A suspected flaky test gets one controlled rerun; both outcomes remain in evidence.
- Network/API outages move remote work to the offline queue while local work continues.
- Credit exhaustion pauses new reasoning; already-launched deterministic jobs may continue.
- Resume always begins with repository/state reconciliation and the cheapest decisive test.
- Performance regressions over 10% on stable tests require investigation before acceptance.
- No phase is complete because code exists; its exit gate and evidence must pass.
- Do not push, publish, deploy externally, or use private credentials without explicit scope.

## Escalation Triggers

Continue around non-blocking issues and defer them. Escalate only when progress truly needs:

- A legal owner/license choice.
- PyPI/domain/trademark ownership or private signing/release credentials.
- Authorization for an external write such as push, PR mutation, publication, or deployment.
- A destructive action affecting pre-existing user data or unrelated work.
- Acceptance of data loss, an incompatible migration, or a material expansion beyond the
  single-node RC scope.
- An unavailable mandatory external service after local substitutes and retries are exhausted.

## Retry and Timeout Budget

| Operation | Timeout | Retry rule |
|---|---:|---|
| File/query inspection | 15s | Retry once only for transient process failure |
| Network read | 60s | Three attempts with persisted exponential backoff |
| Dependency download | 10m | Two attempts; then defer to online queue |
| Focused compile/test | 5m | No unchanged retry for deterministic failure |
| Full short suite | 10m | One controlled retry only if flaky evidence exists |
| Race suite | 15m | One controlled retry; preserve both logs |
| Container/package build | 20m | One retry for daemon/network failure |
| Soak/chaos | 60m | Resume from scenario checkpoint; never fabricate completion |

## Checkpoints and Resume

- Human plan: this file.
- Machine orientation: [`tasks/autonomy/STATE.json`](autonomy/STATE.json).
- Resume protocol: [`tasks/autonomy/PROTOCOL.md`](autonomy/PROTOCOL.md).
- Semantic checkpoint: cohesive Git commit whose targeted gate passed.
- Runtime logs/evidence: `.deepdata-run/` (local and ignored unless sanitized).
- After every correction, add a reusable rule to [`tasks/lessons.md`](lessons.md).

On resume: read the goal, plan, lessons, and state; inspect Git status and processes; verify
the last commit/test; reconcile offline/deferred work; record the next exact action; continue.
