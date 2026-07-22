# DeepData Pre-Release Status

**Last reconciled:** 2026-07-19

**Candidate state:** Implementation checkpointed through `e01c41b` on
`gnhf/i-want-you-to-mnake-26a28a`; the exact candidate SHA is not frozen.

**Technical verdict:** **Pre-exact-SHA validation, not ready to tag.** The narrow
single-node RC implementation is substantially complete and has passed preliminary local
gates, but the final dependency/toolchain tree has not completed the exact-SHA evidence
matrix.

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

| Priority | Gate still open | Required evidence |
|---|---|---|
| P0 | Final dependency and toolchain tree | Verify the module graph, rebuild with the pinned Go toolchain, and rerun vet, short, race, focused persistence, crash, compatibility, and unsupported-surface tests on one frozen SHA. |
| P0 | Exact-SHA security and artifact proof | Rerun dependency, secret, static, filesystem, image, and SBOM checks; triage supported-scope findings; build and smoke every promised Linux amd64 artifact; generate the exact-tree evidence report. |
| P1 | Final deployment parity | Rebuild the candidate image and repeat direct container, Compose, manifest, and corrected Helm/kind auth, probe, persistence, shutdown, upgrade, and rollback contracts. |
| P1 | Operational fault and migration rehearsal | Retain a real whole-root backup/restore result, process-level disk-full and permission-denied behavior, and an explicit legacy export/import semantic rehearsal. |
| P1 | Long-running correctness | Run deliberate VDB correctness, mixed-load soak, restart-under-load, memory-drift, and chaos scenarios without treating the older mega benchmark as current proof. |
| P1 | Network isolation contract | Add and validate a chart NetworkPolicy or document and test a precise operator-managed isolation requirement. |
| External | Remote CI | Obtain a green required workflow run on the exact candidate SHA. A branch push alone does not trigger the current main-push-or-PR workflow. |

No item in this table is waived by a preliminary pass from the dirty worktree.

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
- [ ] Complete the final dependency-aware local matrix on one frozen SHA.
- [ ] Complete operational fault, migration, VDB correctness, soak, restart-under-load,
      memory-drift, and chaos evidence.
- [ ] Complete exact-SHA security, image, SBOM, artifact, and evidence-report gates.
- [ ] Obtain green remote CI on the exact candidate SHA.
- [ ] Perform the final adversarial review with a clean worktree and no untracked
      release-critical files.

## Current Decision

The worktree may proceed through final correction, local validation, cohesive commits, and
the authorized push of the current branch. It must not be described as a release-ready,
exact-SHA-attested, signed, tagged, or published RC until the open technical and external
gates above are resolved.
