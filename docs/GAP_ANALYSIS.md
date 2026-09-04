# DeepData gap and upgrade analysis

Dated 2026-09-01 at a443cd0. Status words are a bug in this document: every row cites evidence; the only status is a gate id in tasks/gates.json.

## 0. How to read

Verdicts use four words. **live**: reachable from the RC binary (`cmd/deepdata/main.go:3198` canonicalOnly, `cmd/deepdata/server.go:3414` canonicalRCSurface). **REVIVE**: dead today, returns under a named gate. **KEEP-DORMANT**: stays in the tree; no code path returns before its gate's ADR. **RETIRE**: deleted from the tree under SYS-03; git history is the archive.
Gate ids are the only roadmap vocabulary. Every id resolves in `tasks/gates.json`; `docs/PRE_RELEASE_STATUS.md` is its rendering and the one place a status word may appear.
Line numbers are at a443cd0 unless a sha is named. Docs rewritten by the 2026-09-01 slice are cited as they stood at a443cd0, before the rewrite.
Companions: [ARCHITECTURE.md](ARCHITECTURE.md) (the tower, machine-read non-goals), ADRs 0001-0007 under [decisions/](decisions/0001-narrow-rc-to-single-node-caller-vectors.md), [mcp.md](mcp.md), [PROTOCOL.md](../tasks/PROTOCOL.md), [todo.md](../tasks/todo.md), [README.md](../README.md).

## 1. State of the tree

All 29 tracked code trees. LoC counts tracked source at a443cd0 (a443cd0 changed only docs, scripts and tasks, so 5dafec2 counts hold). CI job names are the `name:` fields in `.github/workflows/ci.yml`.

| path | LoC | reached from RC | CI job | verdict (gate) | evidence |
|---|---|---|---|---|---|
| cmd/deepdata | 27756 | yes, the RC binary | RC Go contract; RC race; Compile proof | live; embedders REVIVE (CTL-02); cmd/deepdata/embedder_providers.go and cmd/deepdata/cost_tracker.go retired under SYS-03 | `cmd/deepdata/main.go:3198`, `cmd/deepdata/server.go:3414` [^1] |
| cmd/deepdata-mcp | 703 | separate binary; not imported, in no CI list | none | live; rewrite (CTL-03); CI coverage (CI-05) | `cmd/deepdata-mcp/main.go:147` forwards the HTTP body verbatim |
| cmd/cli | 990 | no: every path is legacy V1/V2, removed at 9d1672b | none | retired under SYS-03 | 9d1672b (1344 deletions) |
| cmd/gentoken | 153 | separate binary | none | live | prints a curl against a legacy route (§5 row 7) |
| internal/collection | 16358 (8465 src, 7893 test) | yes: the engine | RC Go contract; RC race | live; UsageTracker becomes durable (CTL-05) | `internal/collection/usage.go` (217 lines, memory only) |
| internal/cluster | 10445 | no: zero importers; non-RC | Experimental (non-RC), continue-on-error | retired under SYS-03 | `.github/workflows/ci.yml:271-288` |
| internal/cowrieutil | 485 | no: reached only through internal/storage from `cmd/deepdata/main.go:3651-3656` | RC Go contract test list | retired under SYS-03 | `.github/workflows/ci.yml:71` |
| internal/encoding | 238 | no: zero importers | RC Go contract test list | retired under SYS-03 | `.github/workflows/ci.yml:72`; journal §5.3 Glyph row |
| internal/extraction | 3232 | no: registered only when not canonicalOnly, `cmd/deepdata/server.go:2111` | Experimental (non-RC) | KEEP-DORMANT (MEM-01) | `.github/workflows/ci.yml:285` |
| internal/feedback | 2618 | no: registered only when not canonicalOnly, `cmd/deepdata/server.go:2102` | Experimental (non-RC) | retired under SYS-03 | journal §5.3 feedback row |
| internal/filter | 1604 | yes | RC Go contract | live | `.github/workflows/ci.yml:73` |
| internal/graph | 943 | no: EnableGraphRAG (`cmd/deepdata/collection_http.go:337`) has zero callers; PageRank is not reached | Experimental (non-RC) | KEEP-DORMANT (MEM-01) | `cmd/deepdata/collection_http.go:112` |
| internal/hybrid | 863 | yes | RC Go contract | live | `.github/workflows/ci.yml:74` |
| internal/index | 29662 | hnsw, flat and `internal/index/segmented.go` yes; IVF, DiskANN, PQ, quantized and GPU types rejected by `internal/collection/durable_store.go:589` | all of internal/index in Experimental (non-RC); internal/index/simd and the vendored hnsw module in RC Go contract | `internal/index/segmented.go` KEEP (RCV-06); rejected types retired under SYS-03 | `.github/workflows/ci.yml` lines 69, 78-80, 277 [^2] |
| internal/logging | 546 | yes | RC Go contract | live | `.github/workflows/ci.yml:76` |
| internal/obsidian | 706 | no: skipped when canonicalOnly, `cmd/deepdata/main.go:3503` | Experimental (non-RC) | retired under SYS-03 | `.github/workflows/ci.yml:288` |
| internal/releaseinfo | 28 | yes | via cmd/deepdata | live | artifact version parity step, `.github/workflows/ci.yml:46` |
| internal/security | 7500 | authz half yes; tls, encryption, rotation and audit files have zero live users | RC Go contract (vet and test); RC race | authz half live; dormant half retired under SYS-03 | journal :663-712 [^3] |
| internal/sparse | 1744 | yes | RC Go contract | live | `.github/workflows/ci.yml:78` |
| internal/storage | 1162 | no: never on the RC path | RC Go contract (vet and test); RC race | retired under SYS-03 | [^4] |
| internal/telemetry | 592 | yes, inert for V3 | RC Go contract | live (§5 row 6) | [^5] |
| internal/testutil | 40 | test-only | none | live | zero non-test importers |
| internal/wal | 2753 | no: its only importer is internal/cluster (non-RC) | RC Go contract (vet and test); RC race | retired under SYS-03 | `.github/workflows/ci.yml` lines 55, 75, 99 |
| client | 1000 | no: only importer is cmd/cli; legacy paths | none | retired under SYS-03 | 9d1672b |
| sdk/python | 2983 (1514 src, 1469 test) | yes: the canonical client | Canonical Python client | live | `.github/workflows/ci.yml:118-119` |
| benchmarks (Go outside review/) | 3009 | tooling, not in the binary | none | live; `benchmarks/README.md` rewritten by the 2026-09-01 slice; benchmarks/GAP_ANALYSIS.md replaced by this file | `benchmarks/README.md` |
| benchmarks/review | 1554 | not imported; not in `.github/workflows/ci.yml` | none | live | `benchmarks/review/product_manager_test.go:137` stats this file |
| desktop | 465 (rs, html, js, css) | no: Tauri shell, non-RC | none | RETIRE under SYS-03; ADR [0006](decisions/0006-re-expand-to-agent-memory-platform.md) accepted by the owner 2026-09-01 | 70 tracked files, no CI job |
| tests/ui | 2025 (.ts) | no: browser UI e2e specs whose CI job was removed at d139313 | none | RETIRE under SYS-03; ADR [0006](decisions/0006-re-expand-to-agent-memory-platform.md) accepted by the owner 2026-09-01 | d139313 |

[^1]: cmd/deepdata embedder revival covers `cmd/deepdata/embedder_mode.go` and `cmd/deepdata/onnx_impl.go` (ONNX stays opt-in behind a build tag, never the default); `cmd/deepdata/main.go:3275` hard-wires NewHashEmbedder(1) today.
[^2]: internal/index holds hnsw (with heap) and simd subpackages plus flat, segmented and the rejected families at the top level. Three layers disagree on the index vocabulary (`internal/collection/types.go:174`, `internal/collection/collection.go:198-209`, `internal/collection/durable_store.go:565-593`); SYS-02 unifies them.
[^3]: cmd/deepdata uses only auth.go and rbac.go symbols; audit.go (1082), rotation.go (956), encryption.go (855) and tls.go (549) have zero live users (`tasks/journal/2026-09-01-redesign-architects.md:663-712`). The journal's §1.2 table also lists rbac in the dormant half; the symbol evidence wins, and SYS-03 deletes on import evidence, not on that list.
[^4]: VectorStore.Save (`cmd/deepdata/main.go:837`) reaches format.Save at :959 only from the legacy engine; RC shutdown uses collectionHTTP.Close (`cmd/deepdata/main.go:3674`); STORAGE_FORMAT (`cmd/deepdata/main.go:144`) is parsed and inert.
[^5]: SetupSimple at `cmd/deepdata/main.go:3257`; deepdata_* counters are touched only by legacy handlers (`cmd/deepdata/server.go:451-452` in the /insert handler and :1023-1024 in the /query handler) on the default registry, while the metrics route serves the custom registry (`cmd/deepdata/metrics.go` lines 52 and 334, served at `cmd/deepdata/server.go:1507`; the deepdata_* collectors register on the default registry at `internal/telemetry/metrics.go:78`). No V3 route increments anything.

## 2. Drift inventory

| id | claim | where (at a443cd0 unless noted) | truth | preventing rule |
|---|---|---|---|---|
| D1 | candidate SHA is a99fe53 | `docs/PRE_RELEASE_STATUS.md` lines 5 and 118, and tasks/autonomy/STATE.json:33, both at 5dafec2 | HEAD is a443cd0; the status page is rendered from `tasks/gates.json`; STATE.json was deleted at a443cd0 | `scripts/gates.py` render and check; freshness is computed from git (`scripts/gates.py:75-83`), never typed |
| D2 | "No technical gates remain" | tasks/autonomy/STATE.json:137 at 5dafec2 | `docs/PRE_RELEASE_STATUS.md:5` renders a computed readiness count over 25 release gates | gates.py check --release fails unless every release gate is fresh; the sentence has nowhere to live |
| D3 | license, backup drill and Cowrie boxes shown unfinished | `tasks/todo.md` :198, :142-143, :212-213; STATE.json :135, :144, :113 at 5dafec2; `CHANGELOG.md` lines 4-5 and 60; `docs/PRE_RELEASE_STATUS.md:243-248` at 5dafec2 | closed at b1f28be (Apache-2.0), bc1fa27 (drill, OPS-01), 442cdb6 (cowrie gnn dependency removed; local CSR/PageRank in its place) | fixed by the 2026-09-01 docs rewrite; guarded by R12 (no checkboxes) and R4 |
| D4 | `nine unary RPCs` | `README.md:23`, `internal/collection/API.md:210`, `docs/installation.md` :9 and :198, `docs/why-vectordb.md:11` | `api/proto/deepdata/v3/deepdata.proto` declares 11 unary RPCs (rpc lines 241-251); Upsert and GetDoc landed at a99fe53 | fixed by the 2026-09-01 docs rewrite; guarded by R1 |
| D5 | `five mutations`; `internal/collection/API.md:132` denies upsert on V3 | `internal/collection/API.md:11`, `docs/why-vectordb.md` :16 and :43, `README.md:21` (mutation row lacks upsert) | six journal mutations since a99fe53: CreateCollection, DeleteCollection, Insert, BatchInsert, DeleteDoc, Upsert ([2026-08-07-hardening-upsert](../tasks/journal/2026-08-07-hardening-upsert.md), ADR [0003](decisions/0003-upsert-and-get-doc-join-canonical.md)) | fixed by the 2026-09-01 docs rewrite; guarded by R2 and R4 |
| D6 | unauthenticated-route claim about /metrics | `docs/grafana/README.md:20`, `internal/collection/API.md:63-64`, `docs/security.md` :135 and :149 | auth-gated since 043ad5d (`cmd/deepdata/server.go:1507`) | R4; grafana README and API.md fixed by the 2026-09-01 docs rewrite; `docs/security.md` :135 and :149 patched by hand in the same slice (it was outside the workflow's file set) |
| D7 | group commit ranked as the leading ingest lever | `docs/BENCHMARKS.md:42` | retired at `tasks/todo.md:385`: HNSW construction under the collection lock dominates; order is segments, then allocs, then group commit ([2026-08-22-hardening-ingest](../tasks/journal/2026-08-22-hardening-ingest.md), ADR [0005](decisions/0005-ingest-levers-segments-then-allocs-then-group-commit.md)) | fixed by the 2026-09-01 docs rewrite; guarded by R4 |
| D8 | five incompatible QPS numbers (273, ~500, 6,683, 20,644, 8,377) | 273: `benchmarks/results/2026-03-11_competitive.md:17`; ~500: docs/benchmarks.md:43 (deleted at 5dafec2); 6,683: `benchmarks/results/mega/REPORT.md:143`; 20,644: benchmarks/GAP_ANALYSIS.md:81 (removed by this rewrite) and `benchmarks/vectordbbench/results/COMPREHENSIVE_REPORT.md:372`; 8,377: `docs/BENCHMARKS.md` :29 and :59 | one page owns numbers: `docs/BENCHMARKS.md` (e76116e caveats, 6cc5ef0 equal-effort run); the rest are older harnesses at other settings | deliberately unchecked by the linter (§8); other pages point at BENCHMARKS.md instead of quoting |
| D9 | IVF, DiskANN, PQ and quantization listed as available features | benchmarks/GAP_ANALYSIS.md:11-24 (removed by this rewrite); docs/benchmarks.md, internal/index/README.md and deepdata-system-map.html (removed at 5dafec2) | rejected by `internal/collection/durable_store.go:589`; non-goals in [ARCHITECTURE.md](ARCHITECTURE.md) | R8 non-goal labels |
| D10 | browser UI e2e box at `tasks/todo.md:155`, checked at 6bb19de, says a CI job builds the binary for it | `tasks/todo.md:155` | that job was removed at d139313; no browser job exists in `.github/workflows/ci.yml`; tests/ui RETIRE under SYS-03 (ADR [0006](decisions/0006-re-expand-to-agent-memory-platform.md) accepted by the owner 2026-09-01) | fixed by the 2026-09-01 docs rewrite (DOC-02); guarded by R4 (the tool name is a known-false phrase) and R12 |
| D11 | phantom files referenced from docs | `tasks/journal/2026-09-01-redesign-architects.md:68` names `benchmarks/testdata/vectors.go` and benchmarks/competitive/run_comparison.py | half wrong: vectors.go is tracked; only run_comparison.py is absent, and at a443cd0 no doc in the linter's scope references it (the journal names it at :68 and :1170; `scripts/check_docs_contract.py:609` uses it as a known-missing self-test fixture) | R7 (backticked paths must exist in the tree) |
| D12 | docs/benchmarks.md beside `docs/BENCHMARKS.md` (case collision) | none remain | docs/benchmarks.md (138 lines) deleted at 5dafec2 | R7 |
| D13 | internal/index/README.md described index types the RC rejects | none remain | deleted at 5dafec2 (391 lines); package documentation moves to doc.go files under SYS-04 | R7 |
| D14 | SDK docstring says servers serialize CollectionInfo with capitalized wire names | `sdk/python/deepdata/models.py:126-128`; alias table :131-143 | bc1fa27 made the wire format snake_case; the aliases are backward compatibility only | R4 (the phrase is a known-false pattern); the comment is code, outside the docs rewrite's file set |
| D15 | four status stores (todo.md boxes, STATE.json, PRE_RELEASE_STATUS.md, .deepdata-run receipts) with a precedence list | `tasks/PROTOCOL.md:18-27` names three of the four (todo.md, STATE.json, .deepdata-run receipts); PRE_RELEASE_STATUS.md is the fourth, self-declared at `docs/PRE_RELEASE_STATUS.md:5` as of 5dafec2; :43-44 sets the dirty-tree rule | one ledger, `tasks/gates.json`; STATE.json deleted at a443cd0; receipts stay local evidence cited from journals ([2026-07-22-exact-sha-evidence](../tasks/journal/2026-07-22-exact-sha-evidence.md)) | `scripts/gates.py` check; PROTOCOL rewrite (DOC-02) |
| D16 | branch gnhf/... and a .gnhf/ working directory | `git branch -r` | the run was archived at 5dafec2 (no tracked .gnhf files remain); the remote-tracking branch origin/gnhf/i-want-you-to-mnake-26a28a still exists and is kept (PROTOCOL, Authority) | outside the linter; `tasks/journal/2026-09-01-redesign-architects.md` records the archive |

## 3. Process drift

- Evidence receipts were bound to a3de516 (six gates in `tasks/gates.json`; all ten local receipts per `tasks/journal/2026-09-01-redesign-architects.md:929`) and d5b2d3a (six gates), then product commits kept landing. `scripts/gates.py:75-83` now computes freshness from `git diff` against each gate's scope, so `docs/PRE_RELEASE_STATUS.md:5` reflects the tree rather than a typed receipt.
- Boxes were checked by docs commits, not by the code that earned them: `tasks/todo.md:155` was checked at 6bb19de and refuted by d139313 the next day.
- `tasks/PROTOCOL.md:43-44` says a dirty tree is unverified; gates.py check --release enforces it with the clean-tree fingerprint instead of prose.
- The journal's D11 is half wrong (§2).
- Three docs contradicted the code on metrics auth (D6) for 25 days with nothing to catch it; R4 now does.
- §1 numbers were computed at 5dafec2; `git show --stat a443cd0` touches only docs, scripts and tasks, so they hold at a443cd0.

## 4. Sacrificed-ambition inventory

Source: journal §1.2 and the plan's §7.1. Every row that names a non-goal carries the word that makes it one.

| tree | LoC | why dead | serves the agent-memory thesis? | verdict and gate |
|---|---|---|---|---|
| cmd/deepdata embedders (cmd/deepdata/embedder_providers.go 476, `cmd/deepdata/embedder_mode.go` 418, `cmd/deepdata/onnx_impl.go` 399) | ~1.3k | behind the wall: `cmd/deepdata/main.go:3275` hard-wires NewHashEmbedder(1); the /api/embed handler (`cmd/deepdata/server.go:2577`) is not registered on the RC surface | yes, need 2 (text in, text out) | REVIVE (CTL-02): Ollama default, OpenAI-compatible kept, ONNX opt-in build tag, hash only when named; embedder_providers.go (Gemini, cost ledger) retired under SYS-03 |
| internal/feedback | 1.5k | gated behind not-canonicalOnly, `cmd/deepdata/server.go:2102` | the accretive loop, but UsageTracker already is the ranking signal | retired under SYS-03 (journal §5.3 feedback row); MEM-02 adds POST /docs/{id}/feedback fresh |
| internal/graph | 0.6k | EnableGraphRAG has zero callers (`cmd/deepdata/collection_http.go:112`, :337); PageRank is not reached | later, as a class-C signal over text-in | KEEP-DORMANT (MEM-01): no code returns before the ADR |
| internal/extraction (LLM entity and relation extraction, temporal graph) | 2.1k | gated behind not-canonicalOnly, `cmd/deepdata/server.go:2111` | later, class B/C over text-in | KEEP-DORMANT (MEM-01): no code returns before the ADR |
| internal/encoding (Glyph tabular encoder) | 0.1k | zero importers | no: max_chars truncation is the token control | retired under SYS-03 |
| internal/obsidian | 0.7k, 0 tests | skipped when canonicalOnly, `cmd/deepdata/main.go:3503` | no | retired under SYS-03 |
| Recommend and Discover (`internal/collection/collection.go:1760`, :1888) | 0.3k | no route | later | retired under SYS-03; git history is the archive |
| IVF, DiskANN, PQ, quantized and GPU index types in internal/index | 9.7k source, 15.0k with tests (the 31 tracked top-level files whose names contain diskann, ivf, pq, quantization or gpu; journal :42 estimated 11.6k) | rejected by `internal/collection/durable_store.go:589`; three layers disagree on the vocabulary (footnote 2) | no | retired under SYS-03; SYS-02 unifies what remains; `internal/index/segmented.go` kept (RCV-06) |
| internal/cluster, internal/wal, internal/storage, internal/cowrieutil | 14.8k (§1 totals; 9.3k excluding tests; journal :43 estimated 12k) | zero importers from the RC path; a second WAL and snapshot system; STORAGE_FORMAT parsed and inert (`cmd/deepdata/main.go:144`) | no: non-RC | retired under SYS-03 |
| internal/security dormant half (tls, encryption, rotation, audit files) | 3.4k (the four named files; 5.2k with their tests; journal :44 says 4.2k because it counts rbac.go, see footnote 3) | zero live users (footnote 3) | later, not now | retired under SYS-03 |
| client (Go) and cmd/cli | 2k | every path legacy V1/V2, 404 since 9d1672b | no | retired under SYS-03 |
| desktop (Tauri shell), web UI embed, tests/ui browser e2e specs, vdb-test-suite (69 tracked files) | 465 rs/html/js/css; 2025 ts | non-RC; the browser CI job was removed at d139313 | no | RETIRE under SYS-03; ADR [0006](decisions/0006-re-expand-to-agent-memory-platform.md) accepted by the owner 2026-09-01 |
| cgo and private deps: mattn/go-sqlite3 (cost ledger), onnxruntime_go, Neumenon/cowrie, Neumenon/shard (`go.mod` lines 6, 9, 12, 26) | n/a | dormant features only | ONNX yes, opt-in; the rest no | SYS-03 drops cowrie, shard and sqlite (removed with their packages); ONNX stays behind a build tag, not default |

## 5. Agent-ergonomics gaps

The twelve gaps of journal §3, in its order. What already landed for agents is in [2026-08-17-agent-retrieval](../tasks/journal/2026-08-17-agent-retrieval.md) (bde4f94).

| # | gap | root cause | fixing gate |
|---|---|---|---|
| 1 | no text-to-vector path on any agent surface | the /api/embed handler (`cmd/deepdata/server.go:2577`) is not on the RC surface; `cmd/deepdata/main.go:3275` hard-wires NewHashEmbedder(1); QueryText at `cmd/deepdata/collection_http.go:98` is a fossil nothing reads | CTL-02 |
| 2 | HTTP and MCP errors carry no code, hint, request id or retry verdict | `cmd/deepdata-mcp/main.go:147` forwards the body verbatim; writeCanonicalOperationError at `cmd/deepdata/collection_http.go:638` writes text with no code, hint or request id | CTL-01 |
| 3 | no schema or capability discovery: no OpenAPI, no proto reflection, no status route, no MCP resources | the contract exists only as Go types; nothing serves it | CTL-04 (self-description), CTL-03 (MCP resources) |
| 4 | discovery is admin-gated while search is read-gated | `cmd/deepdata/collection_http.go` :409 and :426 require admin; :454 and :507 require read | CTL-04 |
| 5 | no pagination; top_k capped at 1000; MCP returns up to 16 MiB as one text blob | no max_chars or response_format on the MCP path | CTL-03 (journal §5.3 pagination row: truncation plus steering hint, no cursor) |
| 6 | observability dark for V3 | QueryTimeMs declared at `internal/collection/types.go:461` and never set at `internal/collection/collection.go` :830, :993, :2054; deepdata_* counters live on the wrong registry (footnote 5) | CTL-04 carries the per-result fields; no gate owns registry unification |
| 7 | Go client and CLI dead | client and cmd/cli target legacy routes removed at 9d1672b; cmd/gentoken prints a curl against one | SYS-03 |
| 8 | MCP schemas empty where it matters | `cmd/deepdata-mcp/main.go` declares queries, filters and fallback as bare objects; no outputSchema, no annotations | CTL-03 |
| 9 | score_floor direction inverts by field type (dense at most, sparse at least) and is reported nowhere at runtime | documented in prose only; no score_direction in any response | CTL-04 |
| 10 | engine errors unclassified at the root | `internal/collection/collection.go:511-523` returns bare fmt.Errorf; only :531 wraps ErrInvalidSearchArgument; the classifier is duplicated at `cmd/deepdata/collection_grpc.go:279-302` and `cmd/deepdata/collection_http.go:1063-1079` | CTL-01 |
| 11 | permanent limits reported as retryable | `cmd/deepdata/collection_http.go:651-653` maps tenant and collection limits to 429; `cmd/deepdata/collection_grpc.go:468-471` to ResourceExhausted | CTL-01 (409 / FailedPrecondition, retryable false; journal §5.3) |
| 12 | GET /v3/tenants/{t} returns 200 with zeros for an unknown tenant; correct but unsaid | a tenant exists iff it owns a collection; there is no create-tenant mutation among the six | CTL-04 (contract text) |

## 6. The tower

Seven layers, one source, N projections, every projection checked in CI (journal §5.1). Prose lives in [ARCHITECTURE.md](ARCHITECTURE.md); decisions in [decisions/](decisions/0001-narrow-rc-to-single-node-caller-vectors.md).

| layer | contents | gate |
|---|---|---|
| L6 truth | `tasks/gates.json` rendered to `docs/PRE_RELEASE_STATUS.md`; ARCHITECTURE non-goals block; ADRs; `scripts/check_docs_contract.py` | DOC-01, DOC-02 |
| L5 agent | cmd/deepdata-mcp with six memory verbs whose input schemas are the L2 files | CTL-03, CI-05 |
| L4 clients | sdk/python (pydantic fields equal L2); generated proto stubs, diff-gated | SDK-01, SDK-02 |
| L3 transports | HTTP /v3 and the V3 proto service, thin; shared internal/apierror, embed_text.go and serverRuntime | CTL-01, CTL-02, SYS-01 |
| L2 contract | api/contract/v3 JSON Schema and operations list; /v3/status is its runtime projection | CTL-03, CTL-04 |
| L1 engine | internal/collection: hnsw, flat and inverted indexes, fusion, filters, UsageTracker; typed sentinels in `internal/collection/limits.go` | CTL-01, CTL-05 |
| L0 durability | DurableStore journal and snapshot v2 (class A); usage sidecar (class B); indexes (class C) | DUR-01..05, RCV-01..06 |

Durability classes (journal §5.2): **A canonical**: the six journal mutations, snapshot v2, schema; corruption fails closed. **B accreted signal**: usage frecency, later feedback weights; corruption is a loud discard (error log plus a status field) and the collection stays up. **C derived**: indexes; rebuilt from A on load.

Where the architects disagreed, one side was picked (journal §5.3):

| topic | picked | rejected | why |
|---|---|---|---|
| contract source | hand-written JSON Schema under api/contract/v3, embedded, drift tests each direction | generator from Go types | the schema is the contract; Go is one projection |
| quota-limit status | 409 / FailedPrecondition, code quota_exceeded, retryable false | 403 | 403 conflates with auth; the remedy differs |
| embedding-mismatch status | 409 / FailedPrecondition | 400 | the request is well-formed; server state cannot honor it |
| accretion store | durable UsageTracker sidecar (~80 LoC) | a signal package (~250 LoC) | one signal needs no package; the class rule is what matters |
| Glyph encoding | retire internal/encoding | keep, cheap | non-deterministic map order, zero consumers; max_chars is the token control |
| embedders kept | Ollama default, OpenAI-compatible (`cmd/deepdata/main.go:1606`), ONNX opt-in build tag, hash never default | Ollama and ONNX only | the OpenAI adapter exists in stdlib Go; deleting it saves nothing |
| pagination | none: max_chars truncation plus steering hint | cursor | the engine has no offset; pageCursor (`cmd/deepdata/server.go:3435`) serves legacy list only |
| module rename | declined; drop the Canonical prefix instead (SYS-04) | rename to deepdata | cosmetic, touches every import path |
| feedback loop | do not revive internal/feedback | revive as class B | UsageTracker is already the signal; make it durable (CTL-05) |
| docs graph | linter plus generated blocks | mermaid generator | a diagram nobody's test reads |

## 7. Roadmap

Gate ids only, in the plan's §9.2 order. Each row names the tests that prove it; the ledger holds the evidence.

| gate | what it proves | tests that prove it |
|---|---|---|
| CTL-01 | errors are prompts: code, hint, docs pointer via internal/apierror on every surface | TestSearchErrorsAreTypedSentinels in `internal/collection/agent_retrieval_test.go`; HTTP envelope, request_id echo, 429 Retry-After and gRPC ErrorInfo/RetryInfo in `cmd/deepdata/apierror_transport_test.go`; MCP isError plus structuredContent in `cmd/deepdata-mcp/main_test.go`; envelope fields and the retry verdict in `sdk/python/tests/test_errors.py` |
| CTL-02 | text in, text out; Ollama first; hash only when named | TextToSparse deterministic; embedding schema validation and dim fill; create with hash, insert texts, search texts, hit carries embedded_by |
| CTL-03 (+CI-05) | MCP rewritten on the shared api/contract package | tools/list has outputSchema and annotations for all six verbs; concise truncation sets truncated plus hint; remember routes one doc to POST, id to PUT, many to batch; resources/read serves the contract; cmd/deepdata-mcp in the CI package list |
| CTL-04 (+DOC-03) | self-description and per-result confidence | contract_test.go reflects json tags, limits and the proto service set against operations.json; test_contract.py on the SDK; /v3/status shape; score_direction present; API.md route table generated from the routes subcommand |
| CTL-05 | durable usage survives restart | touch, close, reopen, usage_boost still reorders; corrupt sidecar leaves the collection up with the usage signal flagged unloaded |
| SYS-01 | serverRuntime extracted from `cmd/deepdata/main.go`; func main() constructs no VectorStore and NewVectorStore has no live caller | all canonical tests green with the legacy engine unconstructed |
| SYS-02 | one IndexTypes vocabulary | table test: every accepted type builds and round-trips the journal |
| SYS-03 | archive tag archive/pre-narrowing-v2; RETIRE rows of §4 deleted; `go.mod` loses cowrie, shard and sqlite (removed with their packages) | go build, go vet and go test -short over ./... after the deletion |
| SYS-04 | Canonical prefix dropped; a doc.go in every live package | gofmt -l empty; the ARCHITECTURE non-goals line for server-managed embeddings removed once CTL-02 lands |
| MEM-01 | ADR for graph and extraction revival as class-B/C signals over text-in | decision only; no code returns before it |
| MEM-02 | POST /docs/{id}/feedback stored durably | when a harness asks; round-trip test through restart |
| RCV-03..06 | bounded recovery ([2026-08-28-bounded-recovery](../tasks/journal/2026-08-28-bounded-recovery.md)); ADR [0007](decisions/0007-persisted-segment-count-never-derived-from-gomaxprocs.md) | unified snapshots under a memory cap; generated small-journal tests fail closed on truncated and corrupt records; replay under a cgroup hard cap; `internal/index/segmented.go` under the cold-start envelope |

Order: CTL-01, CTL-02, CTL-03 (+CI-05), CTL-04 (+DOC-03), CTL-05, SYS-01, SYS-02, SYS-03, SYS-04; MEM-01 and MEM-02 after CTL-02. RCV-03..06 run in parallel by hand; RCV-03 is the only release gate among them (`tasks/gates.json` release_gate), and none gates the platform. DOC-01 and DOC-02 are this rewrite.

## 8. How this stays true

- `scripts/check_docs_contract.py` runs over README.md, CHANGELOG.md, docs/**/*.md, internal/collection/API.md, sdk/python/README.md, benchmarks/*.md, tasks/todo.md and tasks/PROTOCOL.md (tasks/journal/, docs/decisions/, benchmarks/results/ and benchmarks/competitive/live/ excluded); `.github/workflows/ci.yml` runs it, then `scripts/gates.py` check, after the version-parity step. R1: any count near RPC must be 11. R2: any spelled-out mutation count must be six. R4: a list of known-false phrases (the metrics-auth claim, the license claim, the old RPC count, the upsert denial, the wire-name claim, the browser tool, the ingest-lever claim). R5: relative links resolve. R7: backticked paths exist in `git ls-files`. R8: non-goal words from the ARCHITECTURE block need a negation word in the same row, item or paragraph. R12: no checkboxes in tasks files. DOC-01 is the run.
- Generated blocks: `docs/PRE_RELEASE_STATUS.md` from `tasks/gates.json` (`scripts/gates.py` render, checked by check); the ARCHITECTURE non-goals block, read by the linter; the API.md route table from the routes subcommand (DOC-03). Hand edits fail the check.
- Computed staleness: `scripts/gates.py:75-83` diffs each gate's evidence commit against its scope; nothing stores "stale". check --release also requires the clean-tree fingerprint.
- Journal and ADR discipline: one journal per session under tasks/journal/ with Evidence, Decisions and Lessons sections; every decision becomes a numbered ADR under docs/decisions/ or "none". Local receipts under .deepdata-run/ are cited from journals, never from docs.
- Deliberately unchecked: numbers (one page, `docs/BENCHMARKS.md`, owns them), tone, external URLs.
