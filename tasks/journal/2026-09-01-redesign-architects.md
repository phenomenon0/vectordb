# 2026-09-01 — redesign-architects
Commits: none (planning session; tree dirty with the 2026-08-28 recovery work) · Gates touched: none

## What happened
Gap/upgrade analysis and agent-first redesign of DeepData. Three read-only explorations (surfaces; engine + dormant trees; docs + plans) fed three architect deliberations, reconciled by the main session into the plan whose sections 1–5 are reproduced below. Sections 6–9 of that plan (truth ledger, disposition, Workflow, slices) are executed in the following commits. This file is the Workflow's input; it is never edited after this session.

## Evidence
All file:line citations below were verified against the working tree at `6adde4c` (dirty) on 2026-09-01.

## Decisions
→ docs/decisions/0006-re-expand-to-agent-memory-platform (proposed)

## Lessons
none yet

---

# Part 1 — synthesized plan, sections 1–5

## 1. State of the system at HEAD (evidence)

### 1.1 What is live (reachable from the shipped binary)
- `const canonicalOnly = true` (`cmd/deepdata/main.go:3198`); `canonicalRCSurface` allowlists `/v3/tenants/`, `/healthz`, `/readyz`, `/livez`, `/metrics` and 404s ~60 other registered routes.
- V3 HTTP router: `cmd/deepdata/collection_http.go:389-514` (manual `SplitN` switch; no route table). 11 operations: tenant info (admin), list/create collections (admin), get/delete collection (read/admin), insert, delete doc, batch insert, upsert, get doc, search (read).
- gRPC `deepdata.v3.DeepData`: **11** unary RPCs (`api/proto/deepdata/v3/deepdata.proto:240-252`); no reflection, no `grpc_health_v1`.
- Engine `internal/collection` (8.5k src LoC): `DurableStore` single mutex boundary (`durable_store.go:88`), six mutation types (`:17-23`), fault latching → `/readyz` 503, CRC-framed journal, snapshot v2 streaming format (dirty tree), fail-closed on legacy/corrupt state. Dense HNSW/Flat/(opt-in Segmented), sparse BM25, two-field fusion, metadata filters incl. undocumented `regex`/`geo_*`.
- Agent retrieval contract (`bde4f94`): `score_floor`/`fallback`/`usage_boost` → `weak_match`/`best_score`/`fell_back_to`; consistent across HTTP/gRPC/Python SDK.
- Python SDK `sdk/python/deepdata`: only 1:1 surface; pydantic cross-field validators (`models.py:296-353`); typed errors with `.retryable`.
- MCP `cmd/deepdata-mcp/main.go` (449 LoC): 5 tools, untyped object args, raw-text errors, single env tenant. **Not in any CI package list.**
- Benchmarks (`docs/BENCHMARKS.md`, sift-100k): 3,614 vec/s durable+indexed vs Qdrant 103,965; time-to-searchable 27.7 s vs 3.0 s (~9x, weakest axis); search 8,377 / 49,020 qps vs 1,170 / 1,469. Bottleneck = single-threaded HNSW build under the collection lock (~155 µs/doc, 817 allocs/doc).

### 1.2 What is dead (compiles, unreachable)
| Tree | LoC | Why dead | Serves agent-memory thesis? |
|---|---|---|---|
| `cmd/deepdata` embedders (`embedder_providers.go` 476, `embedder_mode.go` 418, `onnx_impl.go` 399: Gemini/OpenAI/Ollama/ONNX BGE-small/hash; `/api/embed`) | ~1.3k | behind wall; `main.go:3274` hard-wires `NewHashEmbedder(1)` | **Yes — the missing text-in path** |
| `internal/feedback` (explicit+implicit relevance feedback → boosts; own JSON WAL) | 1.5k | handlers gated `!canonicalOnly` (`server.go:2096`) | **Yes — the accretive loop** |
| `internal/graph` (CSR + PageRank/PPR; `hybrid.HybridSearchWithGraph`) | 0.6k | `EnableGraphRAG` has zero callers; drags graph→extraction→feedback into live import graph | Later |
| `internal/extraction` (LLM KG extraction, Ollama/OpenAI, temporal KG) | 2.1k | gated `!canonicalOnly` | Later |
| `internal/encoding` (Glyph tabular encoder, 50-62% token savings) | 0.1k | zero importers | Maybe — cheap |
| `internal/obsidian` (vault sync) | 0.7k, 0 tests | skipped when `canonicalOnly` | Later/retire |
| `Recommend`/`Discover` (`collection.go:1760/1888`) | 0.3k | no route | Later |
| IVF/DiskANN/PQ/quant/GPU (`internal/index`) | 11.6k | rejected by `validateCanonicalSchema` (`durable_store.go:589`); three layers disagree on the index vocabulary (`types.go:174`, `collection.go:198-209`, `durable_store.go:589`) | No (RC non-goal) |
| `internal/cluster` + `internal/wal` + `internal/storage` + `internal/cowrieutil` | 12k | zero importers; two WAL/snapshot systems; `STORAGE_FORMAT` parsed but inert | No |
| `internal/security` dormant half (tls/encryption/rotation/audit/rbac) | 4.2k | zero live users | Later |
| `client/` (Go) + `cmd/cli` | 2k | **every path legacy V1/V2 → 404** | Regenerate or delete |
| `desktop/` (Tauri, with a server binary in `src-tauri/binaries/`), web UI embed + `tests/ui` Playwright specs, `vdb-test-suite/` | — | not RC; 3.4 GB gitignored artifacts. (Root `deepdata`/`deepdata-server`/`cli` binaries are **untracked** build output — `rm`, not a history problem; the only oversized tracked blob is `docs/models/.../tokenizer.json` 711 KB) | No |
| cgo deps for dormant features: `mattn/go-sqlite3` (cost ledger), `onnxruntime_go`; private `Neumenon/cowrie`, `Neumenon/shard` | — | — | ONNX yes; rest no |

### 1.3 In-flight uncommitted work (2026-08-28 → 29)
Bounded-memory recovery after the 4.71 GB journal OOM (`readAfter` → `validateAndRepair` + `streamReplay`; snapshot v2; typed-vector cloning; `internal/index/segmented.go`; `DEEPDATA_BIND_HOST`). Its own checklist (`tasks/todo.md:392-470`) has 3 open items: bounded-memory unified snapshots box, generated small-journal fail-closed tests, replay rehearsal under a cgroup hard cap. Safety rules stand: never start the preserved journal copies (`/home/omen/var/deepdata`, `/run/media/omen/Storage/miniexa/deepdata`) with a pre-fix binary; never point tests at production journals. `PROTOCOL.md:44`: a dirty tree is unverified work.

---

## 2. Drift inventory (claim → truth → fix)

| # | Claim | Where | Truth | Fix |
|---|---|---|---|---|
| D1 | Candidate SHA = `a99fe53` | `docs/PRE_RELEASE_STATUS.md:5,:118`, `tasks/autonomy/STATE.json:33` | HEAD is 16 commits ahead incl. a breaking http refactor (`9d1672b`), a new contract + binary (`bde4f94`), a wire-format change (`bc1fa27`) | regenerate from ledger (§6) |
| D2 | "No technical gates remain in-repo" | `STATE.json:137` | `todo.md` Phase 8 (11 boxes) + Phase 9 (5) fully open | delete STATE.json; ledger |
| D3 | License pending; backup drill pending; Cowrie `go mod tidy` blocker | `todo.md:198,:142`, `STATE.json:135,:144,:113`, `CHANGELOG.md:5,:60`, `PRE_RELEASE_STATUS.md:243-248` | closed at `b1f28be`, `bc1fa27` (`todo.md:364`), `442cdb6` (`todo.md:363`) | patch |
| D4 | "nine unary RPCs" | `README.md:23,:194-199`, `internal/collection/API.md:210-220`, `docs/installation.md:9,:198`, `docs/why-vectordb.md:11` | 11 (`Upsert`, `GetDoc`) | generate block from proto |
| D5 | "five mutations"; upsert/get-doc "not a good fit" | `docs/why-vectordb.md:16,:43`, `API.md:11`, `README.md:22` | six mutations + get-doc shipped `a99fe53`; `cookbook.md` correct | patch/generate |
| D6 | "`/metrics` is unauthenticated" | `docs/grafana/README.md` | auth-gated since `043ad5d` (`server.go:1507`, tested) | patch; dashboard has 4 replication/failover panels with no emitter |
| D7 | group-commit = top ingest lever | `docs/BENCHMARKS.md:42` | retired one commit later (`todo.md:385`): parallel segment build → alloc reduction → group commit deferred | patch |
| D8 | Five incompatible QPS numbers (273 / ~500 / 6,683 / 20,644 / 8,377) | `benchmarks/competitive/live/results/REPORT.md`, `docs/benchmarks.md`, `benchmarks/results/mega/REPORT.md`, `benchmarks/GAP_ANALYSIS.md`, `docs/BENCHMARKS.md` | only BENCHMARKS.md explains (client overhead) | archive the March files; one benchmark page |
| D9 | FP16/Uint8/PQ/Binary/IVF/DiskANN/graph-rerank "Yes" | `benchmarks/GAP_ANALYSIS.md:11-24`, `docs/benchmarks.md:58-67`, `benchmarks/README.md`, `deepdata-system-map.html` | all RC non-goals (`README:34-38`, `todo.md:45-46`) | retire/regenerate |
| D10 | `[x]` Playwright launches CI binary | `todo.md:155` | `ci.yml` has zero playwright | uncheck / drop gate |
| D11 | `benchmarks/testdata/vectors.go`, `competitive/run_comparison.py` exist | `benchmarks/README.md`, `docs/benchmarks.md:130` | don't exist | linter: dead file refs |
| D12 | `docs/benchmarks.md` + `docs/BENCHMARKS.md` | both present | case-collide on macOS/Windows (CI cross-compile targets); neither linked from README | delete the March one |
| D13 | `internal/index/README.md` "VectorDB Enhancement Plan (12-month roadmap), Phase 1 Complete" | Dec 2025 | a third roadmap, different product name | delete |
| D14 | SDK parses title-case aliases "because current servers emit them" | `sdk/python/deepdata/models.py:131-143` | server emits snake_case since `bc1fa27` | patch comment (code harmless) |
| D15 | Four status stores, `PROTOCOL.md:20-29` precedence not followed; receipts in gitignored `.deepdata-run/` | — | unverifiable from a clone | ledger + committed receipts index |
| D16 | Branch `gnhf/i-want-you-to-mnake-26a28a`; `.gnhf/` 18 MB hidden via `.git/info/exclude` | — | abandoned April run against deleted V2 architecture | rename branch; archive `.gnhf/` |

Also: zero ADRs (four real decisions scattered across four files); `.agents/` `.codex/` empty; `docs/models/bge-small-en-v1.5/tokenizer.json` (711 KB) in the docs tree; `docs/security-patterns.md` cites Tauri/replication; `.claude/settings.local.json` has stale one-off allowlist entries.

---

## 3. Agent-ergonomics gaps, ranked (from the driver's seat)

1. **No text→vector path on any agent surface.** `/api/embed` exists (`server.go:2577`) and is 404'd. An LLM cannot emit a 384-float vector → the MCP server is theoretical. `searchRequestRaw.QueryText` (`collection_http.go:98`) is a fossil of the intended design.
2. **Errors are unstructured plain text** on HTTP and MCP (40+ `http.Error()`; MCP forwards body verbatim at `main.go:147`). No code/field/hint/request_id/retryable. gRPC's `canonicalGRPCError` is the one good classifier; `writeCanonicalOperationError` (`collection_http.go:638-656`) knows the taxonomy and throws it away.
3. **No schema/capability discovery**: no OpenAPI, no gRPC reflection, no `/v3/status`, no MCP resources. To insert one doc an agent must already know tenant, collection, every field name/dim/encoding.
4. **Discovery is admin-gated while search is read-gated** (`collection_http.go:409,:426` vs `:454,:507`); a least-privilege MCP agent gets 403 on `list_collections`, its only discovery tool.
5. **No pagination/cursor**; `top_k` ≤ 1000; MCP dumps up to 16 MiB as one text blob; no `response_format`.
6. **Observability dark for V3**: zero `withMetrics` on V3 routes; `deepdata_*` in the wrong registry; no OTel spans; `SearchResponse.QueryTimeMs` (`types.go:461`) is declared but **never set** at any of the three construction sites (`collection.go:830,:993,:2054`) — a lie in a struct, ~8-line fix at `TenantManager.SearchCollection`.
7. **Go client + CLI dead** (100% legacy paths); `gentoken` prints a `curl` against a dead route.
8. **MCP schemas empty where it matters**: `queries`/`filters`/`fallback` bare objects; `include_vectors` unschema'd; no outputSchema so `weak_match` arrives undescribed; no annotations.
9. **`score_floor` direction inverts by field type** (dense ≤, sparse ≥), documented in four prose places, reported nowhere at runtime — silently wrong after `fell_back_to`.
10. **Engine errors are unclassified at the root.** `collection.go:511-523` returns bare `fmt.Errorf` for queries-empty / field-count / `top_k` / `ef_search`; only `:531+` wraps `ErrInvalidSearchArgument`. So an unguarded violation reaching the engine becomes **500 / `Internal`** — which is exactly why `collection_grpc.go:279-302` and `collection_http.go:1063-1079` each re-implement the same five checks. Root cause: five missing `%w`. Fixing them deletes ~60 lines of transport duplication.
11. **Permanent limits are reported as retryable.** `ErrTenantLimitExceeded`/`ErrCollectionLimitExceeded` → 429 (`collection_http.go:651-653`) and `ResourceExhausted` (`collection_grpc.go:466-469`); `StoreLimits.MaxTenants` is immutable for the process lifetime, so every standard retry policy (incl. the SDK's) retries forever against a wall. Must be 409 / `FailedPrecondition`. Rate-limit 429 also lacks `Retry-After`; SDK `RateLimitError.retry_after` is dead code.
12. **Not a bug, but unsaid**: `GET /v3/tenants/{t}` → 200 with zeros for an unknown tenant is *correct* — there is no create-tenant mutation among the six; a tenant exists iff it owns a collection. The contract must say so.

**Keystone structural fact (why the monolith can't be cut today):** `main.go:3318` builds `NewVectorStore(0, 1)` — the dead legacy engine (`main.go:53-113`, ~60 fields) — solely because the **live** RC keeps its auth state there: `apiToken`, `jwtMgr`, `requireAuth`, `acl`, `quotas`, `canonicalTenantRL`, `authFailureRL` (`main.go:83-101`); `main.go:~3425` reads five of them to build the gRPC interceptor. Those seven fields must be extracted before anything legacy is deletable.

### Already right — build on, don't replace
bde4f94 prose contract (`types.go:385-404/466-478`, `proto:148-193`, `models.py:296-373`, `README:136-155`) · `canonicalGRPCError` · gRPC pre-validation (`collection_grpc.go:279-302`) · pydantic validators as de-facto JSON Schema · `/readyz` named-check shape · `X-Request-ID` echo (`server.go:3362-3377`) · `normalizeMetricsPath` (`cmd/deepdata/metrics.go:280-303`) · `gentoken -json` · uniform `[A-Za-z0-9_-]{1,64}` identifier rule at 4 points · `CollectionInfo` already carries fields · `scripts/check_version_contract.py` (one source → 6 artifacts, CI-enforced) · `scripts/hardening_check.sh` dirty-tree-fingerprint receipts · `tasks/lessons.md` Correction→Rule · `docs/distributed-architecture.md` as the model negative-scope doc · Decision Rules / Escalation Triggers / Retry budget in `todo.md`.

---

## 4. Research principles honored
Anthropic "Writing effective tools for agents": consolidate rather than 1:1 wrap; namespace; natural-language identifiers; `response_format` concise|detailed; paginate/truncate with steering; errors that say what to do next; describe tools like onboarding a new hire; run an eval loop. Qdrant MCP: two tools + server-side embedding is enough for most agents. MCP 2025-06-18: `outputSchema`/`structuredContent`, `readOnlyHint/destructiveHint/idempotentHint/openWorldHint`, resources, `resource_link`. Agent-memory systems: extract→consolidate→retrieve, scopes, temporal validity, source attribution. "Error messages are prompts."

---

## 5. Redesign thesis — the tower

**Stance.** I am the agent driving this. In order, I need to: (1) learn the system from one place; (2) speak text and get ranked text back; (3) be told the next call when I'm wrong; (4) know how confident each answer is; (5) have six verbs, not sixty; (6) have my use of it make it better; (7) never be lied to by a doc or a status. Every element below serves one of those seven; anything that doesn't is out.

### 5.1 The tower (bottom → top)
```
L6 Truth       tasks/gates.json ─render→ docs/PRE_RELEASE_STATUS.md · docs/ARCHITECTURE.md (machine-read non-goals) · docs/decisions/ · scripts/check_docs_contract.py R1–R11
L5 Agent       cmd/deepdata-mcp — 6 memory verbs whose inputSchemas ARE the L2 files · resources deepdata://contract, deepdata://status
L4 Clients     sdk/python (pydantic fields ≡ L2, CI-tested) · api/gen gRPC stubs (generated from proto, diff-gated)
L3 Transports  HTTP /v3 + gRPC deepdata.v3 — thin; share internal/apierror, cmd/deepdata/embed_text.go, serverRuntime (auth+limits)
L2 Contract    api/contract/v3/{schemas/*.json, operations.json, CONTRACT.md} — THE source · GET /v3/status is its runtime projection
L1 Engine      internal/collection: Collection (hnsw|flat|inverted, fusion, filters, UsageTracker) · typed sentinels in limits.go · vectors only
L0 Durability  DurableStore: journal + snapshot v2 (class A) · usage.json sidecar (class B) · indexes (class C)
```
**Invariant — one source, N projections, every projection checked in CI.** L1↔L2: `cmd/deepdata/contract_test.go` reflects json tags of request/response structs against schema `properties` and asserts `Canonical*` limits == schema `maximum`. L3↔L2: `operations.json` `grpc_rpc` set == proto service methods (kills the 9-vs-11 class) + existing canonical-surface tests. L4↔L2: `sdk/python/tests/test_contract.py` (`model_fields` == properties; catches today's `TenantIndexConfig` missing `segments`). L5↔L2: by construction (`contract.Schema(name)`) + tool-name set test. L6↔all: R1–R11 + generated blocks + evidence-bound gates. Nothing is hand-maintained in two places.

### 5.2 Durability classes — the rule that lets accretion exist without endangering the core
| Class | Contents | On corruption | Where |
|---|---|---|---|
| **A canonical** | six journal mutations, snapshot v2, schema incl. the new `embedding` binding | **fail closed** (unchanged) | `journal.go`, `snapshot.go` |
| **B accreted signal** | `usage.json` frecency (later: feedback weights) | **loud discard**: error log + `/v3/status.signals.usage.loaded=false`; the collection stays up | `<collection>/usage.json`, written with each snapshot and on `Close` |
| **C derived** | HNSW/flat/inverted indexes | rebuild from A on load (already true) | memory |

Why B ≠ A: a ranking *hint* that fails closed takes correct-by-similarity answers down with it. (Overrides A's "fail closed like the rest of the store"; adopts B's class rule.)

### 5.3 Where the architects disagreed — picked, not averaged
| Topic | Picked | Rejected (flagged) | Why |
|---|---|---|---|
| Contract source | **A**: hand-written JSON Schema in `api/contract/v3/`, `//go:embed`, drift tests each direction | B: `cmd/contractgen` (~180 LoC) from Go types → `contract.json`/`openapi.yaml`/`_contract.py` + diff-gate | MCP and pydantic need JSON Schema anyway; a generator is code + a diff-gate for the same bytes. Grafted from B: limits asserted equal in `contract_test.go`. Revisit B if a third client language appears. OpenAPI: skip until a human consumer exists. |
| Quota-limit status | **409 / `FailedPrecondition`, code `quota_exceeded`, `retryable:false`** (B, and §3 #11) | A: 403 | 403 conflates with auth (remediation differs: credentials vs. delete a collection / raise limits). Both are non-retryable, which is the real fix. |
| Embedding-mismatch status | **409 / `FailedPrecondition`, `embedding_mismatch`** (B) | A: 400 | The request is well-formed; server state can't honor it. Same class as quota. |
| Accretion store | **A**: durable `UsageTracker` sidecar (~80 LoC) | B: `internal/collection/signal/` package (~250 LoC) | One signal doesn't need a package; the *class rule* (5.2) is what matters and is kept from B. |
| Glyph encoding | **retire `internal/encoding`** (A) | B: "maybe, cheap" | Non-deterministic map order, zero consumers; `response_format` + `max_chars` is the token control. |
| Embedders kept | **A**: Ollama (default), OpenAI-compatible (already in `main.go:1606`), ONNX opt-in build tag, `hash` explicit-only | B: Ollama+ONNX only | The OpenAI adapter already exists in stdlib Go; deleting it saves nothing. Both agree: retire `embedder_providers.go` (5 vendors). |
| Pagination | **A**: none; `max_chars` truncation + steering hint; re-query with larger `top_k` (cap 1000) | cursor | Engine has no offset; `pageCursor` (`server.go:3435`) serves legacy list only. |
| Module rename → `deepdata` | **not done** | B | Cosmetic; touches every import path; `why-vectordb.md` rationale stands. Drop the `Canonical` prefix instead (sed, after retire). |
| Feedback loop | **A**: don't revive `internal/feedback` (string IDs, own WAL, LLM sentiment) | B: revive as class B | `UsageTracker` is already the ranking signal; make it durable. |
| Docs graph | **C**'s linter + generated blocks | B: `scripts/gen_graph.sh` mermaid | Rung 1: a diagram nobody's test reads. |

### 5.4 Text in, text out (need 2)
- `VectorField.Embedding *EmbeddingConfig{Provider, Model, Dim}` (`json:"embedding,omitempty"`, `internal/collection/types.go`) — **journaled with the schema**: `durableCreateCollection.Schema` (`durable_store.go:41-43`) is JSON, so `omitempty` is replay-compatible with every existing journal. `validateCanonicalSchema` (`durable_store.go:579`) fills `Dim` from the server embedder when 0; `dim` mismatch → 400; sparse fields may bind `{provider:"bm25"}` only.
- Requests gain `texts map[string]string` keyed by field name (identical on HTTP/gRPC/SDK). Proto `text` is `reserved` in three messages (`deepdata.proto:109-110,:124-125,:212-213`) → the name is `texts`; `SearchRequest = 12`, `InsertRequest = 7`, next free in `BatchDoc`/`UpsertRequest`; `EmbeddingConfig embedding = 6` in `VectorFieldConfig`. Same field in `texts` and `vectors` → 400 `field:"texts.<name>"`; unbound field → 400 + hint "bind an embedding on this field or send a vector"; server embedder ≠ binding → 409 `embedding_mismatch`.
- `cmd/deepdata/embed_text.go` — `resolveTexts(schema, emb, serverCfg, texts, vectors, isQuery)`, shared by HTTP and gRPC, **above the engine** (engine stays vectors-only). Dense: `EmbedQuery` for queries, `Embed` for docs. Sparse `bm25`: export the term-hash path (`migration.go:194`, `hashTerm` `:237`) as `vcollection.TextToSparse(text, dim)` — deterministic, so any client can reproduce it.
- One embedder per process: `DEEPDATA_EMBEDDER=none|ollama|openai|onnx|hash` (default `none`; `hash` is never implicit). Replaces `NewHashEmbedder(1)` at `main.go:3275` — the plumbing to `newCanonicalHTTPHandler(store, swappableEmbedder, …)` (`:3346`) already exists; `collection_http.go` simply never reads it. Configured-but-unreachable at startup → exit 1 (same pattern as the persistence refusal at `:3315`); `none` + `texts` → 503 `embedder_unavailable` with hint. `onnx_impl.go` loses its silent hash fallback; `/readyz.embedder_initialized` (`server.go:1647`) becomes truthful. `embedder_mode.go:261-267` and `scripts/fetch_bge_small.sh:4` agree on one model path.
- Responses carry `embedded_by {field: "provider:model"}` (omitted when the caller sent vectors). The server stores text only where told — callers (and MCP `remember`) put it in `metadata.text` explicitly.
- No hidden default fallback strategy on HTTP (error + hint); the MCP tool supplies the dense→sparse default and *says so in its description*.

### 5.5 Errors are prompts (need 3)
`internal/apierror`: `Error{Code, Message, Hint, Field, RequestID, Retryable, RetryAfterMS}`; `FromEngine(err)` is the **single** `errors.Is` table; `WriteHTTP` (sets `Retry-After`, reads `logging.RequestIDKey`); `GRPCStatus` attaches `errdetails.ErrorInfo{Reason: code, Domain: "deepdata", Metadata{hint,field,request_id}}` + `RetryInfo` (genproto already in `go.sum`).

| code | HTTP / gRPC | retryable |
|---|---|---|
| `invalid_argument` | 400 / InvalidArgument | no |
| `not_found` | 404 / NotFound | no |
| `already_exists` | 409 / AlreadyExists | no |
| `unauthenticated` · `permission_denied` | 401 / 403 | no |
| `quota_exceeded` (tenant/collection limits) | **409 / FailedPrecondition** | **no** (was 429 — §3 #11) |
| `embedding_mismatch` | 409 / FailedPrecondition | no |
| `payload_too_large` | 413 / ResourceExhausted | no |
| `rate_limited` | 429 / ResourceExhausted + `Retry-After: 1` | yes, `retry_after_ms:1000` |
| `embedder_unavailable` · `unavailable` (persistence fault) | 503 / Unavailable | yes |
| `internal` | 500 / Internal | no |

Root cause first: add `ErrCollectionNotFound`, `ErrDocumentNotFound`, `ErrCollectionExists`, `ErrInvalidArgument` beside `ErrInvalidSearchArgument` in `limits.go`; `%w` at `manager.go:80,102,148,293,309,368` and the five bare `fmt.Errorf` at `collection.go:511-523`; delete `strings.Contains("already exists")` (`collection_grpc.go:172`) and the duplicated pre-validation (`collection_grpc.go:279-302`, `collection_http.go:1063-1079`). `writeCanonicalOperationError` becomes three lines; `canonicalGRPCError` becomes `apierror.GRPCStatus(apierror.FromEngine(err))`; a 10-line gRPC interceptor propagates `x-request-id`. MCP decodes the envelope → `isError:true`, `content[0].text = "not_found: … Hint: …"`, `structuredContent` = envelope. SDK `APIError` gains `code/hint/field/request_id/retry_after`; `should_retry` honours `retryable`.

### 5.6 Every answer carries its own confidence (need 4)
Engine `SearchResponse` gains `score_direction` (`"lower_is_better"|"higher_is_better"`, set in `finalizeSearch` from the existing `lowerIsBetter`; hybrid fused → higher) and finally sets `query_time_ms` (`types.go:461`, currently never assigned — §3 #6). Transport search response: `+ score_direction, query_time_ms, embedded_by, request_id` (proto `SearchResponse` 6–9; SDK response models `extra="ignore"` so an older SDK survives a newer server). `CollectionInfo.Fields` → `[]FieldInfo{VectorField; ScoreDirection}` (flat JSON; proto `VectorFieldConfig.score_direction = 7`, output-only). Fixes §3 #9 at the root: the direction is reported, not documented. **No** explain mode, cursor, or HTTP truncation flags (413 + hint is the flag).

### 5.7 Six verbs, not sixty (need 5) — `cmd/deepdata-mcp` rewrite
| Tool | Maps to | Annotations |
|---|---|---|
| `deepdata_recall` | `POST …/search` — `query` (text) or `queries` (vectors); `top_k` ≤ 50; `filters`; `score_floor`; `fallback` (default dense→sparse when both bound, stated in description); `usage_boost`; `response_format concise\|detailed`; `max_chars` | readOnly, idempotent |
| `deepdata_remember` | items 1..100 `{text\|vectors, metadata, id?}` → `POST /docs` · `/docs/batch` · `PUT /docs/{id}` when `id` set; text also stored at `metadata.text` | non-destructive |
| `deepdata_forget` | `DELETE /docs {doc_id}` | destructive, idempotent |
| `deepdata_get` | `GET /docs/{id}` over `ids` 1..50 | readOnly |
| `deepdata_collections` | `GET /collections` or `/collections/{name}` (returns `FieldInfo` incl. `score_direction`, `embedding`) | readOnly |
| `deepdata_create_collection` | `POST /collections` with `preset:"memory"` or explicit `fields` | non-destructive |

Rules: tenant is process-bound (`DEEPDATA_TENANT`, credential-scoped), never an argument; `collection` defaults to `DEEPDATA_COLLECTION=memory`; inputSchemas are `contract.Schema(name)` verbatim (filter `$defs` lists the 15 operators from `filter.go:26-40` + `$and/$or/$not`); `outputSchema` + `structuredContent` on every tool; concise mode elides metadata strings > 300 chars then drops tail hits with `truncated:true` + hint naming the omitted ids; vectors are never returned over MCP; per-process `CollectionInfo` cache (drop on any 4xx). Resources: `deepdata://contract` (markdown, embedded `CONTRACT.md`), `deepdata://status` (proxy of `/v3/status`). **Not added**: tenant/admin tools, `delete_collection`, `ef_search`/`hybrid_params`/`include_vectors` knobs, explain, cursors, HTTP transport, prompts/sampling, Glyph. `./cmd/deepdata-mcp` joins the CI package list (`ci.yml:60-76`).

### 5.8 Learn the system from one place (need 1)
- **Runtime**: `GET /v3/status` (added to `canonicalRCSurface` + `normalizeMetricsPath`; read-gated): `version`, `contract{http,grpc,mcp}`, `embedding{provider,model,dim,available}`, `limits` (from `Canonical*` consts + env RPS/burst), `capabilities{texts,filters[15],hybrid[rrf,weighted,linear],fallback,usage_boost,index_types}`, `signals{usage{loaded}}`, `request_id`. Nothing hand-maintained.
- **Static**: `docs/ARCHITECTURE.md` (one screen: tower, layer→truth file→proven by→documented in, machine-read `<!-- non-goals -->` block); `api/contract/v3/CONTRACT.md` (the agent-facing contract, also the MCP resource).
- **Permission fix** (§3 #4): collection list/get become **read**-gated (`collection_http.go:409,:426`) so a least-privilege agent can discover.

### 5.9 Using it makes it better (need 6)
`UsageTracker.Export()/Import()` (`usage.go`) + `DurableStore` writes `usage.json` at snapshot and `Close`, loads at open per class B (~80 LoC). `ponytail:` comment names the ceiling (whole-file rewrite; move into snapshot format past ~250k entries). Deferred: `POST /docs/{id}/feedback {signal:±1}` (a one-line `Touch` when a harness asks); journaling search hits (never — pollutes the mutation journal).

### 5.10 The keystone and the cut
1. Extract `serverRuntime{apiToken, jwtMgr, requireAuth, acl, quotas, canonicalTenantRL, authFailureRL}` from `VectorStore` (`main.go:83-101`); rewire the gRPC interceptor (`main.go:~3425`); stop constructing `NewVectorStore` (`:3318`). ~150 LoC moved, zero behavior change; every canonical test must stay green.
2. Tag `archive/pre-narrowing-v2` at the pre-cut commit; then one commit on `develop` deletes the RETIRE rows of §7.1 (~55k LoC), rewrites `cmd/deepdata/server_test.go:76-110` with `net/http`, `go mod tidy`.
3. Index vocabulary: one `IndexTypes` slice in `types.go` read by `ParseIndexType` (`:174`), `createDenseIndex` (`collection.go:198-209`), `validateCanonicalSchema` (`durable_store.go:589`); ordinals never renumbered.
4. Then `sed` the `Canonical` prefix away and add a 5-line `doc.go` to each live package.

**Assumption stated for review (not blocking):** "ubiquity" = reachable from any agent or runtime through MCP/HTTP/gRPC/SDK — not "ships a desktop app". `desktop/`, web-UI embed, `tests/ui` are therefore RETIRE (archive branch, git-reversible). If they must stay, flip those three rows to KEEP-DORMANT; nothing else in this plan changes. ADR 0006 stays `proposed` until the owner accepts.

---


---

# Part 2 — architect A — agent-in-the-driver's-seat control surface (Claude Fable)

# DeepData agent control surface — design and implementation plan (perspective: the agent in the driver's seat)

## 0. Corrections to the brief (verified in code)

| Brief says | Code says | Consequence |
|---|---|---|
| `embedder_providers.go` = Gemini/OpenAI/Ollama/ONNX/hash | `/home/omen/Documents/Project/DeepData/cmd/deepdata/embedder_providers.go` is Gemini/Voyage/Jina/Cohere/Mistral. `Embedder` interface + Hash/OpenAI/Ollama live in `cmd/deepdata/main.go:1515-1800`; ONNX is `onnx_impl.go` behind `//go:build onnx` (CGO). CI runs `CGO_ENABLED=0`. | ONNX cannot be the CI-proven first path. Ollama (stdlib HTTP, already probes `/api/tags` in `embedder_mode.go`) is the cheapest revival. |
| Embedder is dormant | `main.go:3272-3275`: canonical mode constructs `NewHashEmbedder(1)` as a placeholder and injects it into `newCanonicalHTTPHandler(store, swappableEmbedder, ...)` (`main.go:3346`). `CollectionHTTPServer` never reads it (no `embedder` reference in `collection_http.go`). | The plumbing exists; only the selection and the read side are missing. |
| Add `text` to proto | `deepdata.proto:109-110`: `reserved 4; reserved "text";` in InsertRequest (and BatchDoc/UpsertRequest). | The name `text` is reserved too. Canonical name becomes `texts` (map field → plural matches `vectors`/`queries`). |
| `next_cursor` on search | Engine has no offset/cursor; `pageCursor`/`encodePageToken` (`server.go:3435-3479`) serve the legacy list path only. | Search pagination is speculative (ladder rung 1). Re-query with larger `top_k` (cap 1000). Skip. |
| Revive `internal/feedback` | String IDs, own WAL, LLM sentiment. Canonical IDs are `uint64`; `UsageTracker` (`internal/collection/usage.go`) already is the ranking signal, just non-durable. | Do not revive. Make the existing tracker durable. |
| MCP not in CI | Confirmed: `.github/workflows/ci.yml:60-76` package list omits `./cmd/deepdata-mcp`; the `./...` at line 80 is scoped to `internal/index/hnsw`. | Add to list in the MCP slice. |
| Text-to-sparse needs new code | `internal/collection/migration.go:175-231` already has `generateSparseVector` (tokenize + `hashTerm % dim`, tf values). | Rung 2: export it as `TextToSparse`. |
| Go client/CLI "dead" | `client/` is imported once: `cmd/deepdata/server_test.go:14,76-110` (legacy V1 test). `cmd/cli` imports nothing else. | Deletion needs one test rewritten with `net/http`. |

Existing untyped error hack to remove: `collection_grpc.go` uses `strings.Contains(err.Error(), "already exists")`; `manager.go:80,102,148,293,309,368` return plain `fmt.Errorf("... not found")`.

---

## 1. Text-in / text-out

### Decision
- **Provider order**: `ollama` first (stdlib, CGO-free, existing `OllamaEmbedder` + `/api/tags` probe), `openai` second (already in main.go, env key), `onnx` third as opt-in build tag with the silent hash fallback removed, `hash` allowed only as an explicitly named provider for CI/integration (never implicit). Remote vendors in `embedder_providers.go` stay dormant.
- **One embedder per server process.** No registry (rung 1 — nothing needs two models today). Env: `DEEPDATA_EMBEDDER=none|ollama|openai|onnx|hash` (default `none`), reuse existing `OLLAMA_URL`, `OLLAMA_EMBED_MODEL`, `OPENAI_API_KEY`. Replaces the `NewHashEmbedder(1)` placeholder at `main.go:3272-3275`. If configured and unreachable at startup: log and `os.Exit(1)` (same pattern as persistence refusal at `main.go:3315`). `none` + text request → `503 embedder_unavailable` with hint.
- **Config lives on the field, journaled with the schema.** `internal/collection/types.go`:
  ```go
  type EmbeddingConfig struct {
      Provider string `json:"provider"`          // "ollama" | "openai" | "onnx" | "hash" | "bm25"
      Model    string `json:"model,omitempty"`   // e.g. "nomic-embed-text"; "" for bm25
      Dim      int    `json:"dim,omitempty"`     // resolved by server at create time when 0
  }
  // VectorField gains:
  Embedding *EmbeddingConfig `json:"embedding,omitempty"`
  ```
  Journal record `durableCreateCollection.Schema` (`durable_store.go:41-43`) is JSON, so `omitempty` is replay-compatible with existing journals. `validateCanonicalSchema` (`durable_store.go:579`) gains: dense field with `embedding` and `Dim==0` → fill `Dim` from server embedder (server passes its `EmbeddingConfig` into the store at open); `embedding.dim != field.Dim` → `ErrInvalidArgument`; sparse field may bind `{provider:"bm25"}` only.
- **Request shape (identical on HTTP, gRPC, SDK)**: `texts map[string]string` keyed by field name, alongside optional `vectors`/`queries`. Same field in both → `400 invalid_argument`, `field:"texts.<name>"`. Field with `texts` but no `embedding` binding → `400 invalid_argument`, hint "bind an embedding on this field or send a vector". Binding provider/model differs from the server's → `400 embedding_mismatch`.
  - Insert `POST .../docs`: `{id?, texts?, vectors?, metadata?}`; batch docs same; upsert `PUT .../docs/{id}` same.
  - Search: `{texts?: {dense:"...", sparse:"..."}, queries?: {...}, top_k, filters, score_floor, fallback, usage_boost, ...}`. `texts` entries are embedded then merged into `queries`; all existing validation (`CanonicalMaxSearchFields=2`, fallback/hybrid exclusivity at `collection.go:538`) applies unchanged.
  - Proto: `map<string,string> texts = 12;` in SearchRequest, `= 7` in InsertRequest, next free number in BatchDoc/UpsertRequest; `EmbeddingConfig embedding = 6;` in `VectorFieldConfig`.
- **Where embedding happens**: transport layer, not engine (engine stays vectors-only). New `/home/omen/Documents/Project/DeepData/cmd/deepdata/embed_text.go`:
  ```go
  // resolveTexts embeds each texts[field] with the field's bound provider and writes into vectors.
  // Returns embedded_by (field -> "provider:model"). Shared by HTTP and gRPC handlers.
  func resolveTexts(schema *vcollection.CollectionSchema, emb Embedder, serverCfg EmbeddingConfig,
      texts map[string]string, vectors map[string]*vcollection.Vector, isQuery bool) (map[string]string, error)
  ```
  Dense: `emb.EmbedQuery` for queries, `emb.Embed` for docs (Ollama nomic prefixes already differ). Sparse `bm25`: `vcollection.TextToSparse(text, field.Dim)` — exported from `migration.go` (new `internal/collection/textsparse.go`, ~30 LoC, deterministic so any client can reproduce it).
- **Response says who embedded**: `embedded_by: {"dense":"ollama:nomic-embed-text","sparse":"bm25"}` on insert/upsert/batch and search responses (omitted when caller supplied vectors). Proto `map<string,string> embedded_by`.
- **Storing text**: the server never writes text anywhere but where told. Callers (and the MCP `remember` tool) put it in `metadata.text` explicitly.
- **Fail-loud fix**: `onnx_impl.go` — remove the `HashEmbedder` fallback on runtime error; return the error. `/readyz` check `embedder_initialized` (`server.go:1647`) becomes truthful: true only when `DEEPDATA_EMBEDDER != none` and the probe passed.

### Push back
Server-side default fallback ("if two bound fields and no strategy, assume dense→sparse") is hidden magic. Keep HTTP explicit (error with hint); put the default in the MCP tool, whose description states it.

---

## 2. MCP tool set

### Decision: six namespaced tools, memory verbs, stdio only
Rename from `search/insert/upsert/get_document/list_collections` (RC, no compatibility owed). Prefix `deepdata_` because a client will have 30+ tools from several servers. Memory verbs because that is what an agent intends; each description states the exact HTTP operation so nothing is hidden.

| Tool | Maps to | Annotations |
|---|---|---|
| `deepdata_recall` | `POST /v3/tenants/{t}/collections/{c}/search` | readOnly, idempotent, openWorld=false |
| `deepdata_remember` | items without `id` → `POST .../docs/batch` (or `/docs` for one); with `id` → `PUT .../docs/{id}` | readOnly=false, destructive=false, idempotent=false |
| `deepdata_forget` | `DELETE .../docs` `{doc_id}` | destructive=true, idempotent=true |
| `deepdata_get` | `GET .../docs/{id}` fanned out over `ids` (1..50) | readOnly, idempotent |
| `deepdata_collections` | `GET .../collections` or `GET .../collections/{name}` when `name` given | readOnly, idempotent |
| `deepdata_create_collection` | `POST .../collections` with `preset:"memory"` or explicit `fields` | readOnly=false, destructive=false, idempotent=false |

Tenant is process-bound (`DEEPDATA_TENANT`, credential-scoped), never a tool argument. `DEEPDATA_COLLECTION` (default `memory`) is the default for `collection`. Raw `vectors`/`queries` remain as typed optional escape hatches so an agent with its own embedding tool still works.

### `deepdata_recall` inputSchema (fragment; `$defs` shared across tools from `api/contract/v3/schemas/`)
```json
{
  "type": "object", "additionalProperties": false,
  "properties": {
    "query":       {"type":"string","minLength":1,"maxLength":8000,
                    "description":"Natural-language query. Embedded server-side into every field of the collection that has an `embedding` binding. Prefer this over `queries`."},
    "queries":     {"type":"object","additionalProperties":{"oneOf":[{"$ref":"#/$defs/dense"},{"$ref":"#/$defs/sparse"}]},
                    "description":"Precomputed vectors keyed by field name. Only if you already have vectors from the collection's bound model."},
    "collection":  {"type":"string","pattern":"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$","description":"Defaults to DEEPDATA_COLLECTION."},
    "top_k":       {"type":"integer","minimum":1,"maximum":50,"default":5},
    "filters":     {"$ref":"#/$defs/filter"},
    "score_floor": {"type":"number","description":"Drop hits worse than this. Direction follows score_direction of the searched field: dense = distance (keep <= floor), sparse/hybrid = similarity (keep >= floor). 0 disables."},
    "fallback":    {"type":"object","additionalProperties":false,"required":["primary","secondary"],
                    "properties":{"primary":{"type":"string"},"secondary":{"type":"string"},"threshold":{"type":"number"}},
                    "description":"Try primary field; if zero hits or best score worse than threshold, try secondary. Default when the collection has one dense and one sparse bound field: {primary: dense, secondary: sparse}."},
    "usage_boost": {"type":"number","minimum":0,"exclusiveMaximum":1,"default":0,
                    "description":"Blend recency-of-use into ranking (0 = pure similarity). 0.2-0.3 is a sensible memory setting."},
    "response_format": {"type":"string","enum":["concise","detailed"],"default":"concise"},
    "max_chars":   {"type":"integer","minimum":500,"maximum":40000,"default":6000,
                    "description":"Output budget for concise mode; long metadata values are elided and tail hits dropped, with `truncated:true` and a `hint`."}
  },
  "oneOf":[{"required":["query"]},{"required":["queries"]}],
  "$defs": {
    "dense":  {"type":"array","items":{"type":"number"},"minItems":1},
    "sparse": {"type":"object","additionalProperties":false,"required":["indices","values"],
               "properties":{"indices":{"type":"array","items":{"type":"integer","minimum":0}},
                             "values":{"type":"array","items":{"type":"number"}}}},
    "filter": {
      "type":"object",
      "description":"Metadata filter. {field: {$op: value}} with ops $eq $ne $gt $gte $lt $lte $in $nin $contains $startswith $endswith $regex $exists $geo_radius{lat,lon,radius_km} $geo_bbox{top_left,bottom_right}; combine with $and/$or (arrays) and $not.",
      "patternProperties": {
        "^\\$(and|or)$": {"type":"array","items":{"$ref":"#/$defs/filter"}},
        "^\\$not$":      {"$ref":"#/$defs/filter"},
        "^[^$].*$":      {"type":"object","minProperties":1,
                          "patternProperties":{"^\\$(eq|ne|gt|gte|lt|lte|in|nin|contains|startswith|endswith|regex|exists|geo_radius|geo_bbox)$":{}},
                          "additionalProperties":false}
      },
      "additionalProperties": false
    }
  }
}
```
(Op list is verbatim from `internal/filter/filter.go:26-40` + `$and/$or/$not` at 690-706.)

### `deepdata_recall` outputSchema / concise `structuredContent`
```json
{"hits":[{"id":42,"score":0.13,"metadata":{"text":"...","source":"..."}}],
 "count":5,"best_score":0.13,"score_direction":"lower_is_better","weak_match":false,
 "fell_back_to":"","truncated":false,"hint":"","request_id":"…"}
```
`content[0].text` carries the same JSON (spec requires text alongside `structuredContent`). `detailed` = raw server JSON passthrough (so fields added later in HTTP surface without MCP edits) minus vectors — vectors are never returned over MCP.

### `deepdata_remember`
```json
{"required":["items"],"properties":{
  "collection":{"type":"string"},
  "items":{"type":"array","minItems":1,"maxItems":100,"items":{
    "type":"object","additionalProperties":false,
    "properties":{"text":{"type":"string","minLength":1,"maxLength":32000},
                  "metadata":{"type":"object"},
                  "id":{"type":"integer","minimum":1,"description":"Set to overwrite an existing memory (upsert)."},
                  "vectors":{"$ref":"#/$defs/vectors"}},
    "oneOf":[{"required":["text"]},{"required":["vectors"]}]}}}}
```
Behaviour stated in description: `text` is sent as `texts` for every bound field and stored under `metadata.text`. Output: `{stored:[{id, embedded_by}], count}`.

### Pagination / truncation with steering
No cursor (see §0). `concise` mode enforces `max_chars`: elide metadata string values > 300 chars ("…"), then drop hits from the tail; set `truncated:true`, `hint:"3 of 8 hits omitted: lower top_k, add filters, or deepdata_get ids [45,46,47]"`. `deepdata_collections` output is small; no paging.

### Resources (two static, no templates)
- `deepdata://contract` — `text/markdown`, the contract doc embedded via `api/contract` (see §4).
- `deepdata://status` — `application/json`, proxy of `GET /v3/status`.
Skip resource templates for per-collection schema: `deepdata_collections {name}` already returns it, and tool calls are the reliable channel across clients.

### Server-side knowledge the MCP needs
Per-process cache `map[collection]CollectionInfo` (fill on first use via `GET .../collections/{name}`, drop on any 4xx for that collection). Used to: pick bound fields for `texts`, build default `fallback`, and set `score_direction` in concise output before the server emits it (S3 makes the server authoritative).

### What NOT to add
Tenant/admin tools; `delete_collection` (humans, via HTTP); `ef_search`, `include_vectors`, `hybrid_params` knobs (MCP `fallback` is enough — the current tool forwards `include_vectors` unschema'd, drop it); explain mode; cursors; MCP over HTTP; prompts/sampling capabilities; Glyph encoding of results (non-deterministic map order, no consumer — `concise` + `max_chars` is the token control).

---

## 3. Errors as prompts

### One envelope
```json
{"code":"not_found","message":"collection notes not found for tenant mcp",
 "hint":"Call deepdata_collections to list existing collections, or deepdata_create_collection to create it.",
 "field":"collection","request_id":"5f3c…","retryable":false,"retry_after_ms":0}
```
Go type in new package `/home/omen/Documents/Project/DeepData/internal/apierror/apierror.go`:
```go
type Code string
const (
    InvalidArgument     Code = "invalid_argument"      // 400 / InvalidArgument
    NotFound            Code = "not_found"             // 404 / NotFound
    AlreadyExists       Code = "already_exists"        // 409 / AlreadyExists
    Unauthenticated     Code = "unauthenticated"       // 401 / Unauthenticated
    PermissionDenied    Code = "permission_denied"     // 403 / PermissionDenied
    QuotaExceeded       Code = "quota_exceeded"        // 403 / ResourceExhausted, retryable=false
    PayloadTooLarge     Code = "payload_too_large"     // 413 / ResourceExhausted
    RateLimited         Code = "rate_limited"          // 429 / ResourceExhausted, retryable=true, Retry-After
    EmbeddingMismatch   Code = "embedding_mismatch"    // 400 / FailedPrecondition
    EmbedderUnavailable Code = "embedder_unavailable"  // 503 / Unavailable, retryable=true
    Unavailable         Code = "unavailable"           // 503 / Unavailable, retryable=true (persistence fault)
    Internal            Code = "internal"              // 500 / Internal
)
type Error struct {
    Code Code `json:"code"`; Message string `json:"message"`; Hint string `json:"hint,omitempty"`
    Field string `json:"field,omitempty"`; RequestID string `json:"request_id,omitempty"`
    Retryable bool `json:"retryable"`; RetryAfterMS int64 `json:"retry_after_ms,omitempty"`
}
func (e *Error) Error() string
func FromEngine(err error) *Error                 // single errors.Is table over vcollection sentinels
func WriteHTTP(w http.ResponseWriter, r *http.Request, e *Error)   // sets Retry-After, request_id from logging.RequestIDKey
func GRPCStatus(e *Error) error                   // status.WithDetails(ErrorInfo{Reason:code, Domain:"deepdata", Metadata:{hint,field,request_id}}, RetryInfo)
```
`errdetails` comes from `google.golang.org/genproto/googleapis/rpc` already in `go.sum:113` (rung 5).

### Typed engine sentinels (root-cause fix)
Add to `/home/omen/Documents/Project/DeepData/internal/collection/limits.go` next to `ErrInvalidSearchArgument`: `ErrCollectionNotFound`, `ErrDocumentNotFound`, `ErrCollectionExists`, `ErrInvalidArgument` (insert/create validation). Wrap with `%w` at `manager.go:80,102,148,293,309,368`, the create "already exists" site, and document validation in `collection.go`. Error strings change slightly; tests asserting on text get updated in the same slice (per `tasks/lessons.md`: "Caller-Validation Errors Must Be Typed at the Engine Boundary").

### Where it lands
- HTTP: `writeCanonicalOperationError` (`collection_http.go`) becomes a 3-liner calling `apierror.WriteHTTP(w, r, apierror.FromEngine(err))`; the ~25 direct `http.Error` sites in tenant handlers route through `apierror.WriteHTTP` with `InvalidArgument` + `field`. `handleTenantInfo` for a nonexistent tenant stays 200 (tenants are implicit) but the hint field is unnecessary there.
- 429 split (`server.go:319-329`): tenant limiter → `RateLimited`, `Retry-After: 1` (`canonicalTenantRL` refills per second), `retry_after_ms:1000`. `ErrTenantLimitExceeded`/`ErrCollectionLimitExceeded` → `QuotaExceeded` **403**, `retryable:false`. Rationale: a hard limit under 429 makes every generic client (incl. our SDK's `_RETRYABLE_STATUS_CODES`) retry uselessly; a distinct status fixes dumb clients, `code` fixes smart ones.
- gRPC: `canonicalGRPCError` (`collection_grpc.go:457-477`) → `apierror.GRPCStatus(apierror.FromEngine(err))`; delete the `strings.Contains("already exists")` hack; the Search pre-validation block (279-302) returns `InvalidArgument` with `field`. Add a 10-line unary interceptor that reads/creates `x-request-id` metadata into `logging.RequestIDKey` if none exists.
- MCP: replace `main.go:146-147` (`errors.New(strings.TrimSpace(raw))`) with envelope decode; `errorResult` returns `isError:true`, `content[0].text = "not_found: collection notes not found. Hint: …"`, `structuredContent` = envelope. Non-JSON bodies (proxies) fall back to `{code:"internal", message: raw}`.
- SDK: `errors.py` — `APIError` gains `code, hint, field, request_id, retry_after`; new `ConflictError` (`already_exists`), `QuotaExceededError` (`quota_exceeded`); `classify_error(status, body, headers)` prefers `code`, falls back to status; `_utils.should_retry` uses `err.retryable` and honours `retry_after`. Drop the dead `client/errors.go` docstring reference.

---

## 4. Self-description and one source of truth

### `GET /v3/status` (add to `canonicalRCSurface` and `normalizeMetricsPath`; same auth guard as tenant routes)
```json
{"version":"0.2.0-rc.1",
 "contract":{"http":"v3","grpc":"deepdata.v3.DeepData","mcp":"2025-06-18"},
 "embedding":{"provider":"ollama","model":"nomic-embed-text","dim":768,"available":true},
 "limits":{"max_search_fields":2,"max_top_k":1000,"max_ef_search":4096,"max_batch_documents":10000,
           "max_batch_bytes":12582912,"tenant_rps":100,"tenant_burst":200},
 "capabilities":{"texts":true,"filters":["eq","ne","gt","gte","lt","lte","in","nin","contains","startswith","endswith","regex","exists","geo_radius","geo_bbox"],
                 "hybrid":["rrf","weighted","linear"],"fallback":true,"usage_boost":true,"index_types":["hnsw","flat","inverted"]},
 "request_id":"…"}
```
All values come from the existing constants (`types.go` `CanonicalMax*`, env RPS/burst, `filter.go` operators) — nothing hand-maintained.

### Per-collection schema with score direction
`CollectionInfo.Fields` (`manager.go:181`) becomes `[]FieldInfo`:
```go
type FieldInfo struct {
    VectorField
    ScoreDirection string `json:"score_direction"` // "lower_is_better" | "higher_is_better"
}
```
JSON stays flat (embedding). Proto `VectorFieldConfig.score_direction = 7` (output-only, documented). Computed with the same predicate as `scoreLowerIsBetter` (`collection.go`): dense → lower.

### Single source of truth: JSON Schema files, everything else validated against them
Proto cannot be it (JSON `filters` is an object, proto is `Struct`); pydantic cannot be it (Go does not read Python); Go structs can be it only if we write a reflection→JSON-Schema generator (~150 LoC, rung 7) — not worth it. Hand-written JSON Schema (rung 6: it is data, not code) in a Go-embeddable package:

```
api/contract/contract.go              // package contract; //go:embed v3/*; func Schema(name string) json.RawMessage; Operations() []Operation; ContractMarkdown() string
api/contract/v3/operations.json       // [{name, http_method, http_path, grpc_rpc, mcp_tool, sdk_method}] — 11 rows + status
api/contract/v3/schemas/{search_request,insert_request,upsert_request,batch_request,create_collection_request,filter,error,status,collection_info,search_response}.json
api/contract/v3/CONTRACT.md           // agent-facing doc; also served as MCP resource
```
Drift tests (all CI-run):
- `cmd/deepdata/contract_test.go`: reflect json tags of the HTTP body/response structs == schema `properties`; `deepdatav3.File_*_proto` service methods == `operations.json` `grpc_rpc` set (kills the 9-vs-11 class of drift); proto message JSON names ⊆ schema properties (minus `tenant_id`/`collection` which are path params on HTTP); `apierror.Error` tags == `error.json`.
- `cmd/deepdata-mcp`: tool `inputSchema`s are built **from** `contract.Schema(...)` — drift impossible by construction; test asserts tool names == `operations.json` `mcp_tool` set.
- `sdk/python/tests/test_contract.py`: `TenantSearchRequest.model_fields` etc. == schema properties (path `Path(__file__)/../../../../api/contract/v3/schemas`). This immediately catches the existing `TenantIndexConfig` allowlist missing `segments` (`validateCanonicalIndexParams` accepts it).
- OpenAPI: skip (rung 1). The schemas are 90% of one; wrap later if a human consumer appears.

---

## 5. Search response

Add to the engine `SearchResponse` (`types.go:~450`): `ScoreDirection string json:"score_direction"` set in `finalizeSearch` (it already computes `lowerIsBetter`; hybrid fused path sets `higher_is_better`). Transport response (`tenantSearchJSONResponse`, `collection_http.go:358-367`) becomes:
```json
{"status":"success","tenant_id":"acme","documents":[...],"scores":[...],
 "candidates_examined":7,"best_score":0.13,"weak_match":false,"fell_back_to":"",
 "score_direction":"lower_is_better","query_time_ms":1.8,
 "embedded_by":{"dense":"ollama:nomic-embed-text"},"request_id":"…"}
```
Proto `SearchResponse`: `double query_time_ms = 6; string score_direction = 7; map<string,string> embedded_by = 8; string request_id = 9;`. SDK `TenantSearchResponse` gains the four fields with defaults; ensure response models use `extra="ignore"` (requests keep strict) so an older SDK survives a newer server.

Decisions: no truncation flags on HTTP (the 413 `payload_too_large` envelope with hint "lower top_k or include_vectors=false" is the flag); no `next_cursor` (§0); **no explain mode** — `candidates_examined`, `fell_back_to`, `best_score`, `score_direction`, `query_time_ms`, `embedded_by` are the explanation an agent acts on. Add per-hit fused-score components only when someone asks "why is doc X here" and the hybrid fuser is changed to keep them.

---

## 6. Accretive loop

Minimal durable signal: make `UsageTracker` survive restarts. `internal/collection/usage.go` gains `Export() []UsageEntry{ID uint64; Score float64; LastSeen int64}` / `Import([]UsageEntry)`; `DurableStore` writes `<collection dir>/usage.json` when it writes a snapshot and on `Close`, loads it on open (missing file = empty, malformed = fail closed like the rest of the store). ~80 LoC, no journal format change, loses at most one snapshot interval of frecency — acceptable for a ranking hint.

Defer: explicit `feedback` tool/endpoint (it would be a one-line `tracker.Touch(id, weight)` behind `POST .../docs/{id}/feedback {signal:-1|1}` — add when an agent harness actually asks for it); `internal/feedback` revival (wrong ID type, LLM dependency); journaling every search hit (pollutes the mutation journal). `ponytail:` comment on the sidecar naming the ceiling: "per-collection file rewritten whole; move into snapshot format if usage tables exceed ~250k entries".

---

## 7. Kill or fix: Go client and CLI

**Delete both.** Ladder:
1. Does a Go client need to exist? Go consumers already have a generated, always-current client: `api/gen/deepdata/v3` gRPC stubs (rung 2). `client/` speaks V1 `/insert`/`/query` with the cowrie codec, which the RC refuses to serve (`canonicalRCSurface`).
2. Does a CLI need to exist? For agents the MCP server is the CLI. For humans, `curl` + `CONTRACT.md`. `cmd/cli` is V1 + obsidian import, unreachable in canonical mode.
3. Cost of deletion: rewrite the one importer, `cmd/deepdata/server_test.go:76-110`, with `net/http` + `encoding/json` (~40 LoC, rung 3) or drop that test block if it only proves the walled-off V1 handler. `go mod tidy` (cowrie stays; used by storage/wal).
A thin `cmd/deepdata-cli` over V3 is YAGNI; revisit if a human workflow shows up that MCP/curl cannot serve.

---

## 8. Ordering — slices, each leaving every surface consistent

| # | Slice | Why here | Files (LoC) | Tests |
|---|---|---|---|---|
| **S0** | Errors as prompts | Tiny, unblocks everything (`embedder_unavailable`, `embedding_mismatch` need it); an agent gets hints on day one | `internal/apierror/apierror.go` new (~110); `internal/collection/limits.go` +4 sentinels; `manager.go` 7 wraps, `collection.go` ~15; `collection_http.go` ~80 changed; `collection_grpc.go` ~40; `server.go` ~10; `cmd/deepdata-mcp/main.go` ~30; `sdk/python/deepdata/errors.py` + `_utils.py` ~70 | `agent_retrieval_test.go`: `errors.Is` for not-found/exists. `agent_retrieval_http_test.go`: 404 envelope JSON + `request_id` echo, 429 `Retry-After`, quota 403. `agent_retrieval_grpc_test.go`: `ErrorInfo.Reason`, `RetryInfo`. `main_test.go`: `isError` + `structuredContent.code`. Python `test_errors.py` +3. |
| **S1** | Text-in/text-out (Ollama first) | The single thing that makes an agent able to use the DB at all | `types.go` (+`EmbeddingConfig`, `ScoreDirection`) ~25; `durable_store.go` validation ~25; `textsparse.go` new ~30; `cmd/deepdata/embed_text.go` new ~120; `main.go` embedder selection ~40; `onnx_impl.go` −10; `collection_http.go` ~80; `collection_grpc.go` ~60 (+`embedder` field); proto ~40 + regen; SDK `models.py` ~60; README ~30 | `agent_retrieval_test.go`: `TextToSparse` deterministic, `embedding` schema validation, dim auto-fill. `agent_retrieval_http_test.go`: create with `embedding{provider:hash}`, insert `texts`, search `texts` → hit + `embedded_by`; texts+vectors conflict → 400 `field:"texts.dense"`; `DEEPDATA_EMBEDDER=none` → 503. gRPC mirror. Python: `test_tenant_v3.py` models; `test_integration.py` text round-trip with CI server on `DEEPDATA_EMBEDDER=hash`. |
| **S2** | MCP redesign | Now that text works, give the agent the ergonomic layer | `api/contract/contract.go` + `v3/schemas/*.json` (~250 JSON); `cmd/deepdata-mcp/main.go` rewrite (~500); `ci.yml` +`./cmd/deepdata-mcp ./api/contract` | `main_test.go`: rewrite 5 existing; `tools/list` has `outputSchema`+`annotations` for all 6; recall concise truncation → `truncated`+`hint`; recall sends `texts` + default `fallback` from cached schema; remember routes 1→POST, id→PUT, N→batch; forget → DELETE body; `resources/read deepdata://contract` non-empty; envelope → `isError`. |
| **S3** | Self-description + anti-drift | Agent can introspect; CI proves the tower is one contract | `/v3/status` handler ~60 + allowlist; `manager.go` `FieldInfo` ~15; proto `score_direction`, response fields; `tenantSearchJSONResponse` +4; `api/contract/v3/operations.json`; `cmd/deepdata/contract_test.go` ~120; `sdk/python/tests/test_contract.py` ~40; SDK `client.status()` + fix `segments`; `CONTRACT.md` ~150 | The drift tests are the tests; plus HTTP status shape, collection info `score_direction`, search `query_time_ms>0`. |
| **S4** | Durable usage | Makes `usage_boost` real across restarts | `usage.go` ~40; `durable_store.go` ~40 | `agent_retrieval_test.go`: touch, close, reopen, `usage_boost` still reorders. |
| **S5** | Delete `client/`, `cmd/cli` | Independent; can run any time, last because it is the least agent-enabling | `rm -r client cmd/cli` (−~1500); `server_test.go` rewrite ~40; `go mod tidy`; README/docs refs; `errors.py` docstring | Existing suite green; `go vet ./...`. |

Rough total: ~1.6k LoC added, ~1.5k removed, ~5-6 focused days. Each slice ends with `go test ./cmd/deepdata ./cmd/deepdata-mcp ./internal/collection ./api/contract` and the Python job green.

### Critical Files for Implementation
- /home/omen/Documents/Project/DeepData/cmd/deepdata/collection_http.go
- /home/omen/Documents/Project/DeepData/cmd/deepdata-mcp/main.go
- /home/omen/Documents/Project/DeepData/internal/collection/types.go
- /home/omen/Documents/Project/DeepData/cmd/deepdata/collection_grpc.go
- /home/omen/Documents/Project/DeepData/sdk/python/deepdata/models.py


---

# Part 2 — architect B — systems coherence (Claude Opus)

I read the cited code. Five items in the brief are wrong, and one of them changes the sequencing. Corrections first, then the architecture.

---

# DeepData as one tower — systems-coherence architecture

## 0. Corrections to the shared brief

**0.1 `QueryTimeMs` is not "computed by the engine and dropped by transports." It is never set at all.**
`internal/collection/types.go:461` declares it; the only three construction sites (`internal/collection/collection.go:830`, `:993`, `:2054`) omit it. It serializes as `0` from the engine and is absent from `tenantSearchJSONResponse`. It is a lie in a struct, not a plumbing gap. Consequence: this is a ~6-line fix at one boundary, not a transport fix.

**0.2 The root binaries are not checked in.** `.gitignore` lists `/deepdata`, `deepdata-server`, `cli`. `git ls-files` confirms zero tracked binaries. The 72 MB at repo root is untracked local build output. The *only* oversized tracked blob is `docs/models/bge-small-en-v1.5/tokenizer.json` (711 KB) and the Tauri `gen/schemas/*.json` (~330 KB). Deleting the root binaries is an `rm`, not a git-history problem.

**0.3 `GET /v3/tenants/{id}` returning 200-with-zeros for a nonexistent tenant is correct, not a bug.** There is no create-tenant mutation — the six types at `durable_store.go:17-23` contain none. A tenant exists iff it owns a collection. The response is accurate. What's missing is that nothing *says* so. This is a documentation fix, and it belongs in the contract artifact, not in code.

**0.4 The 429 overload is a worse bug than described.** `writeCanonicalOperationError` (`collection_http.go:648`) maps `ErrTenantLimitExceeded` / `ErrCollectionLimitExceeded` to **429**, and `canonicalGRPCError` (`collection_grpc.go:466-469`) maps them to **ResourceExhausted**. Both signal *retryable* to every standard client policy. These are permanent admission refusals — the tenant cap is `StoreLimits.MaxTenants`, immutable for the process lifetime. An agent will retry forever against a wall. This must be 409 / `FailedPrecondition`.

**0.5 The transports do not "duplicate" validation for want of a classifier — they duplicate it because the engine's own errors are unclassified.** `internal/collection/collection.go:512-523` returns bare `fmt.Errorf` for `queries`-empty, field-count, `top_k`, and `ef_search`; only `:531+` wraps `ErrInvalidSearchArgument`. So an unguarded `top_k=5000` reaching the engine falls through `writeCanonicalOperationError` to **500** and through `canonicalGRPCError` to **Internal**. That is exactly why `collection_grpc.go:279-302` and `collection_http.go:1063-1079` each re-implement the same five checks. **Root cause is five missing `%w` verbs.** Fix them and ~60 lines of transport duplication delete themselves. This is the highest value-per-line change in the repo and it moves to slice 1.

**0.6 The keystone finding the brief missed.** `main.go:3306` calls `NewVectorStore(0, 1)` under `canonicalOnly`. That is not cosmetic. `VectorStore` (`main.go:53-113`) is the *dead legacy engine*, and it is where the **live** RC keeps its auth state: `apiToken`, `jwtMgr`, `requireAuth`, `acl`, `quotas`, `canonicalTenantRL`, `authFailureRL`. `main.go:3421` reads five of those to build the gRPC interceptor. The shipped server allocates a 60-field legacy struct with WAL paths, BM25 lexical stats and a metadata bitmap index **so that seven auth fields have an address**. This is the single reason `main.go` and `server.go` cannot be cut, and it forces the slice order: extract those seven fields before anything else is deletable.

---

## 1. The tower, named

Seven layers. Each has exactly one truth file. Each arrow is either *generation* (mechanical) or *assertion* (a CI check), never "someone remembers."

| # | Layer | Single artifact of truth | Arrow up |
|---|---|---|---|
| 0 | **Substrate** | `internal/collection/journal.go` (CRC frame, store-ID bind, LSN) + `snapshot.go` (v2 stream format) | none — format is frozen, changes need an ADR |
| 1 | **Engine semantics** | `internal/collection/durable_store.go` (six mutation constants, fault latch) | none — transports may not add semantics |
| 2 | **Admission contract** | `internal/collection/types.go` + `limits.go` (constants, errors, `Search` preconditions) | **ROOT OF TRUTH** |
| 3 | **Contract projection** | `api/contract.json`, generated | `go run ./cmd/contractgen` → `git diff --exit-code` |
| 4 | **Transports** | `api/proto/deepdata/v3/deepdata.proto` (gRPC shape), `api/openapi.yaml` (HTTP shape, generated) | proto *asserted* against contract.json; openapi *generated* from it |
| 5 | **Clients** | `sdk/python/deepdata/models.py`, `cmd/deepdata-mcp/main.go` | pydantic bounds asserted against contract.json in pytest; MCP schemas `go:embed` contract.json |
| 6 | **Legibility** | `ARCHITECTURE.md` | contract tables + import graph generated between markers, diff-gated |

### Why Go is the root and not the proto

The proto is a good file with good comments, but it cannot be the root:

- It cannot express the contract. `fallback` XOR `hybrid_params`, `primary != secondary`, `score_floor` direction inverting by field type, `usage_boost ∈ [0,1)` — none of these are proto3-expressible. They are already Go code at `collection.go:505-543`. Rooting in proto means the actual rules live *nowhere* and are copied into three places, which is today's failure.
- The arrow would be inverted. `protoc` generates Go from proto; if proto were root, the engine would derive from generated code.
- Half the arrow already exists and works: `collection_http.go:1068` and `collection_grpc.go:291` both import `vcollection.CanonicalMaxSearchFields`. Extend that, don't replace it.
- Pydantic is a replica, and the proof it drifts is in the tree: `models.py:131-142` still carries `AliasChoices("name","Name")` for a title-case wire format the server abandoned at `bc1fa27`.

So: **Go declares, contract.json projects, everyone else derives or is asserted.**

### `cmd/contractgen` — the one new program

~180 LoC, `main` package, no dependencies beyond stdlib + `internal/collection`. It imports the engine package (legal — it's in-module), reads the exported constants and one hand-written rule table that lives next to them, and writes three files:

```
api/contract.json          # limits, index vocabulary, error taxonomy, search preconditions, ops list
api/openapi.yaml           # OpenAPI 3.1 for the 10 V3 HTTP routes
sdk/python/deepdata/_contract.py   # frozen dataclass of the same numbers
```

The rule table is the only hand-written part, and it lives in `internal/collection/contract.go` as data next to the code that enforces it:

```go
// contract.go — the rules Search enforces, as data. contractgen projects this;
// Search executes it. Adding a rule here without enforcing it fails TestContractRulesAreEnforced.
var SearchPreconditions = []Precondition{
    {ID: "queries_non_empty", Message: "at least one query field is required"},
    {ID: "fallback_xor_hybrid", Message: "fallback and hybrid_params are mutually exclusive"},
    ...
}
```

`TestContractRulesAreEnforced` in `internal/collection` drives one violating `SearchRequest` per entry and asserts `errors.Is(err, ErrInvalidSearchArgument)`. That test is what makes the table honest — without it, the table is documentation and documentation rots.

### CI checks — three steps, all modeled on the existing pattern

The repo already has the right shape twice: `scripts/check_proto_generated.sh` (generate → `git diff --exit-code`) and `scripts/check_version_contract.py` (one source → six artifacts → `raise SystemExit`). Copy both.

1. **`scripts/check_contract_generated.sh`** — `go run ./cmd/contractgen && git diff --exit-code -- api/contract.json api/openapi.yaml sdk/python/deepdata/_contract.py`. Identical body to `check_proto_generated.sh`. Catches: constant changed in Go, artifacts not regenerated.

2. **`scripts/check_contract.py`** — asserts, in `check_version_contract.py` style with `raise SystemExit`:
   - proto `service DeepData` RPC count and names == `contract.json.operations`
   - proto `index_type` comment enumerates exactly `contract.json.index_types`
   - every `<!-- contract:begin:X -->` block in `README.md` and `ARCHITECTURE.md` matches the rendered table
   - every `deepdata/models.py` `Field(...)` numeric bound has a matching `contract.json` limit
   Catches: the 12 documented drifts, all of them, permanently.

3. **`sdk/python/tests/test_contract_parity.py`** — imports `_contract` and asserts each pydantic validator rejects a payload built from each `contract.json` precondition. Catches: SDK admission narrower or wider than the server's.

Add `./cmd/deepdata-mcp` and `./cmd/contractgen` to the CI `go vet` and `go test` package lists. MCP being absent from CI is why it drifted.

---

## 2. Collapse the wall

The right shape is: **there is no wall, because there is nothing on the other side.** `canonicalRCSurface` (`server.go:3414-3423`) is nine lines guarding 50 `mux.Handle` calls and ~30k LoC. Delete the routes; the guard becomes unnecessary by construction.

### Order is forced by 0.6

**Step A — extract the auth runtime (~120 LoC new, keystone).**
New file `cmd/deepdata/runtime.go`:

```go
// serverRuntime is the process-wide state the V3 surface actually needs.
// It replaces the seven fields the RC was borrowing from the legacy VectorStore.
type serverRuntime struct {
    apiToken      string
    jwtMgr        *security.JWTManager
    requireAuth   bool
    acl           *security.ACL
    quotas        *security.TenantQuota
    tenantRL      *rateLimiter          // per-authenticated-tenant
    authFailureRL *authFailureLimiter   // per-peer-IP failed auth
}
```

Rewrite `main.go:3421` and `server.go`'s guard construction to take `*serverRuntime`. Mechanical; no behavior change; provable by the existing `cmd/deepdata` test suite passing unchanged. After this, `VectorStore` has zero live referents.

**Step B — the deletion (one commit, ~14,000 LoC out of `cmd/deepdata`).**

Deleted outright:
`server.go`'s 50 legacy routes • `collection.go` (400, legacy V2) • `collection_handlers.go` • `feedback_http.go` (441) • `extraction_http.go` (362) • `migration.go` (494) • `prefilter.go` (635) • `quantization.go` (606) • `cache.go` (572) • `vector_types.go` (485) • `sparse_distance.go` • `cost_tracker.go` (377) • `mode.go` • `embedder_mode.go` (418) • `onnx_impl.go` / `onnx_stub.go` (399) • `VectorStore` and everything reachable only from it in `main.go` • the legacy test files (`vectordb_test.go` 1662, `benchmark_test.go` 903, `wal_recovery_test.go` 822, `server_test.go` 523, `multitenant_test.go` 571, `persistence_characterization_test.go` 386, `sparse_api_test.go`, `decode_vector_test.go` legacy half).

Renamed and reduced:
- `server.go` 3586 → **`httpserver.go` ~420**. Survives: the five middlewares (`corsMiddleware`, `otelMiddleware`, `requestIDMiddleware`, `recoveryMiddleware`, `requestTimeoutMiddleware`), `/healthz` `/readyz` `/livez` `/metrics`, `guard`, `RegisterCanonicalHandlers` wiring, `canonicalTenantIDFromPath`, `canonicalRateLimitTenant`, `ageMillis`.
- `main.go` 4032 → **~550**. Survives: flag/env parsing, `validateEnvConfig`, `validateCanonicalAuthEnvironment`, listener setup, `canonicalListenerAddresses`, durable store open, gRPC server construction, graceful shutdown.

Deleted because they were only reachable from deleted code: `const canonicalOnly` (23 sites), `NewHashEmbedder(1)` placeholder, `SwappableEmbedder`, `loadOrInitStore`, `existingLegacyRootArtifacts`, `STORAGE_FORMAT` parsing, `canonicalRCSurface`.

**No build tags.** A `//go:build legacy` file is a wall with better marketing — it still compiles in editors, still shows in grep, still costs an agent a read to discover it's dead. Rung 1 of the ladder: does the legacy surface need to exist? No. Git history is the archive. Tag the pre-deletion commit `pre-narrowing-v2` and delete.

`cmd/deepdata` net: **27,756 → ~7,400** tracked Go LoC.

### The one-line fixes (do these in slice 2, before the deletion, so they're independently bisectable)

| Fix | Change | Lines |
|---|---|---|
| V3 metrics | `server.go:3379`: `routed = globalMetrics.HTTPMiddleware(routed)` — `normalizeMetricsPath` already handles `/v3/tenants/:id` (`metrics.go:295-298`) | **1** |
| Unify registries | `metrics.go:213`: `promhttp.HandlerFor(prometheus.Gatherers{mc.registry, prometheus.DefaultGatherer}, ...)` — makes `internal/telemetry`'s seven `deepdata_*` series reachable | **1** |
| gRPC reflection | `main.go:~3425`: `reflection.Register(grpcSrv)` + import | **2** |
| Delete `withMetrics` | falls out with the legacy routes it wrapped | −14 |
| Error taxonomy | five `%w: ErrInvalidSearchArgument` at `collection.go:512-523` | **5** |
| 429 → 409 | `collection_http.go:648`, `collection_grpc.go:466` | **4** |
| `include_vectors` | proto: `optional bool include_vectors` | **1** |

Not one-liners, but small:

- **gRPC health (~25 LoC).** `grpc_health_v1` needs a `health.Server` whose status is driven by the same `PersistenceError()` that `/readyz` reads, so the two can't disagree. Register `""` and `"deepdata.v3.DeepData"`, flip to `NOT_SERVING` on fault latch. Wire it to the existing `persistenceHealth` closure at `main.go:3427`.
- **`QueryTimeMs` (~8 LoC).** Set it once at `TenantManager.SearchCollection`, not in the three `Collection.Search` returns — one site, and it measures what the caller actually waited for. Then `tenantSearchJSONResponse` + one proto field.

---

## 3. Revive vs retire, per tree

Bias: delete. 30k LoC behind a nine-line allowlist is a liability an agent must read past on every exploration. Ladder applied per tree.

### REVIVE-NOW

| Tree | LoC | Why |
|---|---|---|
| **`internal/encoding` (Glyph)** | 238 | Rung 2 — already here, already CI-tested, already imports `vcollection.Document`. 50-62% token savings on exactly the payload an agent consumes. Wire as `Accept: text/glyph` on `POST .../search` and as MCP `content[0].text` alongside `structuredContent`. Cheapest thesis win in the repo. |
| **`internal/collection/usage.go`** | 217 | Live, bounded, decayed, already reachable via `usage_boost`. Promote to the durable signal side-store in slice 10. |
| **`Recommend`/`Discover`** | ~300 | Vector arithmetic over already-indexed docs; no new deps, no schema change, no new mutation type. "More like this / unlike that" is a primitive an agent uses constantly. Route them: `POST .../search/recommend`, `POST .../search/discover`, two new RPCs, two contract entries. |
| **`embedder_providers.go` (Ollama + ONNX-BGE only)** | ~250 of 476 | The thesis is text-in. Keep `OllamaEmbedder` (self-hosted, matches "self-hosted agent memory") and `ONNXEmbedder` (offline, deterministic, no network in the write path). **Retire Gemini + OpenAI providers** — cloud embedders put a network call in the admission path of a fail-closed store; if the model version changes under you, the collection is silently poisoned. Not RC-safe. |

### REVIVE-LATER (behind a named, persisted, per-collection gate)

| Tree | LoC | Gate |
|---|---|---|
| **`internal/graph`** | 943 | Schema flag `graph: {enabled, damping}`. Class C (derived, rebuildable). `hybrid.HybridSearchWithGraph` (`fusion.go:261`) already takes the third result set — the fusion side is done. |
| **`internal/feedback`** | 2,618 | Land the *storage* (slice 10) as the signal side-store; land `booster.go` behind `usage_boost`'s successor. Its own JSON WAL must go — one durability mechanism per class. |
| **`internal/extraction`** | 3,232 | LLM KG extraction is the accretion half of extract→consolidate→retrieve, but it is an *out-of-band* job, not a request path. Move to `cmd/deepdata-extract` (separate binary, reads/writes over the V3 API like any client). Zero coupling to the server. |
| **`docs/models/bge-small-en-v1.5/tokenizer.json`** | 711 KB | Keep — it's the ONNX path's only non-fetchable asset and `scripts/fetch_bge_small.sh` covers the rest. |
| **`sugarme/tokenizer` + `onnxruntime_go`** | dep | Keep, behind `//go:build onnx`. cgo in the default build is a portability tax; a build tag here is legitimate because there are two *real* configurations, not one live and one dead. |

### RETIRE (delete from tree; git history is the archive)

| Tree | LoC | Ladder verdict |
|---|---|---|
| **`internal/index` DiskANN/PQ/IVF/GPU/quant/mmap/shard/lru/payload/binary** | ~25,000 | Rung 1. Every one is an RC non-goal, none is reachable from `createDenseIndex` after the vocabulary fix, `diskann` has no case at all (`collection.go:209` default error). Survivors: `interface.go`, `hnsw.go`, `flat.go`, `segmented.go`, `simd/`, vendored `hnsw/`. Also deletes the `Neumenon/shard` private dep (only importers: `shard_persist.go`, `shard_columns.go`). |
| **`internal/cluster`** | 10,445 | Rung 1. Zero importers, EXPERIMENTAL, single-node RC. Largest dead tree in the repo. |
| **`internal/wal` + `internal/storage` + `internal/cowrieutil`** | 4,400 | Two WAL/snapshot systems is the definition of incoherence. `journal.go`+`snapshot.go` won. `cowrieutil` is imported only by `wal`, `storage`, `cluster` — all three dying. |
| **`Neumenon/cowrie` dep** | — | After the above plus `server.go:133/149/1028` (`codec.FromRequest`) and `client/`, it has **zero importers**. A private, unfetchable dependency in `go.mod` means a cold agent cannot `go build`. Removing it is the single biggest legibility win in `go.mod`. |
| **`client/` (Go)** | 1,000 | 100% legacy V1/V2, imports cowrie. |
| **`cmd/cli`** | 990 | Same. `cmd/gentoken` (153) survives — `gentoken -json` is the one scriptable CLI output and CI uses it. |
| **`internal/obsidian`** | 706 | Zero tests, zero importers, an integration not a capability. If wanted, it's an MCP client, not server code. |
| **`internal/security` dormant half** | ~5,100 | `audit.go` 1082, `rotation.go` 956, `encryption.go` 855, `tls.go` 549 + tests. Verified: `cmd/` uses exactly `ACL, AuthorizationUnauthenticated, AuthorizeTenantAccess, AuthorizeTenantPermission, GetTenantContextFromContext, IsAuthorizationFailure, JWTManager, NewACL, NewJWTManager, NewTenantQuota, SecureCompare, TenantContext, TenantContextKey, TenantQuota` — all in `auth.go` + `rbac.go`. Dormant security code is worse than none: it reads as a guarantee. |
| **`cost_tracker.go` + `mattn/go-sqlite3`** | 377 | Rung 1 — the embedding cost ledger is for cloud providers we're retiring. Deletes the only cgo dep in the default build. |
| **`desktop/`** | 70 files + 3.4 GB artifacts | Rung 1. A Tauri app for a headless single-node server. Delete the directory. |
| **`cmd/deepdata/web-ui` + `tests/ui`** | ~15 Playwright specs incl. `knowledge-graph.spec.ts`, `chat.spec.ts` | Rung 1. Tests for a UI the RC 404s. Actively misleading — they describe features as if shipped. |
| **`vdb-test-suite/`** | Python harness | Targets the legacy API. `benchmarks/` (post-`50fe829` canonical recall harness) is the live replacement. |
| **`docs/benchmarks.md` (the case-collided inflated one), `benchmarks/GAP_ANALYSIS.md`, `internal/index/README.md`, `PRE_RELEASE_STATUS.md`, `STATE.json`, `deepdata-system-map.html`** | ~4 roadmaps + 2 status stores | Four roadmaps and four status stores is not drift, it's the absence of a single truth. Delete all six. `tasks/todo.md` (open work) + `tasks/lessons.md` (append-only Correction→Rule) + `CHANGELOG.md` (shipped) are three roles, no overlap, and `lessons.md` is the one artifact in this repo that structurally cannot rot. |
| **root `deepdata`, `deepdata-server`, `cli`, `baseline_bench.txt`, `phase1a_bench.txt`, `results.json`, `run.sh`, `.env`** | 72 MB | Untracked or stale. `rm`. Add `.gnhf/` to `.gitignore` properly rather than `.git/info/exclude` (an exclude only you can see is a trap for the next agent). |

### ARCHIVE-BRANCH

One branch, `archive/pre-narrowing-v2`, pointed at the commit *before* the deletion, pushed, and named in `ARCHITECTURE.md`. That is the entire archive strategy. Do not create per-tree branches — an agent that finds five archive branches has to read five.

**Net: ~62,000 LoC deleted, ~40,000 remain.** Two private deps and one cgo dep leave `go.mod`.

---

## 4. Reconcile the index vocabulary

Four disagreeing lists today:

- `types.go:174` `ParseIndexType` → `hnsw, ivf, flat, diskann, inverted`
- `collection.go:143-207` `createDenseIndex` → `hnsw, ivf, flat` (**no diskann case**, `:209` default error)
- `durable_store.go:589` `validateCanonicalSchema` → `hnsw, flat` (+`inverted` for sparse)
- `deepdata.proto:19` comment → `"hnsw", "flat", or "inverted"`

The narrowest is correct. **Single list, single place**, in `internal/collection/types.go` next to the enum:

```go
// IndexTypes is the complete vocabulary. ParseIndexType, createDenseIndex,
// validateCanonicalSchema, contractgen, and the proto comment all derive from
// this slice. Adding an entry without a createDenseIndex case fails
// TestEveryIndexTypeConstructs.
var IndexTypes = []IndexType{IndexTypeHNSW, IndexTypeFLAT, IndexTypeInverted}
```

Then:
1. Delete `IndexTypeIVF` and `IndexTypeDiskANN` from the enum. This is a **persisted-value change** — check `UnmarshalJSON`'s integer path at `types.go:141-148` (`n > int(IndexTypeInverted)`); renumbering shifts every persisted integer. **Do not renumber.** Keep the ordinals, delete only the parse cases and the constructor case, and add an explicit rejection so a v2-era journal fails loud rather than silently reinterpreting: `case IndexTypeIVF, IndexTypeDiskANN: return fmt.Errorf("index type %s was removed in 0.3; export and re-create the collection", it)`.
2. `ParseIndexType` iterates `IndexTypes`.
3. `createDenseIndex` loses its IVF branch; its `default` becomes unreachable-for-valid-schemas because `validateCanonicalSchema` is the gate.
4. `validateCanonicalIndexParams`'s three `allowed` maps move next to `IndexTypes` as one `map[IndexType][]string` — this is also the source for the proto comment and OpenAPI enum.
5. `contractgen` emits `contract.json.index_types` and `index_params`; `check_contract.py` asserts the proto comment.
6. `TestEveryIndexTypeConstructs` ranges `IndexTypes`, builds a one-field schema for each, asserts `createDenseIndex`/`createSparseIndex` succeeds. Eleven lines, and it is the thing that makes the list real.

---

## 5. The agent-memory layer on the engine

The organizing idea that makes this coherent, and the sentence I'd put at the top of `ARCHITECTURE.md`:

> **DeepData has three durability classes. The RC's fail-closed guarantee applies to class A only. Every new capability must declare its class.**

| Class | What | Mechanism | On corruption |
|---|---|---|---|
| **A — canonical** | schemas, documents | `journal.go` + `snapshot.go`, six mutation types, fsync/record, single RWMutex | **fail closed**, latch fault, `/readyz` 503 |
| **B — accreted signal** | usage frecency, feedback boosts | separate append-only file, own lock, batched fsync | **discard and continue** — a lost signal degrades ranking, it loses nothing |
| **C — derived** | HNSW graphs, BM25 stats, graph/PageRank | in-memory, rebuilt from A | **rebuild** |

This taxonomy is not decoration. It is the answer to every "where does X go?" question, and it prevents the specific failure mode that OOM-killed replay twice.

### Text-in / text-out: zero new mutation types, zero journal change

The wrong design is journaling the text and embedding at replay. That makes recovery depend on a model being present, reachable, and *identical* — and it is the same class of error as `segments` from `GOMAXPROCS`. Generalize the lesson already in `tasks/lessons.md`:

> **Never derive persisted state from ambient runtime.** An embedding model is ambient runtime.

The right design: **embed at admission, in the transport/service layer, above `DurableStore`.**

- `CollectionSchema` gains a persisted per-field `embedder: {model_id, dim, normalize}`, carried in the existing `Index.Params`-adjacent slot and admitted by an allowlist exactly like `segments` (`durable_store.go:605-612` is the pattern).
- `POST .../docs` accepts `{"text": {"embedding": "..."}}` as an alternative to `{"vectors": {...}}`. The handler resolves text→vector, then calls the **existing** `insert_document` mutation with a dense vector. The source text is carried in `metadata` (source attribution, and it makes re-embedding on model upgrade possible without re-ingest).
- `DurableStore` never learns text exists. `appendApplyLocked` is untouched. The journal bytes are byte-identical to today's.
- **Fail loud:** if the process's configured `model_id` ≠ the collection's persisted `model_id`, reject the write with a 409 naming both. A collection embedded by two models is silently corrupt in a way no checksum catches.

Six mutation types stay six. That constant is load-bearing and should be commented as such.

### Durable signal: separate side-store, class B

Journaling usage/feedback through `appendApplyLocked` would put a write on the *read* path, through the RWMutex that also spans snapshot and rotate, at query frequency. That is precisely how you rebuild the 4.71 GB / 301,816-doc replay OOM, on purpose. Not negotiable.

New: `internal/collection/signal/` (~250 LoC), one file per tenant under `<base>/signal/<tenant>.log`, length-prefixed CRC records `{collection, doc_id, kind, weight, unix_ms}`, its own `sync.Mutex`, fsync every N records or T seconds. Load at open, best-effort. **On any read error: log, truncate, continue.** `DurableStore` holds a `*signal.Log` and calls it *outside* `mu`. `usage.go`'s in-memory tracker becomes its cache; `feedback`'s boosts become a second `kind`.

Simpler than new mutation types, strictly crash-safe (nothing in class B can fault class A), and it makes the accretion durable — which is the actual thesis.

### Graph: class C, never persisted

`internal/graph` builds CSR + PPR from documents already in memory. Persisting it would create a second authority for the same facts. Build lazily on first graph-weighted search per collection, invalidate on mutation, cap by node count. `fusion.go:261` already accepts it as a third result set.

### What must not change

1. Journal record framing, CRC, store-ID binding, LSN contiguity. Any change needs an ADR and a migration.
2. Snapshot v2 stream format and its "schemas+documents, not index exports" rule.
3. Exactly six mutation types.
4. Fail-closed: apply-after-append failure latches permanently. No feature may add a recoverable-looking path around `latchFaultLocked`.
5. `setDurableReadOnly` / `ErrCanonicalMutationRequired` — all mutation through `TenantManager`.
6. Caller-supplied vectors remain a **first-class, always-available** input. Text-in is additive. An agent that has its own embeddings must never be forced through a server model.

---

## 6. Legibility for a cold agent

### Read order, and it should be printed at the top of the README

```
1. ARCHITECTURE.md            — the tower, the truth table, the three durability classes
2. api/contract.json          — every limit, error, and precondition, machine-readable
3. internal/collection/durable_store.go:17-23   — the six mutation types
4. api/proto/deepdata/v3/deepdata.proto         — the wire shape
5. tasks/lessons.md           — every correction, why, as a rule
```

Five files, under 2,000 lines, and an agent that reads them can answer any structural question. Today the equivalent path is README → `why-vectordb.md` → `API.md` → `PRE_RELEASE_STATUS.md` → `STATE.json` → `todo.md` → `GAP_ANALYSIS.md` → `internal/index/README.md`, four of which contradict the code.

### ARCHITECTURE.md, one screen for the truth table

```markdown
# DeepData architecture

Agent memory over a fail-closed vector store. Text or vectors in, ranked
documents out, signal accretes.

## Truth table
<!-- contract:begin:layers -->
| Layer | Truth file | Derived | Gate |
|---|---|---|---|
| substrate | internal/collection/journal.go, snapshot.go | — | frozen (ADR-0001) |
| engine    | internal/collection/durable_store.go | — | — |
| contract  | internal/collection/types.go, limits.go, contract.go | api/contract.json | check_contract_generated.sh |
| grpc      | api/proto/deepdata/v3/deepdata.proto | api/gen/** | check_proto_generated.sh |
| http      | api/openapi.yaml (generated) | — | check_contract_generated.sh |
| python    | sdk/python/deepdata/models.py | _contract.py | test_contract_parity.py |
| mcp       | cmd/deepdata-mcp/main.go | go:embed contract.json | go test ./cmd/deepdata-mcp |
<!-- contract:end:layers -->

## Durability classes
A canonical · fail closed | B signal · discard and continue | C derived · rebuild

## Package graph
<!-- graph:begin --> (generated mermaid) <!-- graph:end -->
```

Both marked blocks are generated and diff-gated. Prose *between* markers is allowed and hand-written; prose that restates a table is not.

### `doc.go` over package READMEs

Go's native mechanism, `go doc` reads it, `grep -r "^// Package"` enumerates the system in one command. One `doc.go` per surviving `internal/` package, ≤30 lines, stating: what it owns, its durability class, and what it must not do. Delete `internal/index/README.md`. `benchmarks/README.md` and `sdk/python/README.md` stay — different audiences.

### `deepdata-system-map.html` → delete, replace with 20 lines

54 KB of hand-drawn SVG, dated Aug 14, marks DiskANN/IVF/PQ/WebUI live, omits five real packages. It was wrong the day after it was written because nothing regenerated it.

Replacement, rung 3 (stdlib/platform): `scripts/gen_graph.sh` runs `go list -deps ./cmd/deepdata`, filters to `github.com/phenomenon0/`, emits Mermaid between `<!-- graph:begin -->` markers in `ARCHITECTURE.md`. ~20 lines, diff-gated by the same check. It cannot be wrong, because it *is* the import graph. No d3, no HTML, no browser.

### Naming, and two renames worth doing while the tree is small

1. **Drop the `Canonical` prefix.** It means "not legacy." After slice 4 there is no legacy, so it means nothing and every agent must learn it. `CanonicalMaxSearchTopK` → `MaxSearchTopK`, `canonicalGRPCError` → `grpcError`, `authorizeCanonicalHTTP` → `authorizeHTTP`. ~80 identifiers, purely mechanical, and it removes a whole vocabulary. Do it in the same commit as the deletion so the diff is already large and reviewers read it once.
2. **`github.com/phenomenon0/vectordb` → `github.com/phenomenon0/deepdata`.** One product, two names, in every import line of every file. One `go.mod` edit plus a mechanical rewrite.

### Convention rules, stated once in ARCHITECTURE.md

- One truth file per concept. If you need a second, the first is wrong.
- Constants live where they're enforced, and are exported so transports import rather than copy.
- Transports translate; they never validate. Any `if` in a transport that duplicates an engine rule is a bug (0.5).
- Errors are prompts: every 4xx/`InvalidArgument` message names the field, the received value, the bound, and the next action.
- New capability → declare its durability class in its `doc.go` or it doesn't merge.

---

## 7. Sequenced slices

Every slice leaves the tower consistent, compiling, and green. Sized in files/LoC with the evidence each produces.

**Slice 0a — land the in-flight recovery work (do this before touching anything else).**
Commit the +2136/−514 journal/replay diff *with* the two open test items: the generated small-journal fail-closed tests (corrupt record, partial tail, LSN gap, two-pass recovery, crash/reopen, snapshot compat, bounded retained payloads) and the cgroup-capped replay rehearsal. Those are the runnable check for non-trivial logic; the diff is unfinished without them. The third item — bounded-memory unified snapshots — is a *different* code path and does not block; it becomes slice 0b.
Rationale: a 2,650-line uncommitted diff is itself a risk, and every slice below rebases on it. Do not begin the restructure on a dirty tree.
*Evidence:* `internal/collection -race -p 1` green; rehearsal log showing cold start → doc-count parity → dense/sparse/hybrid parity → graceful close → second cold start → peak RSS under cap.

**Slice 0b — bounded-memory unified snapshots.** 1 file, ~200 LoC. *Evidence:* snapshot of a synthetic 300k-doc store under a cgroup cap; v1 readability test retained.

**Slice 1 — error taxonomy. Highest agent-effectiveness per line in the repo.**
Five `%w` verbs at `collection.go:512-523`; `ErrTenantLimitExceeded`/`ErrCollectionLimitExceeded` → 409 / `FailedPrecondition` (0.4); rate-limit 429 gains `Retry-After`; `writeCanonicalOperationError` emits `{"error":{"code","message","field","received","limit","retryable"}}` instead of `http.Error` text; delete the ~60 duplicated pre-validation lines from `collection_grpc.go:279-302` and `collection_http.go:1063-1079`.
**Net −40 LoC and the surface becomes self-teaching.** An agent that gets `top_k must be in [1, 1000]; received 5000` fixes itself; one that gets `500 search failed` files a bug.
*Evidence:* table test asserting every `ErrInvalidSearchArgument` path yields 400/`InvalidArgument` on both transports with a populated `field`.

**Slice 2 — observability one-liners.** 4 one-liners + ~25 LoC health server (see §2). *Evidence:* `curl /metrics | grep deepdata_http_requests_total{endpoint="/v3/tenants/:id/collections/:name/search"}`; `grpcurl -plaintext list`; `grpc_health_probe`.

**Slice 3 — `serverRuntime` extraction (keystone).** 1 new file ~120 LoC, ~40 call-site edits, zero behavior change. *Evidence:* existing `cmd/deepdata` suite passes unchanged.

**Slice 4 — the great deletion.** ~62,000 LoC out; `Canonical` prefix drop; module rename; `go.mod` loses `Neumenon/cowrie`, `Neumenon/shard`, `mattn/go-sqlite3`; ONNX deps move behind `//go:build onnx`. `canonicalRCSurface` deleted, not bypassed. CI's `test-experimental-source` job deleted (nothing left to fail softly). *Evidence:* `go build` with `GOFLAGS=-mod=readonly` from a clean module cache with no private-repo credentials — the first time that has ever worked. `CGO_ENABLED=0` default build.

**Slice 5 — index vocabulary.** ~80 LoC net negative across 4 files + `TestEveryIndexTypeConstructs`. *Evidence:* a v2 journal with an IVF field fails loud with the migration message.

**Slice 6 — the contract.** `cmd/contractgen` ~180 LoC, `internal/collection/contract.go` ~90, three generated artifacts, three CI steps, `TestContractRulesAreEnforced`. Also fixes `doc_count` → `document_count` and drops the stale title-case aliases at `models.py:131-142`. *Evidence:* deliberately bump `MaxSearchTopK` in a scratch commit; all three checks fail.

**Slice 7 — MCP rewrite.** `cmd/deepdata-mcp` 449 → ~600 LoC. `go:embed api/contract.json`; typed input schemas and `outputSchema`/`structuredContent`; `annotations` (`readOnlyHint` on search/get/list, `destructiveHint` on delete); `resource_link` per result document so an agent can fetch without re-searching; Glyph text in `content[0].text` beside the structured block; errors carry the slice-1 JSON verbatim; pagination + steering on list; multi-tenant via tool arg with an env default. Added to CI. *Evidence:* `go test ./cmd/deepdata-mcp` driving initialize/tools-list/tools-call over stdio, asserting every `outputSchema` validates its own response.

**Slice 8 — legibility.** `ARCHITECTURE.md`, one `doc.go` per surviving package, `scripts/gen_graph.sh`, `docs/adr/0001-journal-format-frozen.md` + `0002-three-durability-classes.md`, delete the six rotted documents and `deepdata-system-map.html`, README read-order block. *Evidence:* `check_contract.py` green over README + ARCHITECTURE marked blocks.

**Slice 9 — text-in.** Schema `embedder` block + admission-time embed + Ollama/ONNX providers retained, cloud providers deleted. ~400 LoC net. **This is where the platform stops being theoretical for an agent** — before it, an agent that can't produce vectors cannot use DeepData at all, which makes the entire MCP surface decorative. *Evidence:* model-mismatch 409 test; journal-bytes-identical test proving no format change; recovery test proving replay needs no model.

**Slice 10 — durable signal (class B).** `internal/collection/signal/` ~250 LoC; `usage.go` becomes its cache; `feedback/store.go` reduced to a second record kind, its JSON WAL deleted. *Evidence:* corrupt-signal-file test proving the store still opens and serves.

**Slice 11 — Recommend/Discover routed, then graph behind the schema flag.** *Evidence:* recall harness in `benchmarks/` showing graph weighting changes ranking measurably, or the flag doesn't ship.

Slices 1–8 are the tower. 9–11 are the ambition, and they land on a tower that cannot drift under them.

---

### Critical Files for Implementation

- `/home/omen/Documents/Project/DeepData/internal/collection/collection.go` (root of truth for search preconditions; the five missing `%w` at 512-523; `createDenseIndex` at 134-215)
- `/home/omen/Documents/Project/DeepData/internal/collection/durable_store.go` (six mutation types at 17-23; `appendApplyLocked` at 490; `validateCanonicalSchema` at 589-612 — the schema-param allowlist pattern the `embedder` block copies)
- `/home/omen/Documents/Project/DeepData/cmd/deepdata/main.go` (`VectorStore` at 53-113 — the seven auth fields to extract; `canonicalOnly` at 3198; `NewVectorStore(0,1)` at 3306; gRPC construction at 3421)
- `/home/omen/Documents/Project/DeepData/cmd/deepdata/server.go` (50 legacy `mux.Handle`; middleware chain at 3379; `canonicalRCSurface` at 3414-3423)
- `/home/omen/Documents/Project/DeepData/cmd/deepdata/collection_http.go` (V3 router at 389-514; `writeCanonicalOperationError` at 638-656; duplicated search validation at 1063-1079)
- `/home/omen/Documents/Project/DeepData/scripts/check_version_contract.py` and `/home/omen/Documents/Project/DeepData/scripts/check_proto_generated.sh` (the two enforcement shapes every new gate copies)


---

# Part 2 — architect C — plan/docs legibility + workflow (Claude Fable)

# DeepData plan/docs system — design and rewrite spec

Verified against the tree at `6adde4c` (dirty, +2136/-514). Everything cited below was read; line numbers are current.

## Pushback on the brief (things I found that differ)

1. **The `/metrics` false claim exists in two places, not one.** `docs/grafana/README.md:20` and `internal/collection/API.md:63-64` ("unauthenticated operational routes are … `GET /metrics`"). Truth: `cmd/deepdata/server.go:1503-1507` wraps `/metrics` in `guard(...)`.
2. **API.md has more drift than D4/D5.** `API.md:47-57` route table lacks `PUT/GET …/docs/{id}` (router: `collection_http.go:85-107`, `:888`); `:131-132` "V3 has no upsert"; `:243-244` lists upsert and document fetch as outside the RC; and the whole Search section `:169-204` omits `score_floor`/`fallback`/`usage_boost` — the agent retrieval contract that README documents is absent from the API reference.
3. **README `:21-22`** omits upsert (mutations row) and get-document (reads row), in addition to `:23`/`:194-199`.
4. **CHANGELOG `:4-5` and `:60`** still say no license; its Added list `:11-13` predates upsert/get-doc/agent-retrieval/MCP/V2-removal.
5. **`tree_fingerprint` for a clean tree is a constant**: `e3b0c442…b855` = sha256 of empty input (see the real receipt `.deepdata-run/checks/go-short/receipt.json`). "Evidence came from a clean tree" is therefore a string equality check — I use that in the ledger.
6. **Playwright `:155` isn't false, it's orphaned**: `tests/ui/` has a Playwright config; the CI job it referred to was removed when the UI stopped gating. The gate should be *retired*, not un-checked.
7. **All 10 local receipts are bound to `a3de516` (2026-08-07)**, 16 commits behind HEAD — the receipts are as stale as STATE.json.
8. **`benchmarks/review/product_manager_test.go:135-144` hard-codes `benchmarks/GAP_ANALYSIS.md`** and `:124` `benchmarks/README.md`; `benchmarks/mega_bench.py:1216` and `docs/releasing.md:53` emit/cite `docs/PRE_RELEASE_STATUS.md`. Renames must patch these or keep the paths. I keep `docs/PRE_RELEASE_STATUS.md` as the path (generated), and move GAP_ANALYSIS with a one-line test patch.
9. **`main` is stale**: `main` = `6684226` (merge of PR #3), merge-base `0788cfa`; PR #4 is open on the `gnhf/…` branch titled "do not merge yet". Branch rename affects neither.
10. **`docs/models/…/tokenizer.json` is read by nothing.** Code expects `vectordb/models/bge-small-en-v1.5/` (`embedder_mode.go:261-267`) or `$dataDir/models/…`; `fetch_bge_small.sh:4` writes to `vectordb/models/`. Three paths, none `docs/`. Delete from git; the fetch script exists for a reason.
11. I reject the brief's `tasks/evidence/` receipt copies — a second store is exactly the disease.

---

## 1. One truth ledger: `tasks/gates.json` + `scripts/gates.py` → generated `docs/PRE_RELEASE_STATUS.md`

**Format: JSON.** Reasons: receipts are already JSON (`hardening_check.sh:262-274`) so promoting evidence is a field copy with no parser; Python stdlib parses it; a Markdown table would need a bespoke parser and puts globs/commands in table cells. YAML rejected (no stdlib parser). The ledger is *not* the status page — agents and humans read the render; only the render is one-screen. Same path as today so six inbound links (README:259, releasing.md:53, BENCHMARKS.md:75, mega_bench.py:1216, mega REPORT, todo.md) keep working.

**Status enum (stored):** `open | pass | blocked | deferred | retired`. `stale` is never stored; it is *computed* at render/check time: a `pass` gate whose `scope` pathspecs have any change between `evidence.commit` and the working tree (`git diff --name-only <commit> -- <scope>`) renders as `pass (stale)` and counts as `open` for release. This kills the "carried forward from d5b2d3a on a premise no longer true" class (PRE_RELEASE_STATUS:118-126) mechanically.

**Schema (`schema_version: 1`):**
```json
{
  "schema_version": 1,
  "gates": [
    {
      "id": "EVID-01",
      "statement": "Candidate SHA frozen: clean worktree, go.sum and tool manifests regenerated at that SHA, every release gate's evidence.commit equals it",
      "command": "test -z \"$(git status --porcelain)\" && go mod verify && python3 scripts/gates.py check --release",
      "scope": ["go.mod", "go.sum", "cmd", "internal", "api", "sdk/python", "deploy", "Dockerfile"],
      "release_gate": true,
      "status": "open",
      "evidence": null,
      "note": "was tasks/todo.md Phase 8 :211"
    },
    {
      "id": "DUR-02",
      "statement": "go test -short ./... passes (covers auth allow/deny matrix, canonical surface 404 wall, crash/replay subprocess tests)",
      "command": "scripts/hardening_check.sh go-short",
      "scope": ["**/*.go", "go.mod", "go.sum"],
      "release_gate": true,
      "status": "pass",
      "evidence": {
        "commit": "a3de51621d35f2cde72b4c0025f4d5cc3d805fce",
        "tree_fingerprint": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "date": "2026-08-07T22:12:06-05:00",
        "receipt": ".deepdata-run/checks/go-short/receipt.json",
        "ci_run": null
      }
    }
  ]
}
```
`evidence.tree_fingerprint` must equal the empty-diff constant for `release_gate: true` gates (check enforces). `receipt` is a local, uncommitted path — allowed *only* inside gates.json evidence (the docs linter bans `.deepdata-run/` paths elsewhere). `ci_run` for gates proven remotely (CI-01).

**`scripts/gates.py` (stdlib: json, subprocess, sys, argparse) — three subcommands:**
- `check [--release]`: schema valid; unique IDs; every `pass` has evidence with 40-hex commit that `git cat-file -e` resolves; release gates' fingerprint == clean constant; committed `docs/PRE_RELEASE_STATUS.md` byte-equals a fresh render (same pattern as `check_proto_generated.sh`); **no `[ ]`/`[x]` anywhere under `tasks/`** (checkboxes are banned; status lives here). `--release` additionally fails if any release gate is not a non-stale `pass` or `retired`.
- `render`: writes `docs/PRE_RELEASE_STATUS.md` (below).
- `promote <check-name> <gate-id>`: copies `git_commit/tree_fingerprint/finished_at/receipt-path` from `.deepdata-run/checks/<check>/receipt.json` into the gate, sets `pass` only if receipt `status == passed`, then renders. Manual evidence (kind rehearsal, backup drill) is edited by hand with the same four fields.

**Render (`docs/PRE_RELEASE_STATUS.md`, ≤ 80 lines for ~35 gates):**
```
# Release status — generated from tasks/gates.json; do not edit
Rendered at <sha7>. Regenerate: python3 scripts/gates.py render. CI fails when stale.
Release gates: N pass · N pass(stale) · N open · N blocked · N deferred · N retired
| ID | Gate | Status | Evidence |
|---|---|---|---|
| DUR-02 | go test -short ./... passes | pass (stale: scope changed since) | a3de516 · 2026-08-07 · clean tree |
| EVID-01 | Candidate SHA frozen … | open | — |
## Blocked / deferred / retired (reason column from `note`)
Plan and ordering: tasks/todo.md · Why: docs/decisions/ · History: tasks/journal/
```

**STATE.json: DELETE.** Its content splits: `scope` → ARCHITECTURE.md non-goals block; evidence arrays → gate evidence + `tasks/journal/2026-07-22-exact-sha-evidence.md` and `…-08-07-hardening-upsert.md`; `authority` → PROTOCOL.md (static); `next_action` → todo.md line 2. `hardening_check.sh:51-54,100-102` `state-json` check → `gates-check` running `python3 scripts/gates.py check`.

**`.deepdata-run/`: unchanged, stays gitignored.** Only the four evidence fields are committed, inside gates.json. No receipts index, no copies.

**Seed gate list (IDs the plan and roadmap reference):**
DUR-01..05 (go-storage, go-short, go-race, go-vet-cgo0, crash matrices — all `pass` at `a3de516`/`ece6f23`, will render stale) · SDK-01/02 · PKG-01 container+compose (ece6f23), PKG-02 kind/Helm (d5b2d3a → stale since `2e82859` touched `deploy/`) · OPS-01 backup drill (`bc1fa27`; closes todo:142), OPS-02 legacy export/import (d5b2d3a Stage B; closes todo:109), OPS-03 disk-full/perm-denied · SOAK-01, SOAK-02 memory-drift (14d1442) · SEC-01 scans (ece6f23) · REL-01 version contract, REL-02 license (`b1f28be`; closes todo:198), REL-03 dry-run release workflow, REL-04 `go mod tidy` clean (`442cdb6`; closes todo:212) · CI-01 remote CI green on HEAD (f5d4582, run 29977008196 → stale), CI-02 scale job (deferred), CI-03 live k8s cadence (deferred), CI-04 Playwright/UI (retired: UI not an RC gate; `tests/ui` orphaned), CI-05 `./cmd/deepdata-mcp` in CI package list (open), CI-06 `tests/smoke_test.sh` wired or retired (open) · EVID-01..03 · REV-01 adversarial re-audit (open); REV-02/03/04 (todo:230-232) retired — subsumed by `gates.py check`, the fingerprint rule, and the generated render · RCV-01..06 (section 6) · DOC-01..03 (section 4) · PUB-01..03 (blocked: PyPI name, tag, signing) · reserved prefixes `CTL-`, `SYS-`, `MEM-` for architects A/B — they add gates, not checkboxes.

---

## 2. Split plan from journal

**New `tasks/` layout** (directory `tasks/autonomy/` disappears):
```
tasks/
  todo.md                 live plan: next action + ordered slices as gate IDs; no checkboxes
  gates.json              truth ledger (section 1)
  lessons.md              unchanged, append-only
  PROTOCOL.md             moved from autonomy/, revised
  journal/
    2026-07-17-baseline.md            ← git mv tasks/autonomy/BASELINE.md (+2-line header: toolchain now 1.25.12)
    2026-07-22-exact-sha-evidence.md  ← STATE.json arrays :37-46,:86-125 + PRE_RELEASE :194-230 (memory-drift narrative)
    2026-08-07-hardening-upsert.md    ← PRE_RELEASE :136-192 + STATE :47-66
    2026-08-17-agent-retrieval.md     ← todo.md :297-355
    2026-08-22-hardening-ingest.md    ← todo.md :357-390
    2026-08-28-bounded-recovery.md    ← todo.md :392-470
    2026-09-01-redesign-architects.md ← the three architect outputs verbatim (input to the Workflow)
```
Per-file journal, not one JOURNAL.md: a new file per session is append-only by construction; old files are never edited (a session that corrects the past writes a new entry and a lessons.md rule). Checkboxes banned in `tasks/` (enforced by `gates.py check`), so a journal can never become a fourth status store.

**`tasks/todo.md` template (≤ 90 lines):**
```
# DeepData plan
Next action: <one imperative line naming a gate ID>
Status lives in tasks/gates.json → rendered at docs/PRE_RELEASE_STATUS.md. This file holds ordering only; it contains no checkboxes.
Scope and non-goals: docs/ARCHITECTURE.md · Why: docs/decisions/ · History: tasks/journal/ · Rules: tasks/PROTOCOL.md

## Objective  (≤ 5 lines: self-hosted agent memory platform on the hardened single-node core; RC first, then re-expansion per ADR 0006)

## Slices (ordered; each line = gate IDs · intent · depends-on)
1. RCV-03 RCV-04 RCV-05 RCV-06 · finish bounded recovery · —
2. DOC-01 DOC-02 · plan/docs system (this rewrite) · —
3. CTL-* · control surface as one contract (architect A) · DOC-03 route table
4. SYS-* MEM-* · systems tower, revive/retire, agent-memory layer (architect B) · 3
5. EVID-01 EVID-02 EVID-03 CI-01 REV-01 · exact-SHA evidence at the frozen candidate · 1-4
6. PUB-01..03 · publication (authority-gated) · 5

## Critical path
1 → 5 for the RC tag; 3 → 4 for the platform. 2 runs now and gates everything after it.

## Parked
Deferred/retired gates and reasons: gates.json status=deferred|retired (CI-02, CI-03, CI-04, REV-02..04).
```
Decision Rules / Escalation Triggers / Retry budget (todo.md:244-282) move to PROTOCOL.md — they are rules, not plan.

**Journal entry template:**
```
# YYYY-MM-DD — <slug>
Commits: <range or "uncommitted, see WIP commit …"> · Gates touched: RCV-01 pass, RCV-03 open
## What happened   (narrative; local paths and RSS numbers welcome here)
## Evidence        (commands, results, receipt paths — only place besides gates.json they may appear)
## Decisions       → docs/decisions/NNNN   (or "none")
## Lessons         → tasks/lessons.md      (or "none")
```

**Decision log `docs/decisions/NNNN-<slug>.md` (≤ 20 lines each):**
```
# 0002 — No online compactor in the RC
Date: 2026-07-22 · Status: accepted · Supersedes: — · Evidence: 14d1442
## Context      (≤ 3 lines)
## Decision     (≤ 3 lines)
## Consequences (≤ 4 lines; what becomes true/false, which gate proves it)
```
Backfill: `0001-narrow-rc-to-single-node-caller-vectors` (2026-07-17; todo.md:22-49; context cites the `.gnhf` run prompt as the origin of the branch) · `0002-no-online-compactor` (PRE_RELEASE:226-230) · `0003-upsert-and-get-document-join-canonical-surface` (2026-08-07, a99fe53; supersedes the exclusion at old todo.md:33) · `0004-floor-before-fallback` (bde4f94; lessons.md:87-96) · `0005-ingest-levers-segments-then-allocs-then-group-commit` (6adde4c; todo.md:381-390; supersedes BENCHMARKS.md:42 wording) · `0006-re-expand-to-agent-memory-platform` (2026-09-01; **status: proposed** until the user accepts; body = one-paragraph tower thesis + pointers to GAP_ANALYSIS sections 4-6) · `0007-persisted-segment-count-never-derived-from-gomaxprocs` (lessons.md:108-112; lands with the WIP commit).

**PROTOCOL.md → `tasks/PROTOCOL.md`, revised sources-of-truth (`:18-29` replaced):**
1. The user's approved objective and scope (`docs/ARCHITECTURE.md` non-goals + `docs/decisions/`).
2. Passing tests on the current tree.
3. Git graph and working-tree diff.
4. `tasks/gates.json` — evidence-bound status. Its render `docs/PRE_RELEASE_STATUS.md` is derived; never edit it.
5. `tasks/todo.md` — ordering and next action only.
6. `tasks/journal/`, `tasks/lessons.md`, local `.deepdata-run/` — narrative and logs; never authoritative.
Rules kept: `:29` "a status value never overrides failing evidence"; `:43-44` dirty tree = unverified. Added: "a `pass` whose scope changed since `evidence.commit` is stale and counts as open"; "checkboxes are banned under tasks/". Remove all STATE.json references (`:26, :50, :53-54, :57, :90-97`); `last_validation.command` becomes "the `command` of the gate named in Next action". Cold-start read order (new section, replaces `:47-59`): `tasks/todo.md` first screen → `docs/PRE_RELEASE_STATUS.md` → `git status` → newest `tasks/journal/` → `tasks/lessons.md`. Fold in Decision Rules/Escalation/Retry budget from todo.md. Authority block (from STATE.json:131-136): push of the working branch authorized; publication/credentials not.

---

## 3. `docs/GAP_ANALYSIS.md` (new) and `docs/ARCHITECTURE.md` (new, separate)

Two files because they have different lifetimes: GAP_ANALYSIS is a **dated analysis at a commit** whose roadmap is gate IDs (so it cannot claim status); ARCHITECTURE is the **maintained one-screen map** and is the machine-read source of the non-goals list. Convention stated in both: uppercase filenames in `docs/` are status/evidence/analysis documents; lowercase are guides.

**`docs/GAP_ANALYSIS.md` outline:**
```
# DeepData gap and upgrade analysis — at <sha7>, 2026-09-01
0. How to read (≤ 6 lines): what is true now (§1-2), what we will do (§4-7), how it stays true (§8). Status is never written here; gate IDs point at tasks/gates.json. Competitor numbers: docs/BENCHMARKS.md only.
1. State of the system at HEAD — table: tree · ~LoC · reached from the RC binary? · CI job (rc/experimental/none) · verdict live/dormant/dead · evidence (file:line)
   (rows: cmd/deepdata 16.9k live · internal/index 17.7k mixed · internal/collection 8.5k live · internal/cluster 7.4k dormant · internal/security 5.1k live · internal/extraction 2.1k dormant · internal/feedback 1.6k dormant · cmd/cli 1.0k dead · client 0.7k dead · internal/obsidian 0.7k dormant · internal/graph 0.6k dormant · cmd/deepdata-mcp 0.4k live-not-in-CI · web-ui/desktop … — LoC rounded, stamped "at <sha7>", not linter-checked)
2. Drift inventory — table: # · claim · where (file:line) · truth (file:line) · fix (disposition) · rule that now prevents it (R1..R9 or "one-time")
   (rows D4-D12 + the four extras from the pushback list above)
3. Process drift — why four status stores rotted (one paragraph) and the replacement (section 1 of this plan, summarized in 6 lines)
4. Sacrificed-ambition inventory (from architect B) — table: tree · verdict revive/retire/keep-dormant · why · gate ID
5. Agent-ergonomics gaps (from architect A) — ranked table: gap · current behavior (file:line) · target · gate ID
6. Redesign thesis — the tower (≤ 15-line diagram) + ≤ 5 principles; link ARCHITECTURE.md and decisions/0006
7. Roadmap — slices as gate IDs only (mirrors todo.md slices; no status)
8. How this document is kept true — names scripts/check_docs_contract.py (rules applied to this file: R5 links, R7 file refs, R8 non-goals), scripts/gates.py check, and the rule "any status word here is a bug"
```
`benchmarks/GAP_ANALYSIS.md` is deleted in the same commit; `product_manager_test.go:137` path → `docs/GAP_ANALYSIS.md`; `benchmarks/README.md` link updated.

**`docs/ARCHITECTURE.md` (one screen, ≤ 70 lines):**
```
# Architecture — the tower
<diagram: MCP / HTTP V3 / gRPC v3 / Python SDK  →  one contract (proto + route table + types.go)  →  engine (internal/collection)  →  durability (journal.go, snapshot.go)  →  one locked Linux data dir>
| Layer | Truth file | Proven by | Documented in |
| Contract | api/proto/deepdata/v3/deepdata.proto; cmd/deepdata/collection_http.go dispatch (→ declarative route table when SYS-xx lands) | canonical_surface_test.go, collection_grpc_canonical_test.go | internal/collection/API.md |
| Agent surface | cmd/deepdata-mcp/main.go | cmd/deepdata-mcp tests (CI-05) | docs/mcp.md |
| Engine | internal/collection/collection.go, types.go | go-short (DUR-02) | API.md, cookbook.md |
| Durability | journal.go, snapshot.go, durable_store.go | DUR-01, DUR-05, RCV-* | API.md "Durability behavior" |
| Auth | internal/security | DUR-02 auth matrix | docs/security.md |
| Packaging | Dockerfile, deploy/helm | PKG-01, PKG-02 | installation.md, kubernetes.md |
## Non-goals (machine-read by scripts/check_docs_contract.py — one per line: label | regex)
<!-- non-goals -->
- DiskANN | \bDiskANN\b
- IVF | \bIVF\b
- product/binary quantization | \b(PQ|product quantization|binary quantization)\b
- replication/clustering/failover | \b(replication|failover|cluster(ing)?|shard(s|ing)?)\b
- GraphRAG | \bGraphRAG\b
- desktop | \b(Tauri|desktop (app|installer|wrapper))\b
- web UI as supported surface | \bweb (UI|dashboard)\b
- CUDA | \bCUDA\b
- built-in TLS/encryption at rest | \b(built-in TLS|encryption[- ]at[- ]rest)\b
<!-- /non-goals -->
```
README's "does not include" paragraph (`:34-38`) becomes a generated rendering of this block, so scope has one source. When architect A revives the embedder, the line for "server-managed embeddings" is removed here and the linter follows.

---

## 4. Machine-checked docs: `scripts/check_docs_contract.py`

Stdlib only (`re, pathlib, subprocess, json, sys`). Modes: default = check (fail with a diff/list); `--write` = rewrite generated blocks in place; `--facts` = print the extracted facts as JSON (used by the Workflow's facts agent so authors never re-derive counts). Wired into `ci.yml` `test-rc-go` directly after `check_version_contract.py` (`:44-45`). Checked set: `README.md`, `CHANGELOG.md`, `docs/**/*.md`, `internal/collection/API.md`, `sdk/python/README.md`, `benchmarks/README.md`, `tasks/todo.md`, `tasks/PROTOCOL.md`; excluded: `tasks/journal/`, `docs/decisions/`, `benchmarks/results/**`, `benchmarks/competitive/live/**` (historical, may legitimately describe dead things).

| Rule | Source of truth | Check |
|---|---|---|
| R1 gRPC list | `service DeepData {…}` rpc names in the proto (11) | generated block `<!-- generated:grpc-rpcs -->…<!-- /generated -->` in README.md and API.md must equal the regenerated bullet list; plus prose regex: any `(\b(nine|ten|eleven|twelve|\d+)\b)[^.\n]{0,40}\b(unary )?(gRPC|RPC)s?\b` (and reverse order) must equal 11 in words or digits |
| R2 mutation count | proto RPCs not matching `^(Get|List|Search)` → 6 | prose regex `\b(five|six|seven)\b[^.\n]{0,30}\bmutations?\b` must say six |
| R3 HTTP route table | declarative route table from architect B, exposed as `go run ./cmd/deepdata routes` (TSV method/path/permission) | generated block `<!-- generated:http-routes -->` in API.md. Until the subcommand exists the rule prints `SKIP R3: routes source absent` **and exits non-zero unless `DOC_ALLOW_R3_SKIP=1`** is set in CI — the skip is loud and gate DOC-03 stays open |
| R4 known-false phrases | regression list inline in the script: `(?i)/metrics[^.\n]{0,40}unauthenticated`, `(?i)unauthenticated[^.\n]{0,60}/metrics`, `does not yet have a (legal )?license`, `license (text )?must be resolved`, `(same|exactly) nine`, `V3 has no upsert`, `title-case`, `Playwright`, `group[- ]commit[^.\n]{0,40}(top|first|primary) lever` | any match in the checked set fails; a phrase enters the list when its drift row is fixed |
| R5 dead links | filesystem | every relative `[..](path[#a])` resolves (anchor stripped; http(s) skipped) |
| R6 case collisions | `git ls-files` | two tracked paths equal case-insensitively → fail (`docs/benchmarks.md` vs `docs/BENCHMARKS.md` today) |
| R7 nonexistent file refs | filesystem | backticked `[\w./-]+\.(go|py|sh|md|json|ya?ml|proto|ts|txt)` must exist relative to repo root or the doc's dir; `<repo>/…` placeholders and `.deepdata-run/` are **not** exempt — receipts belong in gates.json, not prose |
| R8 non-goals claimed live | ARCHITECTURE.md `<!-- non-goals -->` block | a line matching a non-goal regex must also match `\b(not|no|outside|unsupported|non-goal|experimental|excluded|rejected|dormant|retired|cannot|never|without|informational|historical)\b`; escape hatch: trailing `<!-- non-goal-ok -->`; `docs/security-patterns.md` excluded by name |
| R9 dashboard metric names | `deepdata_\w+` in `cmd/deepdata/metrics.go` | every `deepdata_\w+` in `docs/grafana/vectordb-dashboard.json` must exist in code |
| R10 MCP tool list | `Name:\s+"(\w+)"` in `cmd/deepdata-mcp/main.go` (or `--print-tools` once A ships it) | generated block `<!-- generated:mcp-tools -->` in docs/mcp.md |
| R11 non-goals rendering | ARCHITECTURE.md block | generated block `<!-- generated:non-goals -->` in README.md |

**Deliberately not checked:** benchmark numbers (only reproducible by rerun; BENCHMARKS.md's "Reproduce" section is the guard), anchors, external URLs, prose tone, JSON examples vs schema (architect A's self-description endpoint should own that), SDK docstrings vs code (mypy strict already gates the code). R8 is the only fuzzy rule and is scoped to product docs for that reason.

Smoke test built in: run on the pre-rewrite tree, the script must report D4 (R1), D5 (R2), D6 (R4), D12 (R6), benchmarks/README phantom files (R7), GAP_ANALYSIS IVF "Yes" (R8 — before it's deleted), dashboard replication panels (R9). The Workflow asserts this (section 7).

---

## 5. Per-file disposition

| File | Disposition | Detail |
|---|---|---|
| `README.md` | REWRITE | New order below. `:21-22` add upsert / get document; `:23` and `:194-199` → generated R1 block; `:34-38` → generated R11 block; `:167-190` MCP section → 6-line pointer to `docs/mcp.md`; `:258-259` → point at ARCHITECTURE, PRE_RELEASE_STATUS (generated), GAP_ANALYSIS, todo.md |
| `internal/collection/API.md` | PATCH | `:11-12` six mutations incl. upsert; `:47-57` add `PUT …/docs/{id} write Upsert by ID` and `GET …/docs/{id} read Get document` (→ generated R3 later); `:63-64` `/metrics` auth-gated when `REQUIRE_AUTH`, cite server.go:1503-1507; `:131-132` replace "no upsert" with PUT description + new "Upsert" and "Get document" subsections (copy shape from cookbook.md:69-75,:143); `:169-204` add `score_floor`/`fallback`/`usage_boost` and response fields `weak_match`/`best_score`/`fell_back_to`; `:210-220` → generated R1 block; `:243-244` drop upsert and document fetch |
| `docs/why-vectordb.md` | PATCH | `:11` drop "exactly nine" (say "an equivalent unary gRPC service"); `:16-17` six mutations incl. upsert; `:43` remove "Upsert/update, document fetch/scan"; add one line pointing at GAP_ANALYSIS for direction; keep filename (module is `vectordb`) |
| `docs/installation.md` | PATCH | `:9-10`, `:198` drop "nine"; commit the uncommitted `DEEPDATA_BIND_HOST` row with the WIP |
| `CHANGELOG.md` | PATCH | `:4-5` license now Apache-2.0; `:60` delete; Added `:11-13` extend with upsert/get-document (a99fe53), agent retrieval fields + MCP server (bde4f94); Changed: legacy V2 HTTP surface removed (9d1672b), snake_case `CollectionInfo` (bc1fa27), journal sync per writer (6ce44e5), bounded-memory recovery (WIP sha) |
| `docs/PRE_RELEASE_STATUS.md` | GENERATE | content replaced by `gates.py render`; prose → journal files (section 2); "no online compactor" → ADR 0002 |
| `tasks/todo.md` | REWRITE | template in section 2; `:297-470` → three journal files; `:244-282` → PROTOCOL.md; `:109/:142/:198` become OPS-02/OPS-01/REL-02 `pass` with commits; `:155` → CI-04 retired |
| `tasks/lessons.md` | KEEP | PATCH: blank line before `:106`; optionally split `:106-127` into five Correction→Rule headings (user's fresh text — ask before restyling) |
| `tasks/autonomy/PROTOCOL.md` | REWRITE + `git mv` → `tasks/PROTOCOL.md` | section 2 |
| `tasks/autonomy/STATE.json` | DELETE | after `gates.py` exists (same commit); `hardening_check.sh` `state-json` → `gates-check` |
| `tasks/autonomy/BASELINE.md` | ARCHIVE | `git mv` → `tasks/journal/2026-07-17-baseline.md` |
| `benchmarks/GAP_ANALYSIS.md` | DELETE | superseded by `docs/GAP_ANALYSIS.md` + `docs/BENCHMARKS.md`; patch `product_manager_test.go:137` and `benchmarks/README.md:49` in the same commit |
| `benchmarks/README.md` | REWRITE | tree section from real `ls` (ddload/, ddload-qdrant/, recall_test.py, comprehensive_bench.py, mega_bench.py, download_datasets.py, review/, competitive/, vectordbbench/, results/); drop `testdata/vectors.go`, `run_comparison.py`, `TestRecall_IVF/DiskANN` unless present; evidence pointer → docs/BENCHMARKS.md |
| `docs/benchmarks.md` | DELETE | unsourced host (Ryzen 9 5900X — matches nothing else), quantization table of non-goals, zero inbound links, case-collides |
| `docs/BENCHMARKS.md` | KEEP | PATCH `:42` "(`tasks/todo.md`: group-commit, parallel index construction)" → "(docs/decisions/0005: parallel segment builds first; group commit deferred)"; keep uppercase per convention |
| `docs/security-patterns.md` | KEEP | PATCH: 3-line header "March 2026 audit; examples citing desktop/replication code are not in the RC; rules remain valid"; excluded from R8 |
| `docs/grafana/README.md` | PATCH | `:20-21` auth-gated; delete `:24-36` once the panels are gone |
| `docs/grafana/vectordb-dashboard.json` | PATCH | remove row "Shard Health (Distributed Mode)" and its 4 panels; R9 guards the rest |
| `docs/models/bge-small-en-v1.5/tokenizer.json` | DELETE | read by nothing (code expects `vectordb/models/` or `$dataDir/models/`); `scripts/fetch_bge_small.sh` fetches it; open a `CTL-` gate for A to make `embedder_mode.go:261-267` and `fetch_bge_small.sh:4` agree on one path |
| `internal/index/README.md` | DELETE | 391-line Dec-2025 "VectorDB Enhancement Plan"; no inbound links; B may add a ≤ 20-line status README in his slice |
| `deepdata-system-map.html` | DELETE | unreferenced, wrong |
| `.gnhf/` | ARCHIVE (out of tree) | `mv .gnhf /home/omen/var/deepdata-archive/gnhf-i-want-you-to-mnake-26a28a && chmod -R a-w …`; remove `.git/info/exclude:7`; cite the path in ADR 0001 context |
| `.agents/`, `.codex/` | DELETE | empty, untracked |
| `.claude/settings.local.json` | PATCH | drop one-off entries (`PenTablet`, `kill -9 3181020`, literal grep/bench commands); keep `go build/vet/test`, `git add/checkout/commit`, `curl`, `ss` |
| `vdb-test-suite/` | KEEP (out of scope for docs) | no docs of its own (only `.pytest_cache/README.md`); its status is a row in GAP_ANALYSIS §1 for B to revive/retire |
| `benchmarks/competitive/live/results/*`, `benchmarks/results/2026-03-11_*` | KEEP | historical; excluded from R8; GAP_ANALYSIS §1 notes they predate the canonical surface and are not evidence (BENCHMARKS.md:58-61 already explains the client-overhead delta) |
| `benchmarks/results/mega/REPORT.md` | KEEP | correctly scoped; link path unchanged |
| `sdk/python/README.md` | PATCH | `:66-67` add "(the server may offer text-in search once CTL-xx lands; see docs/mcp.md)" only when A's gate closes — until then KEEP; `models.py:131-133` comment: code, not docs — hand to A |
| MCP docs | NEW `docs/mcp.md` | install, Claude Desktop config (from README:174-187), generated R10 tool list, env vars, error shape (A's structured errors), resources/outputSchema when A ships them |
| `docs/cookbook.md`, `distributed-architecture.md`, `kubernetes.md`, `security.md`, `SECURITY.md`, `releasing.md`, `troubleshooting.md`, `upgrade-to-0.2-rc.md`, `migration-from-*.md`, `contributing.md` | KEEP | must pass R5/R7/R8 unchanged; the Workflow's KEEP-set assertion fails if they are touched |

**README section order for an agent-first reader (first 40 lines):**
```
1  # DeepData — self-hosted vector search for agents (v0.2.0-rc.1, Linux single node)
2  one sentence; status: docs/PRE_RELEASE_STATUS.md (generated) · map: docs/ARCHITECTURE.md
4  ## Connect an agent (MCP)            ← 12 lines: build, Claude Desktop JSON, env vars, generated tool list
17 ## Contract in one screen           ← generated RPC list (11), route summary (base /v3/tenants/{t}/collections…), auth header, error shape, readiness /readyz
30 ## What it does not do              ← generated non-goals
36 ## Run from source                  ← current :40-68
   ## Canonical HTTP example · ## Agent retrieval fields · ## Python client · ## Persistence and operations · ## Where truth lives (5 pointers) · ## Development · ## License
```

---

## 6. Reconciling the in-flight work

**Gates:** RCV-01 bounded-memory streaming replay (`pass` at the WIP commit if `go test -count=1 -p 1 ./internal/collection ./cmd/deepdata` passes; tests: `durable_mutation_encoding_test.go`, `journal_test.go` deltas) · RCV-02 coverage verification streams headers only (`pass`, same) · RCV-03 bounded-memory unified snapshots (`open`; `snapshot_stream_test.go` is the partial evidence) · RCV-04 generated small-journal fail-closed tests (`open`; command `go test -count=1 -p 1 -run 'Corrupt|Tail|LSN|TwoPass|Reopen' ./internal/collection`) · RCV-05 cgroup-capped rehearsal on the disposable root `/home/omen/var/deepdata-recovery-probe-iBoWBE` (`open`; command names `systemd-run -p MemoryMax=…`; pass criteria: 301,816 docs, dense/sparse/hybrid parity, two cold starts, peak RSS < budget — from todo.md:432-435) · RCV-06 segmented HNSW measured under the same cold-start envelope (`open`; todo.md:425-427 says "review separately" is done but the measurement is not).

**Commit now, as WIP — after running the gates, not before.** PROTOCOL:44 says a dirty tree is unverified; the answer to "unverified" is to verify and checkpoint, not to keep it dirty. The diff is 2,600 lines in the two most dangerous files in the repo (journal.go, snapshot.go), has been uncommitted 4 days, and includes `tasks/todo.md`/`tasks/lessons.md` — the very files the docs rewrite must rewrite. Sequence: `go build ./... && go vet ./... && go test -count=1 -p 1 ./internal/collection ./cmd/deepdata && go test -race -count=1 ./internal/collection`; if green → commit 1 `wip(recovery): bounded-memory journal replay and coverage streaming; snapshot streaming partial (RCV-01/02 pass, RCV-03 open)` including the untracked test files, `internal/index/segmented.go`, and the deletion of `cmd/recalltest/main.go` (superseded by `benchmarks/ddload`; say so in the message); commit 2 `docs(tasks): 2026-08-28 recovery session notes` with todo.md/lessons.md/installation.md as-is (they get restructured in S3). If not green: the first slice becomes "make it green or `git stash` only `snapshot.go` hunks" — do not start docs on a red tree.

**Branch:** `git branch -m gnhf/i-want-you-to-mnake-26a28a develop` (local only; PR #4 tracks the remote ref and is unaffected). Push of `develop` needs explicit authority — STATE.json's "push_current_branch: authorized" covered the old name. Do not delete `origin/gnhf/…`. `.gnhf/` archival mechanics in section 5; the preserved DeepData journal copies at `/home/omen/var/deepdata` and `/run/media/omen/Storage/miniexa/deepdata` are untouched by everything here.

---

## 7. The Workflow (two runs, 15 agents nominal, ≤ 27 with every retry path)

Everything mechanical (git mv/rm, .gnhf move, branch rename, WIP commit) is done by the main loop in Bash **before** the workflow. Agents only read/write/judge; JS routes, retries, asserts.

**Run 1 — `docs-tooling` (3 agents):**
- P1 `facts` (1 agent, schema FACTS): runs greps and returns `{head_sha, version, rpcs[], mutating_rpcs[], routes[{method,path,permission}] (from collection_http.go dispatch), mcp_tools[], non_goals[] (todo.md:43-49 ∪ README:34-38), packages[{path, loc, reached_from_rc_binary, ci_job, hint}], commits_since_a99fe53[{sha,date,doc_impact}], stale_boxes[{file_line, closing_commit}]}`. JS asserts `rpcs.length === 11` and `mutating_rpcs.length === 6` — mismatch throws (the anchor must be right).
- P2 `scripts` (1 agent): writes `scripts/gates.py`, `scripts/check_docs_contract.py`, seeds `tasks/gates.json` from FACTS + STATE.json evidence + section-1 list, edits `hardening_check.sh` (`state-json`→`gates-check`), deletes STATE.json, renders PRE_RELEASE_STATUS.md, runs `check_docs_contract.py` on the still-drifted tree and returns `{files[], linter_exit, caught[]: rule ids and file:line, skipped[]}`. JS asserts `caught ⊇ {R1,R2,R4,R6,R7,R8,R9}` (the smoke test) and `skipped.length === 0`.
- P3 `verify-scripts` (1 agent, refute-first): reads both scripts; runs `gates.py check`, tampers a copy of gates.json (pass without evidence; stale render) to confirm each check fires; returns VERDICTS. Any `false` → one fixer pass (re-invoke P2 prompt with the false list) → re-verify once; still false → throw.
- Main loop reviews `git diff --stat`, commits S2.

**Run 2 — `docs-rewrite` (12 agents nominal):**
- `args = {facts (from run 1), inputs: 'tasks/journal/2026-09-01-redesign-architects.md', jobs: DOC_JOBS}`. JS asserts the six jobs' file sets are pairwise disjoint and disjoint from KEEP_SET before launching — no worktrees needed.
- DOC_JOBS: **J1** `tasks/todo.md` + gates.json roadmap IDs; **J2** `docs/GAP_ANALYSIS.md` (+ delete `benchmarks/GAP_ANALYSIS.md`, patch `product_manager_test.go:137`, `benchmarks/README.md:49`); **J3** `docs/ARCHITECTURE.md`; **J4** `README.md` + `docs/mcp.md` (runs `check_docs_contract.py --write` for generated blocks); **J5** API.md + why-vectordb.md + installation.md + CHANGELOG.md + grafana/* + security-patterns header + BENCHMARKS.md:42 + benchmarks/README.md rewrite; **J6** `docs/decisions/0001-0007` + `tasks/PROTOCOL.md` + journal files' headers.
- `pipeline(DOC_JOBS, author, verify, fixIfNeeded, reverifyIfFixed)`:
  - `author` (schema AUTHOR): `{files[], claims[{claim, evidence:"file:line"}], skipped[]}`. Prompt carries FACTS, the job's disposition rows from section 5, the template, the inputs file. `skipped.length > 0` ⇒ job status `incomplete` (never "completed").
  - `verify` (schema VERDICTS, different agent, refute-first: "grep the tree at HEAD; default to false when no evidence is found; check every factual sentence in the files, not only the author's list"): `{path, verdicts[{claim, location:"file:line", verdict:"true|false|unverifiable", evidence, fix?}]}`.
  - JS: `false.length === 0` → done; else exactly one `fix` agent (author prompt + false list) then one `reverify` over only the previously-false claims. Still false → job status `failed`, listed in the final throw.
- Barrier (justified: needs all docs): `consistency` (1 agent, schema `{conflicts[{doc_a, doc_b, what, chosen, reason}], untouched_keep_set: bool, deleted_present[]}`): same RPC/mutation counts everywhere; every gate ID in todo.md/GAP_ANALYSIS exists in gates.json; every ADR referenced exists; DELETE-set absent; KEEP-set untouched (`git diff --stat`). Conflict rule from the user: choose the more recent/tested source and flag the loser. Conflicts → one fixer → one re-check.
- `lint` (1 agent, schema `{checks[{name, exit_code, tail}], modified_files[]}`): runs `python3 scripts/check_docs_contract.py`, `python3 scripts/gates.py check`, `git diff --check`, `go vet ./benchmarks/review/`, `go test -short -run TestProductManagerReview ./benchmarks/review/`. Non-zero → one fixer → rerun.
- **Acceptance (JS, deterministic):** all jobs `verified`; `conflicts.length === 0`; all lint exit codes 0; `modified_files ⊆ ALLOWED_SET` (DELETE ∪ REWRITE ∪ PATCH ∪ NEW from section 5). Anything else → `throw new Error(JSON of failing jobs/claims)`. Return a manifest `{jobs, lint, consistency}`; the **main loop** (not an agent) reads `git diff --stat`, wires the linter into `ci.yml:44-45`, and commits S3.
- Schemas fixed above; `phase` labels `Facts / Author / Verify / Consistency / Lint`; no `Date.now()` (timestamps come from `args`).

---

## 8. Sequenced slices (each a consistent commit, each with its verification)

| # | Who | Content | Verify |
|---|---|---|---|
| S0 | hand | Two commits: WIP recovery code (+ untracked tests, segmented.go, recalltest deletion); session docs as-is | focused Go gates green; `git status --porcelain` empty |
| S1 | hand | `.gnhf` → `/home/omen/var/deepdata-archive/…` (read-only) + drop exclude line; `git branch -m develop`; `rm -r .agents .codex`; `git mv` PROTOCOL.md → `tasks/PROTOCOL.md`, BASELINE.md → `tasks/journal/2026-07-17-baseline.md`; `git rm docs/benchmarks.md deepdata-system-map.html internal/index/README.md docs/models/…/tokenizer.json`; lessons.md blank line; settings.local.json prune; write `tasks/journal/2026-09-01-redesign-architects.md` | `go build ./... && go vet ./...`; `grep -rn 'autonomy/PROTOCOL\|benchmarks\.md\|system-map' --include=*.md --include=*.sh --include=*.py` returns only `tasks/journal/` hits (todo.md:287 will be rewritten in S3 — acceptable? No: patch that one line here) |
| S2 | Workflow run 1 | `scripts/gates.py`, `scripts/check_docs_contract.py`, seeded `tasks/gates.json`, generated PRE_RELEASE_STATUS.md, STATE.json deleted, hardening_check.sh `gates-check`. Linter **not yet in CI** (tree still drifts) | `gates.py check` = 0; `check_docs_contract.py` ≠ 0 and its output (the known drift list) is pasted into the commit message as S3's contract |
| S3 | Workflow run 2 | All REWRITE/PATCH/NEW/DELETE rows of section 5 except S1's; ADRs; journal splits; todo.md; `benchmarks/GAP_ANALYSIS.md` deletion + PM test patch; linter wired into `ci.yml` after `:45` | `check_docs_contract.py` = 0; `gates.py check` = 0; `git diff --check`; `go vet ./...`; PM review test doc checks pass; `git status` clean |
| S4 | hand, later | `gates.py promote` for DUR-01..04 after re-running `hardening_check.sh --force` at the new HEAD (turns four `pass (stale)` into `pass`); push `develop` only with authority; ADR 0006 `proposed → accepted` when the user accepts the tower | render regenerated in the same commit (check enforces) |

Slices 0-2 leave no half-renamed state: S1 deletes only files with zero inbound links (plus the one todo.md:287 pointer, patched in-slice); GAP_ANALYSIS moves atomically in S3 with its test and link.

---

### Critical Files for Implementation
- `/home/omen/Documents/Project/DeepData/tasks/todo.md` — source of all gate statements, the four stale checkboxes, and the three session narratives to split out
- `/home/omen/Documents/Project/DeepData/tasks/autonomy/STATE.json` — evidence arrays to migrate into `tasks/gates.json` before deletion
- `/home/omen/Documents/Project/DeepData/docs/PRE_RELEASE_STATUS.md` — becomes the generated render; its prose feeds two journal entries and ADR 0002
- `/home/omen/Documents/Project/DeepData/scripts/hardening_check.sh` — receipt schema (`:262-274`) that `gates.py promote` copies from; `state-json` → `gates-check`
- `/home/omen/Documents/Project/DeepData/internal/collection/API.md` — the most-drifted product doc and host of the generated RPC/route blocks
- `/home/omen/Documents/Project/DeepData/api/proto/deepdata/v3/deepdata.proto` — truth source for R1/R2 (11 RPCs, 6 mutating)
