# Task: Agent-oriented retrieval (fff-inspired) — 2026-08-18

> Archived 2026-09-01 from tasks/todo.md:297-355 at a443cd0; unedited below this note.

## Task: Agent-oriented retrieval (fff-inspired) — 2026-08-18

**Scope:** additive, opt-in extensions to the canonical V3 search contract. All new
behavior is zero-value-identical to today when the new fields are absent. No durability,
journal, or schema changes. RC non-goal list untouched.

1. `internal/collection/types.go`: add `ScoreFloor`, `Fallback *FallbackParams`,
   `UsageBoost` to `SearchRequest`; `WeakMatch`, `BestScore`, `FellBackTo` to `SearchResponse`.
2. `internal/collection/usage.go` (new): in-memory per-collection frecency tracker
   (count × exp decay, half-life 1h, capped entries, own mutex). Recorded on returned
   search hits and GetDocument. **Session signal only — not durable, documented as such.**
3. `collection.go`: validate new fields; `searchSingleField` post-processing = floor
   filter → usage-boost re-order (raw scores kept in response) → finalize
   (BestScore = max raw score; WeakMatch = floor>0 && (empty || best<floor));
   fallback ladder: 2 query fields, run primary, if 0 hits (or best<threshold) run
   secondary and set `FellBackTo`; `fallback` and `hybrid_params` mutually exclusive.
4. Proto v3: `FallbackParams` message; SearchRequest fields 9/10/11
   (`score_floor`, `fallback`, `usage_boost`); SearchResponse fields 3/4/5
   (`fell_back_to`, `weak_match`, `best_score`). Regenerate via scripts/generate_proto.sh.
5. gRPC handler (`collection_grpc.go`): map new request fields, return new response fields.
6. HTTP V3 tenant search handler: accept new fields (DisallowUnknownFields struct),
   emit new response fields in `tenantSearchJSONResponse`.
7. Python SDK: `TenantFallbackParams` + request/response model fields + `search()` kwargs
   (sync + async) + client-side mutual-exclusion validation.
8. `cmd/deepdata-mcp` (new): stdio MCP server, no new deps; tools
   `deepdata_search`, `deepdata_get_document`, `deepdata_list_collections`;
   config via `DEEPDATA_URL`, `DEEPDATA_API_TOKEN`, `DEEPDATA_TENANT`.
9. Tests: collection unit tests (floor/weak/best/fallback/usage), canonical HTTP test,
   gRPC mapping test, MCP serve() test, Python pytest+mypy from sdk/python.
10. Docs: README agent-retrieval section; cookbook note.
11. Gates: `go build ./...`, `go test -count=1 -p 1 ./internal/collection ./cmd/deepdata`,
    gofmt, vet; then no-mistakes full review + review section here.

## Review — agent-oriented retrieval (2026-08-17, after full gate pass)

**Verdict: sound, additive, zero-value-identical. One error-code defect found and fixed; two missing transport tests added.**

Reviewed and verified:
- `usage.go`: bounded (250k cap, lowest-score eviction), harmonic decay stays representable for arbitrarily old entries, own mutex, never durable, documented as a nudge (boost < 1 enforced). Prune math (1e-6 threshold reachable by time.Duration range) is tested.
- `collection.go`: floor direction follows the field metric; raw scores preserved under usage re-order; stable sort keeps ties in input order; usage recorded only for the response actually returned.
- Fallback decision is floor-aware: the floor is applied to the primary answer first, so "zero surviving hits" and "best score worse than threshold" are one uniform "weak" predicate (types.go + collection.go doc comments corrected to state this); gRPC enforces exactly-two query fields, the engine caps at CanonicalMaxSearchFields=2, and "both fields present and distinct" implies exactly two, so no extra arity check is needed in the engine.
- Multi-field-without-fusion rejection is typed (ErrInvalidSearchArgument) with a message naming both valid routes; error string checked against the suite (no tests pinned the old wording).
- Validation rejects NaN/negative floor, boost >= 1, hybrid+fallback exclusivity, same-field and unknown-field ladders.
- Proto regen reproducible (scripts/generate_proto.sh produces byte-identical tree).
- Gates: go build ./..., go vet, gofmt on touched files, go test -count=1 -p 1 ./internal/collection ./cmd/deepdata ./cmd/deepdata-mcp, Python pytest (58 passed, 3 skipped), mypy strict — all green.

Defects found and fixed in this review:
1. **Client validation errors returned 500 / gRPC Internal** (engine used untyped fmt.Errorf). Added `collection.ErrInvalidSearchArgument`; wrapped all score_floor/usage_boost/fallback contract violations; HTTP maps to 400, gRPC to codes.InvalidArgument.
2. **Missing transport tests promised in item 9.** Added `cmd/deepdata/agent_retrieval_http_test.go` (canonical V3 HTTP surface: zero-value identity, fallback, floor+weak, usage_boost, five violation cases -> 400, unknown-field rejection) and `cmd/deepdata/agent_retrieval_grpc_test.go` (mapping of all three fields, response markers, engine-level InvalidArgument, handler admission). MCP test previously mocked the HTTP server, so it never proved server-side wiring.

Final re-review (this session) found and fixed:
1. **Stale doc comment**: `FallbackParams` in types.go still said the fallback decision is "independent of ScoreFloor" while the code applies the floor first. Wording corrected in both types.go and collection.go (behavior was already floor-aware and correct).
2. **Unmapped multi-field error**: "multiple query fields require HybridParams" was an untyped error (500/Internal). Now wrapped in `ErrInvalidSearchArgument` → HTTP 400 / gRPC InvalidArgument, message now names fallback as the alternate route.
3. **gofmt drift**: `gofmt -w internal/collection/` had reformatted three pre-existing unformatted files (migration.go, tenant_test.go, filtered_search_integration_test.go); reverted them to keep the RC diff minimal.
4. Verified proto field numbers (9/10/11 request, 3/4/5 response), gRPC request/response mapping, HTTP tenant response JSON tags, MCP canonical insert/upsert endpoints, SDK admission parity (mutual exclusion, finite checks, exactly-two rule).

Final gate after fixes: go build ./..., go vet (3 packages), gofmt on touched files, go test -count=1 -p 1 ./internal/collection (ok) + ./cmd/deepdata (ok, 141.7s) + ./cmd/deepdata-mcp (ok), Python pytest (58 passed, 3 skipped), mypy strict (0 errors).

Open items (pre-existing, not from this task): Cowrie dormant-build-tag limitation (Phase 8), offline backup/restore rehearsal (Phase 4), license choice (Phase 7). Nothing new opened.
