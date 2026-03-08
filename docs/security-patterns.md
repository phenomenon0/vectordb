# Security & Correctness Patterns

Recurring anti-patterns found during the March 2026 audit, with fixes applied.
Use this as a checklist when reviewing new code.

## 1. Authentication != Authorization

**Pattern**: Guard middleware authenticates a request and injects a `TenantContext` into the request context. Inner handlers then extract an identity from a _different_ source (URL path, `X-Tenant-ID` header) and use it for data access without checking it against the authenticated context.

**Example (fixed)**: The v3 `/tenants/{id}` handler extracted `tenantID` from the URL and accepted an `X-Tenant-ID` header override. Any valid token could act on any tenant.

**Fix**: `collection_http.go` now calls `security.GetTenantContextFromContext(r.Context())` and rejects requests where the authenticated tenant doesn't match the URL tenant (unless admin).

**Rule**: Every handler that takes an identity from the URL or headers MUST validate it against the authenticated context from the guard middleware. Never trust URL/header identities alone.

## 2. Go Value Semantics (pass-by-value mutation loss)

**Pattern**: A function takes a struct by value and mutates a field (typically ID assignment). The caller expects to read the mutation but gets the original zero value because Go copies the struct.

**Example (fixed)**: `Collection.Add(ctx, doc Document)` assigned `doc.ID = c.nextID` on the local copy. All HTTP handlers returned `id: 0`.

**Fix**: Changed `Add`, `AddDocument`, and `TenantManager.AddDocument` to take `*Document`.

**Rule**: If a function assigns an ID or mutates state that the caller needs to observe, it MUST take a pointer. When reviewing, look for `func Foo(x SomeStruct)` where `x.Field = ...` appears inside.

## 3. Partial Failure + All-or-Nothing Cursor

**Pattern**: A batch-apply loop stops on the first error, but the cursor (or progress tracker) only advances after the _entire_ batch succeeds. On retry, already-applied entries are replayed, causing duplicate-ID errors or infinite loops.

**Example (fixed)**: `ApplyEntries` stopped on first error. The follower only advanced its WAL cursor after the whole batch. Partial apply + retry = permanent wedge.

**Fix**: `ApplyEntries` now returns `(lastAppliedSeq, error)`. The follower advances the cursor to `lastAppliedSeq` even on partial failure.

**Rule**: Batch operations must either (a) be fully idempotent so replays are safe, or (b) track per-entry progress and advance the cursor to the last successfully applied item.

## 4. TOCTOU in Resource Acquisition

**Pattern**: Check if a resource is available, release it, then use it later. An attacker (or concurrent process) can claim the resource in the gap.

**Examples (fixed)**:
- **Port race** (`lib.rs`): `find_free_port()` bound a port, immediately dropped the listener, then spawned a sidecar on that port. A local attacker could bind the port in the gap. Fixed with bind-and-hold: the listener is kept alive until just before sidecar spawn.
- **WAL recovery**: `openSegment()` reopened a corrupted segment with `O_APPEND` without truncating the bad suffix. Post-recovery writes landed behind corruption and were silently skipped. Fixed by truncating to the last good entry before appending.

**Rule**: When acquiring a resource (port, file region, lock), hold it continuously from check to use. Never check-release-acquire.

## 5. Write Under Read Lock

**Pattern**: Code acquires `mu.RLock()` (read lock), but a code path inside can mutate shared state — causing a data race since multiple goroutines may hold RLock concurrently.

**Example (fixed)**: `DiskANNIndex.Export()` held `mu.RLock()`. The export path called `readFromDisk()`, which could fall through to `readFromDiskLinearScan()`, which mutates `unquantizedOffsetIndex`.

**Fix**: Changed `Export()` to use `mu.Lock()` (write lock).

**Rule**: When reviewing code under `RLock`, trace every function call to confirm nothing writes to shared state. If any path mutates, use a write lock. Pay special attention to "fallback" and "backward compatibility" code paths.

## 6. Unbounded User-Controlled Prometheus Labels

**Pattern**: A Prometheus metric uses a value from user input (request body field, URL path, header) as a label without validation. An attacker can generate infinite unique label combinations, exhausting Prometheus memory.

**Examples (fixed)**:
- `req.Mode` was emitted directly into `SearchRequestsTotal.WithLabelValues()`. Any unknown string was accepted. Fixed with an allowlist (`ann`, `scan`, `lex`).
- `r.URL.Path` was used as a metric label in `HTTPMiddleware`. Paths with dynamic segments (tenant IDs, collection names) create unbounded cardinality. Fixed with `normalizeMetricsPath()` that replaces dynamic segments with `:id`/`:name` placeholders.

**Rule**: Every `WithLabelValues()` call must use either (a) a hardcoded string, (b) a value from a fixed enum/allowlist, or (c) a normalized/bucketed value. Never pass raw user input.

## 7. Inner Break / Outer Continue Mismatch

**Pattern**: In nested loops, `break` exits only the inner loop. If the outer loop has error-handling logic that `continue`s to the next iteration, a `break` from the inner loop falls through to that logic instead of stopping the outer loop.

**Example (fixed)**: Batch insert with `continue_on_error=false`. Sparse vector conversion errors used `break` in the inner field loop, but the outer doc loop's `if hasError { continue }` kept going — producing partial writes when the caller asked for all-or-nothing.

**Fix**: Added explicit `if !req.ContinueOnError { break }` in the outer loop's error check.

**Rule**: When `break` is used in an inner loop to signal "stop processing", ensure the outer loop also checks and breaks. In Go, consider labeled breaks (`break outer`) for clarity.

---

## Audit Checklist

When reviewing new code, check for:

- [ ] Handler extracts identity from URL/header — does it validate against `TenantContext`?
- [ ] Function takes struct by value — does it mutate fields the caller reads?
- [ ] Batch apply stops on error — does the caller advance cursor per-entry?
- [ ] Resource checked then used — is it held continuously?
- [ ] Code under `RLock` — can any code path write?
- [ ] `WithLabelValues()` — is every label bounded?
- [ ] Nested loops with `break` — does the outer loop also stop?
