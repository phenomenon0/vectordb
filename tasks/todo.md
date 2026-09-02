# DeepData — plan

Next action: RCV-03 — unified snapshots load with bounded memory (head of the RC critical path; the platform slices CTL-01..05, SYS-01..04 and MED-01 all pass, MEM-02 waits for a harness).

Status lives only in [tasks/gates.json](gates.json), rendered to [docs/PRE_RELEASE_STATUS.md](../docs/PRE_RELEASE_STATUS.md).
This file holds ordering and dependencies; it has no checkboxes. Gate ids below name the ledger row.

## Objective

Serve the seven agent needs of the 2026-09-01 redesign ([journal, section 5 stance](journal/2026-09-01-redesign-architects.md)):
learn the system from one place; speak text and get ranked text back; be told the next call when wrong;
know how confident each answer is; have six verbs, not sixty; have use of it make it better;
never be lied to by a doc or a status. Hardened single-node core first (the RC), then re-expansion per ADR 0006.

## Slices (gate ids)

Docs (slice S3, this rewrite):
1. DOC-02 — todo, PROTOCOL and lessons rewritten without checkboxes, pointing at the ledger · this slice
2. DOC-01 — docs contract linter passes on the whole tree · after DOC-02
3. DOC-03 — API.md HTTP route table generated from the routes subcommand · after CTL-04

Control surface (journal 5.2, 5.4-5.9):
4. CTL-01 — errors are prompts: code, hint, docs pointer via internal/apierror · after DOC-01
5. CTL-02 — text in, text out: Ollama first, hash embedder only when named · after CTL-01
6. CTL-03 — MCP server rewritten on the shared api/contract package · after CTL-02
7. CTL-04 — self-description: routes subcommand and per-result confidence · after CTL-03
8. CTL-05 — durable usage records survive restart · after CTL-04

Systems tower (journal 5.10):
9. SYS-01 — serverRuntime extracted from main; func main() constructs no VectorStore and NewVectorStore has no live caller · after CTL-05
10. SYS-02 — one IndexTypes vocabulary shared by server, SDK and docs · after SYS-01
11. SYS-03 — archive tag exists; retired packages deleted; cowrie, shard, sqlite removed from go.mod · after SYS-02
12. SYS-04 — Canonical prefix dropped; every live package has a doc.go · after SYS-03

Agent memory (journal 5.9; graph and extraction: journal 1.2, ADR 0006):
13. MEM-01 — ADR recorded for graph and extraction revival before any code returns · after CTL-02
14. MEM-02 — POST /docs/{id}/feedback stores feedback durably, covered by tests · after CTL-02

Recovery (parallel, by hand; RCV-03 is the only release gate among them):
15. RCV-03 — unified snapshots load with bounded memory · after RCV-02
16. RCV-04 — generated small-journal tests fail closed on truncated and corrupt records · after RCV-01
17. RCV-05 — replay rehearsal under a cgroup memory hard cap, probe root only · after RCV-01
18. RCV-06 — internal/index/segmented.go measured under the cold-start memory envelope · after RCV-01

Release path (every rerun binds to the frozen candidate):
19. DUR-05 — focused persistence matrices pass serially · after RCV-03
20. SDK-02 — live authenticated SDK contract, including restart persistence · after DUR-05
21. SOAK-01 — soak holds recall, restart-under-load and memory growth within envelope · after DUR-05
22. SOAK-02 — restart reclaims memory after the memory-drift fix · after SOAK-01
23. SEC-01 — gitleaks, govulncheck, gosec and trivy clean on the frozen tree · after DUR-05
24. REV-01 — every open gate has a named owner decision before freeze · after SEC-01
25. REL-04 — candidate SHA frozen; manifests and changelog name it · after REV-01
26. EVID-02 — truth ledger passes gates.py check --release · after REL-04 and every release gate
27. CI-05 — cmd/deepdata-mcp in the CI package list, its tests run · after CTL-03 (not a release gate)
28. CI-06 — RC smoke job runs tests/smoke_test.sh in CI · after CI-01 (not a release gate)
29. PUB-01 — branch pushed, CI green on the pushed head · after EVID-02; needs owner authority
30. PUB-02 — release tag created and pushed · after PUB-01; needs owner authority
31. PUB-03 — container image and Python package published · after PUB-02; needs owner authority

## Critical path

RC tag: RCV-03 → DUR-05 → SDK-02, SOAK-01 → SOAK-02, SEC-01 → REV-01 → REL-04 → EVID-02 → PUB-01 → PUB-02 → PUB-03.
Platform: DOC-02 → DOC-01 → CTL-01 → CTL-02 → CTL-03 → CTL-04 → CTL-05 → SYS-01 → SYS-02 → SYS-03 → SYS-04 → MED-01;
MEM-01 and MEM-02 fork after CTL-02; DOC-03 lands with CTL-04; CI-05 lands with CTL-03.
The paths share only the tree: a change inside a DUR gate's scope (cmd, internal, api, go.mod, go.sum for DUR-01/03/04) turns that gate stale (gates.py is_stale; DUR-01 note), so freeze last.
PUB-01..03 and SYS-03's archive tag need owner authority (gates.json notes); no other gate does.

## Parked

Each item names the decision or gate that owns it; none is scheduled:
- Module rename to deepdata: not done; SYS-04 drops the Canonical prefix instead (journal 09-01, section 5.3).
- OpenAPI: not generated until a human consumer exists; CTL-03's api/contract JSON Schema is the source (section 5.3).
- GraphRAG, graph reranking and extraction revival: no code returns before MEM-01's ADR; internal/graph and internal/extraction stay dormant behind the canonical wall, internal/feedback goes with SYS-03's deletion (ADR 0006, accepted 2026-09-01).
- Benchmark reruns: none; benchmarks/results/ and benchmarks/competitive/live/ are frozen history, docs/BENCHMARKS.md holds the live numbers, [ADR 0005](../docs/decisions/0005-ingest-levers-segments-then-allocs-then-group-commit.md) fixes the ingest-lever order.
- Code-tree deletion: no package is deleted before SYS-03's archive tag archive/pre-narrowing-v2 exists (owner authority granted 2026-09-01; the tag is local until a push is authorised).
- Image and video: vectors-in today over ADR 0001 (frames embedded at the capture side, blobs stay outside, time-window collections dropped whole for retention); the ephemeral collection class is MED-01, landed 2026-09-01 ([ADR 0009](../docs/decisions/0009-media-is-vectors-in-ephemeral-collections-next.md)).
- desktop/, the web UI embed and tests/ui: retirement is [ADR 0006](../docs/decisions/0006-re-expand-to-agent-memory-platform.md), accepted by the owner on 2026-09-01 (no user of either UI exists yet; the platform is planned for agents); they are deleted under SYS-03.

## Where truth lives

- [tasks/gates.json](gates.json) — the ledger: 53 gates, each with statement, command, scope, release_gate, status, evidence, note; scripts/gates.py check validates it, gates.py promote binds a receipt.
- [docs/PRE_RELEASE_STATUS.md](../docs/PRE_RELEASE_STATUS.md) — generated by gates.py render; never hand-edit, gates.py check fails on a stale render.
- [tasks/PROTOCOL.md](PROTOCOL.md) — the rules: sources of truth, staleness, cold start, dirty tree, decision rules, escalation triggers, retry and timeout budget, authority.
- [tasks/lessons.md](lessons.md) — append-only correction → rule.
- [tasks/journal/](journal/) — one file per session, frozen once written; the 2026-08-17, 2026-08-22 and 2026-08-28 entries are the sections this file carried at a443cd0.
- [docs/GAP_ANALYSIS.md](../docs/GAP_ANALYSIS.md) — claim → truth → fix at HEAD; [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) — the tower and the machine-read non-goals; [docs/decisions/](../docs/decisions/) — ADRs 0001-0009.
- The pre-ledger plan (phases 0-9, frozen RC scope, decision rules, escalation triggers, retry budget) is tasks/todo.md at commit a443cd0 in git history; its phases are the DUR/SDK/PKG/OPS/SOAK/SEC/REL/CI/EVID/REV gates.
