# 0008 — Graph and extraction return only as signals over text-in

- Date: 2026-09-01
- Status: proposed (accepted together with ADR 0006; until then both packages stay dormant)
- Supersedes: — (refines ADR 0006's "graph and extraction stay dormant until MEM-01 records their own ADR")
- Evidence: tasks/journal/2026-09-01-redesign-architects.md — dormant-tree rows (:37-38), durability classes (:126-127), feedback row (:142), architect B on graph as class C built lazily (:680, :781) and extraction as an out-of-band client (:682)

## Context
`internal/graph` (CSR + PageRank/PPR) and `internal/extraction` (LLM knowledge-graph extraction over Ollama/OpenAI) compile but are unreachable behind the canonical wall (journal :37-38). Both serve the agent-memory thesis and both need text-in, which landed with CTL-02 (74b9e65). Reviving them as they were would put an LLM call on the request path and a second persisted authority for facts already in the journal (:781); `internal/feedback` is retired for the same reason (:142, SYS-03).

## Decision
Graph returns as a **class C** signal only: built lazily in memory from a collection's documents on the first graph-weighted search, invalidated on mutation, capped by node count, never persisted, joining fusion as the third result set that `internal/hybrid` already accepts (:680). Extraction returns as an **out-of-band client**, never a server code path: a separate process that reads and writes over the V3 API like any other client, storing extracted facts as ordinary documents through the six journal mutations (class A via the public contract) and relation weights as class-B signals (:682). No new journal mutation types, no LLM provider inside the server beyond the one process embedder `DEEPDATA_EMBEDDER` names. Preconditions before any code returns: CTL-05 proves the class-B loud-discard mechanism, and a harness states the need with a recall measurement to beat. Until then each package carries a `doc.go` naming its status and this ADR (MEM-01), and SYS-03 keeps both rows KEEP-DORMANT.

## Consequences
Neither package enters the RC surface, the CI RC jobs or `tasks/gates.json` as a release gate; `hybrid.HybridSearchWithGraph` stays the integration point and needs no change. MEM-02 (`POST /docs/{id}/feedback`) is independent of both: it is a `Touch` on the usage signal, not a graph edge.
