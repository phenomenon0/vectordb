# 0006 — Re-expand to an agent-memory platform on the hardened core

- Date: 2026-09-01
- Status: proposed
- Supersedes: — (ADR 0001 stays in force for the RC tag; this ADR governs the tree after it)
- Evidence: tasks/journal/2026-09-01-redesign-architects.md (5dafec2) — §5 stance (:106-108), §5.1 the tower (:110-120), §5.2 durability classes (:122-129), §5.3 picked-not-averaged (:131-143), §5.4 to §5.9 (:145-192), §5.10 keystone and cut (:194-200)

## Context
The RC is a vectors-only single-node server (ADR 0001). An agent driving it needs to learn it from one place, speak text and get ranked text back, be told the next call when it is wrong, know each answer's confidence, have six verbs rather than sixty, have its use improve results, and never be lied to by a doc or a status (:108).

## Decision
Build the seven-layer tower, L0 durability to L6 truth (§5.1): text in via server-side embedding above the engine, which stays vectors-only (CTL-02); errors as prompts through one envelope (CTL-01); six MCP verbs on a shared contract package (CTL-03); self-description and per-result confidence (CTL-04); a durable usage signal as durability class B — loud discard, never fail-closed (CTL-05, §5.2); keystone extraction and retirement (SYS-01 to SYS-04); graph and extraction stay dormant until MEM-01 records their own ADR.
Assumption awaiting the owner (§5.10, :200): "ubiquity" means reachable from any agent runtime over MCP/HTTP/gRPC/SDK, not a desktop app. On that reading desktop/, the web-UI embed and tests/ui are retired under SYS-03. Until the owner accepts, those rows are frozen, not deleted.

## Consequences
The gates named above are the proof; none of them is a release gate for the RC tag. ADR 0001 remains the RC scope; docs/ARCHITECTURE.md does not drop server-managed embeddings from its non-goals block until CTL-02 lands (its label says so).
