# 0002 — No online compactor in the RC

- Date: 2026-07-22
- Status: accepted
- Supersedes: —
- Evidence: 14d1442 (fix(hnsw): reclaim tombstones on snapshot Import); tasks/journal/2026-07-22-exact-sha-evidence.md:140-176 (memory-drift finding) and :247-252 (Current Decision); tasks/lessons.md:70-75

## Context
Canonical HNSW Delete is soft-only and no online compaction is reachable in canonical mode; before 14d1442 restart did not reclaim either, so RSS and the on-disk snapshot grew with cumulative deletes (journal :140-147). Separately, an online import swapped live state while keeping the previous WAL generation, so a crash could replay old mutations into the imported snapshot (lessons.md:70-73).

## Decision
No online compaction and no online snapshot import cross the durability boundary in the RC. Reclamation happens at checkpoint and restart (Import skips deleted entries, 14d1442); administrative moves are an offline procedure: stop the server, copy the whole root, export/import explicitly, verify semantically.

## Consequences
Single-process delete churn drifts until the next graceful checkpoint or restart, by design (journal :172-176). OPS-01 tracks the offline backup/restore drill; RCV-01 to RCV-06 track bounded-memory recovery. An online canonical compactor is a future decision with no RC gate.
