# 0009 — Media is vectors-in; ephemeral collections are the next durability class

- Date: 2026-09-01
- Status: accepted (owner, 2026-09-01)
- Supersedes: — (refines ADR 0002's "an online canonical compactor is a future decision" for disposable data)
- Evidence: cmd/deepdata/main.go:3684-3689 (the canonical store checkpoints only through Close/Save at shutdown); ADR 0002 (soft deletes, no online compaction); tasks/journal/2026-09-01-redesign-architects.md durability classes (:122-129); internal/filter/filter.go operator set (eq, range, in, regex, exists, geo); docs/BENCHMARKS.md ingest and time-to-searchable

## Context
The owner will index images and video, some of it from live streams. The engine is modality-blind: a frame embedded by CLIP or a similar model is an ordinary document, metadata carries source, timestamp and frame URL, and range filters answer time-window questions, so search over media works today over the caller-vectors contract of ADR 0001. What does not fit is the durability: every frame insert is a class-A journal record that fsyncs and fails closed, a delete is one soft record per document, and the journal is truncated only at graceful shutdown (ADR 0002). A stream that inserts and evicts frames all day is the churn that grew the 4.71 GB journal behind RCV-01 to RCV-06. Embedding frames in real time is a GPU pipeline whose cost dwarfs indexing.

## Decision
Media enters as vectors. The server embeds text only (CTL-02); frames are embedded at the capture side. Blobs never enter the store; metadata points at them. Retention today is time-window collections, one per hour or day per stream, dropped whole with delete-collection: one journal record that reclaims the whole index. The next durability class is the ephemeral collection: its documents write no journal record, it lives in memory only, it is empty after restart and its upstream rebuilds it, and `/v3/status` capabilities report the class. That is gate MED-01, after SYS-04. A multimodal binding beside `texts` (image bytes or URLs) is not scheduled; it rides on the CTL-02 plumbing above the engine when a batch harness asks for it.

## Consequences
No change to the six mutations, the class-A path or the RC scope; ADR 0002 stands. Media is neither a non-goal in `docs/ARCHITECTURE.md` nor claimed live: it is ADR 0001 applied. MED-01 is not a release gate; it enters the ledger open with a `-run Ephemeral` command and no evidence.
