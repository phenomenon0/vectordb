# Session 2026-08-28 — bounded recovery after Mini-Exa replay OOM

> Archived 2026-09-01 from tasks/todo.md:392-470 at a443cd0; unedited below this note.

## Session 2026-08-28 — bounded recovery after Mini-Exa replay OOM

### Evidence and safety boundary

- [x] Recovered the exact Hermes thread and its continuation. The completed
  `release-v1` collection contained 301,816 documents and answered live
  dense/sparse/hybrid queries, but startup replay of a 4,714,504,171-byte
  journal was OOM-killed twice on the 30 GiB Linux host.
- [x] Preserve both journal copies as read-only recovery evidence:
  `/home/omen/var/deepdata` and
  `/run/media/omen/Storage/miniexa/deepdata`. Do not start either with the
  pre-fix binary, and never use an original copy for a replay experiment.
- [x] Root cause identified in code: `readAfter` retains all raw record
  payloads, startup decodes them all into a second mutation slice, replay then
  builds the live indexes while both copies remain resident, and the recovery
  checkpoint creates additional whole-state JSON copies.
- [x] Attribute and preserve the pre-existing dirty segmented-index and SDK
  work before accepting, revising, or discarding any part of it.

### Implementation phases

- [x] Replace all-at-once journal recovery with a fail-closed bounded-memory
  path. Validate the complete frozen/current artifact sequence before mutating
  loaded state, then reopen and decode/apply one record or bounded batch at a
  time. Preserve partial-tail repair, exact LSN ordering, and format
  compatibility.
- [x] Remove whole-journal retention from checkpoint coverage verification.
  Coverage checks must stream headers/checksums and retain only aggregate LSN
  bounds.
- [ ] Make unified snapshots bounded-memory and crash-safe. Avoid manager,
  tenant, payload, and envelope byte slices coexisting at full-store size;
  retain checksum, store-ID, atomic replace, mode, and fail-closed load
  invariants. Keep v1 snapshot readability or provide a tested migration path.
- [x] Review segmented HNSW separately. It may improve construction throughput,
  but multiple graphs plus merge-on-export can increase peak memory; it is not
  a replay-OOM fix until measured under the same cold-start envelope.
- [ ] Add generated small-journal tests for corrupt-record fail-closed behavior,
  partial tails, LSN gaps, two-pass recovery, crash/reopen, snapshot format
  compatibility, and bounded retained payloads. Run focused Go tests with
  `-p 1`; do not point tests at the production journal.
- [ ] Rehearse on a fresh store under a hard memory cap, then on an explicit
  disposable copy. A recovery candidate passes only after cold start, document
  count parity, dense/sparse/hybrid query parity, graceful close, second cold
  start, and measured peak RSS below the host budget.

### Resource gate

- Reserve at least 8 GiB for the desktop and existing services. No recovery
  probe may rely on swap thrashing as success. Start with one replay worker and
  one index-build worker; increase concurrency only from measured per-worker
  peak memory.
- The RTX 3090 embedding batch remains bounded at the proven
  `batch_size=16`, `max_seq_length=256` until an isolated adaptive probe proves
  a larger batch. The Mac may handle bounded preprocessing/embedding, but the
  persistent DeepData runtime remains on Linux for this release candidate.
- Every heavy command needs an explicit fresh data root, timeout, append-only
  log, hard memory limit, and periodic progress receipt. No service is declared
  recovered from a live query alone.

### Disposable replay checkpoint

- Fresh private root: `/home/omen/var/deepdata-recovery-probe-iBoWBE`
  (`0700`, internal Btrfs). It contains reflinked, `0600` copies of only the
  initialization marker, 257-byte snapshot, and frozen journal. It has no
  copied lock file and is not a production path.
- Source and disposable SHA-256 identities match: marker
  `ec1eed9914fc544afade6b7bcb772ab6f0c25d63e133a78ef6dc56d2edf76940`,
  snapshot `511f27e737e07e727e34a6380526e9e197f591ccaba1524befa4acebabd18b96`,
  frozen journal
  `5801e1028051994a5b54f6bd1673f32844549dced7381bad2d8defadae0554e5`.
- Adversarial review found no WAL-replay blocker. Snapshot promotion remains
  gated on exact cross-pass artifact binding, next-ID/schema preflight, and
  bounded legacy reads. The server must not start this copy until those fixes,
  focused tests, and the cgroup launcher gate all pass.
- A broad `internal/index` run reached the repository's existing 10-minute
  timeout in `TestPQADC_QPS_Comparison`; it was CPU-progressing at low RSS.
  The same run passed `internal/collection`, and its 100k-vector scale check
  reported 381 MiB RSS. Use changed-path tests rather than treating the long
  PQ performance test as a recovery regression.
