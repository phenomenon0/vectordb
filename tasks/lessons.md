# DeepData Workflow Lessons

## Product Scope Must Be Explicit

- Correction: benchmark completion was initially easy to read as overall product completion.
- Rule: every report states its scope, and benchmark status links to the product release plan.

## Run Component Tools from Their Configuration Root

- Correction: mypy appeared green from the repository root but reported 16 strict errors
  when run from `sdk/python`, where the SDK configuration is loaded.
- Rule: validation commands record their working directory and prove that the intended
  component configuration was discovered.

## Autonomy Uses Current Capabilities

- Correction: the original workflow skill assumed an agent could not safely edit source
  autonomously and described shell scripts as surviving model outages too broadly.
- Rule: explicit implementation scope authorizes safe source edits with tested commit
  checkpoints; credit/API outages pause reasoning while local deterministic work may continue.

## Markers Are Not Evidence

- Correction: the old runner template wrote a DONE marker even after a required retry failed.
- Rule: required tasks become complete only after validation passes; failures are recorded as
  failed/deferred, and commits plus test evidence are the semantic checkpoints.

## Validation Must Gate the Next Action

- Correction: a staged diff check reported an error, but a later command in the same orchestration
  batch still created the commit.
- Rule: read and branch on each validation exit code before issuing a commit; never merely print a
  failed gate and continue to the state-changing action.

## Persist Identity and Ordering, Not Just Payloads

- Correction: snapshots preserved vectors and IDs but silently discarded pagination sequence
  numbers; compaction then reused count-derived sequences and could deadlock while saving.
- Rule: every externally observable ordering/identity counter gets a persisted high-water mark,
  codec round-trip tests, compaction coverage, and restart assertions.

## Integrity Must Cover Semantics

- Correction: the historical checksum only covered three counters, and index blobs had neither a
  persisted type nor an artifact checksum, so content corruption and IVF/FLAT-to-HNSW drift passed.
- Rule: canonical checksums cover full logical state; derived artifacts carry typed descriptors,
  independent digests, dimensions, and semantic count checks before they are accepted.

## Recovery Logs Outlive Recovery

- Correction: replay once deleted the WAL immediately, before the recovered state had been
  synchronously written and directory-synced into a snapshot.
- Rule: validate every recovery segment before mutation, replay by a persisted monotonic LSN,
  commit a checkpoint containing that high-water mark, and only then durably remove logs.

## Indeterminate I/O Poisons the Writer

- Correction: an entry could reach the file and then report a close or directory-sync error;
  the next request reused its LSN and made the log unrecoverable.
- Rule: after any write/sync/close outcome that may have reached storage, reject further writes
  and fail readiness until restart validates the artifact; never reuse an uncertain sequence.

## Snapshot Commits Need Their Own Serialization

- Correction: two saves used the same temporary path and an older capture could rename after a
  newer capture, silently replacing the latest durable snapshot.
- Rule: use unique same-directory temporary files and serialize the full capture-to-rename
  commit boundary; test the exact stale-rename interleaving.

## Narrow the RC Before Shipping Unsafe Recovery

- Correction: online import swapped live state while retaining the previous WAL generation, so
  a crash could replay old database mutations into the imported snapshot.
- Rule: if an administrative feature cannot cross the durability boundary transactionally,
  disable it explicitly in the RC and document an offline procedure instead of exposing it.

## Caller-Validation Errors Must Be Typed at the Engine Boundary

- Correction: score_floor/usage_boost/fallback contract violations surfaced as
  HTTP 500 / gRPC Internal because the engine returned plain fmt.Errorf
  values that transports could not classify.
- Rule: validate caller-supplied search parameters and wrap the error in a
  typed sentinel (ErrInvalidSearchArgument); transports map it to HTTP 400 /
  codes.InvalidArgument. Transport tests must cover the violation paths,
  not only the happy path through a mocked upstream.

## Fallback and Confidence Floor Share One Weakness Predicate

- Correction: the FallbackParams doc comment claimed the fallback decision was
  "independent of ScoreFloor" while the code applied the floor to the primary
  answer before deciding. The floor-aware behavior is the correct one: "no
  confident result" (zero hits, or wiped out by the floor) must route to the
  secondary field, so both mechanisms must compose, not bypass each other.
- Rule: when two post-processing stages (filter + route) consume the same
  answer, state explicitly which stage runs first in the doc comment and keep
  the decision a uniform predicate over the post-filter answer.

## Package-Wide gofmt -w Pollutes a Minimal RC Diff

- Correction: running `gofmt -w internal/collection/` reformatted three
  pre-existing unformatted files unrelated to the change, bloating the RC diff
  with pure whitespace noise.
- Rule: format only the files you touched (list them explicitly); if you must
  format a whole package, check `git status` afterwards and revert unrelated
  reformatting before committing.
