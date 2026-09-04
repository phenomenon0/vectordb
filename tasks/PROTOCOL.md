# DeepData working protocol

Purpose: keep work on this repository recoverable and honest across sessions, agents and interruptions.
Status is derived from evidence or it is not status; nothing written here overrides a failing test.

## Sources of truth

In descending order; a lower source never overrides a higher one:

1. Approved scope: the non-goals block in `docs/ARCHITECTURE.md` and the decision records in `docs/decisions/`.
2. Passing tests at HEAD.
3. Git history: the commit graph and the working-tree diff.
4. `tasks/gates.json`, the evidence-bound ledger. Its render `docs/PRE_RELEASE_STATUS.md` is derived; never edit it by hand.
5. `tasks/todo.md`: ordering and next action only.
6. `tasks/journal/`, `tasks/lessons.md` and the local .deepdata-run receipts: narrative and logs, never authoritative.

## Staleness

`scripts/gates.py` never stores "stale". A pass gate is stale when `git diff --quiet <evidence.commit> -- <scope>` reports a change between the evidence commit and the worktree (scripts/gates.py:8-10, :75-83). A stale pass renders as "pass (stale)" and counts as open for release (scripts/gates.py:192-193, :216-217, :275-276).

## No checkboxes

`tasks/todo.md`, `tasks/PROTOCOL.md` and `tasks/lessons.md` carry no "- [ ]" or "- [x]" lines; linter rule R12 fails on them (scripts/check_docs_contract.py:40, :518-525). Status lives only in `tasks/gates.json`; a plan line names a gate id instead of ticking a box.

## Cold start

Read in this order before editing: `tasks/todo.md`, then `tasks/gates.json`, then `docs/PRE_RELEASE_STATUS.md`, then `tasks/lessons.md`, then the newest file in `tasks/journal/`, then `git status` and `git log -5`. Then run the `command` of the gate named as the next action; it is the cheapest decisive test.

## Dirty tree

A dirty tree is unverified work. Attribute every modified or untracked path before touching anything; preserve changes that are not yours and never stash or reset them. Resume inspects agent-owned edits line by line and either completes them or reverts them through a new patch.

## Decision rules

- Correctness and recoverability outrank performance and feature breadth.
- Do not introduce a persistence-format break without a tested dual-read or migration path.
- Never treat corrupt state as an empty new database; preserve originals for recovery.
- A deterministic failure is investigated, not retried unchanged.
- A suspected flaky test gets one controlled rerun; both outcomes remain in evidence.
- Network/API outages move remote work to the offline queue while local work continues.
- Credit exhaustion pauses new reasoning; already-launched deterministic jobs may continue.
- Resume always begins with repository/state reconciliation and the cheapest decisive test.
- Performance regressions over 10% on stable tests require investigation before acceptance.
- No phase is complete because code exists; its exit gate and evidence must pass.
- Do not push, publish, deploy externally, or use private credentials without explicit scope.

## Escalation triggers

Continue around non-blocking issues and defer them. Escalate only when progress truly needs:

- A legal owner/license choice.
- PyPI/domain/trademark ownership or private signing/release credentials.
- Authorization for an external write such as push, PR mutation, publication, or deployment.
- A destructive action affecting pre-existing user data or unrelated work.
- Acceptance of data loss, an incompatible migration, or a material expansion beyond the
  single-node RC scope.
- An unavailable mandatory external service after local substitutes and retries are exhausted.

## Retry and timeout budget

| Operation | Timeout | Retry rule |
|---|---:|---|
| File/query inspection | 15s | Retry once only for transient process failure |
| Network read | 60s | Three attempts with persisted exponential backoff |
| Dependency download | 10m | Two attempts; then defer to online queue |
| Focused compile/test | 5m | No unchanged retry for deterministic failure |
| Full short suite | 10m | One controlled retry only if flaky evidence exists |
| Race suite | 15m | One controlled retry; preserve both logs |
| Container/package build | 20m | One retry for daemon/network failure |
| Soak/chaos | 60m | Resume from scenario checkpoint; never fabricate completion |

## Authority

From the retired state ledger (tasks/autonomy/STATE.json:131-136 at 5dafec2): pushing the working branch was authorized; publication, private credentials and the license choice were not. The standing rule: no push, tag, GitHub Release, PyPI upload, container or Helm registry push, signing, or credential use without explicit authority for that act. PUB-01 to PUB-03 track the publication acts. The remote branch origin/gnhf/i-want-you-to-mnake-26a28a is the archive of the earlier run and is not to be deleted.

## Safety boundary for recovery work

From tasks/todo.md:394-410 and :430-431 at a443cd0 and tasks/lessons.md:117-119:

- The preserved journal copies at /home/omen/var/deepdata and /run/media/omen/Storage/miniexa/deepdata are read-only recovery evidence. Never start either with a pre-fix binary (one without the bounded-memory replay tracked by RCV-01 and RCV-02), and never use an original copy for a replay experiment; work on an explicit disposable copy (RCV-05).
- Never point tests at the production journal; run focused Go tests with -p 1.
- The recovery service binds to loopback and never logs its token; rotate the token before any LAN exposure.

## Recording evidence

`scripts/hardening_check.sh` runs one allowlisted check and writes a receipt at .deepdata-run/checks/CHECK/receipt.json carrying status, git_commit, tree_fingerprint and finished_at (scripts/hardening_check.sh:10, :145, :262-283). `python3 scripts/gates.py promote <check> <gate>` copies those fields into the gate's evidence, sets pass only when the receipt status is passed, and re-renders the status page (scripts/gates.py:322-362). Commit `tasks/gates.json` and `docs/PRE_RELEASE_STATUS.md` together with the change they describe; `python3 scripts/gates.py check` fails when the committed render differs from a fresh one (scripts/gates.py:281-289).

## Journal discipline

One file per session, named tasks/journal/YYYY-MM-DD-slug.md. A session written in place opens with "# YYYY-MM-DD — slug"; then the sections What happened, Evidence, Decisions (pointing at docs/decisions/NNNN or "none") and Lessons (pointing at tasks/lessons.md or "none"). Material archived from elsewhere keeps its original heading, followed by a one-line "> Archived DATE from PATH at SHA" quote, and is not edited below it. A journal file is never edited after its session; corrections go in a new entry.

## Commit discipline

Read and branch on every validation exit code before committing; never print a failed gate and continue to the commit (tasks/lessons.md:28-33). Format only the files you touched; check `git status` afterwards and revert unrelated reformatting (tasks/lessons.md:98-105). Stage only reviewed paths, one semantic commit per cohesive change, `git diff --check` clean. Commit messages end with the trailer Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>.
