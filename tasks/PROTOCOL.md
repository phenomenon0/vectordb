# DeepData Autonomous Resume Protocol

This protocol keeps production-hardening work recoverable across context compaction, agent
timeouts, credit exhaustion, process death, and network loss without pretending that model
reasoning continues while the model is unavailable.

## Two Execution Planes

1. The online reasoning plane owns investigation, code changes, review, replanning, and
   subagents. It is bound to the persistent production-hardening goal.
2. The local execution plane runs only predeclared deterministic commands such as builds,
   tests, benchmarks, packaging, hashing, and report generation.

During an API/credit outage, online reasoning stops. A detached local command already in
progress may finish and write its log/receipt. No shell script is allowed to invent a code
change or resolve an unexpected failure.

## Sources of Truth

In descending order:

1. The user's approved objective and scope.
2. Passing invariants/tests on the current Git tree.
3. The Git commit graph and working-tree diff.
4. `tasks/todo.md` for the human-readable plan.
5. `tasks/autonomy/STATE.json` for machine-readable orientation.
6. Local `.deepdata-run/` logs and receipts.

A checkbox or state value never overrides failing evidence.

## Checkpoint Contract

For each cohesive source change:

1. Record the intended file scope and failing/required test.
2. Preserve unrelated user changes; never stash or reset them.
3. Apply the smallest root-cause fix.
4. Run focused tests, then the appropriate broader gate.
5. Run formatting/static checks and `git diff --check`.
6. Stage only reviewed paths and create one semantic commit.
7. Update state with the verified commit, test, result, and next action.

If interrupted before commit, the dirty tree is treated as unverified work. Resume inspects it
line by line and either completes it or reverts only agent-owned edits through a new patch.

## Resume Audit

Every resumed turn performs these checks before editing:

1. Read the persistent goal, `tasks/todo.md`, `tasks/lessons.md`, and `STATE.json`.
2. Verify repository root, branch, HEAD, and remotes.
3. Inspect `git status`; attribute every modified/untracked path.
4. Compare HEAD with `last_verified_commit` and inspect intervening commits.
5. Rerun `last_validation.command` from its recorded working directory when feasible.
6. Inspect running processes, DeepData ports, containers, disk space, and temp artifacts.
7. Reconcile any local receipt with its command, exit code, input commit, and output hashes.
8. Move network-only actions to `offline_queue` if connectivity is unavailable.
9. Atomically record `resumed_at` and the next exact action.
10. Continue the first ready task on the critical path.

## Failure Classes

| Failure | Response |
|---|---|
| Network timeout, DNS, 502/503 | Persisted backoff; defer after three attempts |
| API rate/credit exhaustion | Stop model work; resume from state when available |
| Deterministic compile/test error | Diagnose immediately; do not retry unchanged |
| Suspected flaky test | One controlled rerun; retain both logs |
| Missing dependency while offline | Queue download; continue independent local work |
| Required credential/legal decision | Record blocker; continue every independent task |
| Unknown external write result | Query remote state before any retry |
| OOM/disk-full/resource collision | Capture evidence; retry only with a safe resource plan |
| Unexpected user/worktree change | Preserve it; avoid overlap or escalate if unavoidable |

## Offline Queue

Work that can proceed offline when dependencies are already present:

- Local code inspection and edits while the agent is available.
- Go/Python/UI builds and tests, smoke tests, cached containers, benchmarks, and report generation.
- Local commits, hashes, manifests, and evidence reconciliation.

Work deferred until network returns:

- Fetch/push, remote CI/PR/release checks, dependency or image pulls, vulnerability databases,
  PyPI/GHCR/Helm publication, cloud-model integration, and external research.

External writes are never blindly retried after an ambiguous timeout.

## State Update Rules

- Update `STATE.json` atomically and keep it valid JSON.
- Increment `generation` for each semantic state transition.
- Record timestamps with timezone and commands with their working directory.
- A `last_verified_commit` is valid only when its recorded validation passed.
- Required failures remain visible in `blocked` or `deferred`; they are never called skipped.
- The persistent goal is complete only after all release invariants in `tasks/todo.md` pass.

## Local Runner Safety

The deterministic runner, when added, must:

- Accept an explicit allowlist of commands; never evaluate model-generated shell text.
- Use argument arrays where possible, bounded timeouts, per-command logs, and exit receipts.
- Mark completion only after its validator passes.
- Record the input Git commit and refuse mutating commands on a changed tree.
- Survive disconnects through a detached process, but never claim it can replace the agent.
