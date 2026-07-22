# Production-Hardening Baseline

Captured: 2026-07-17T23:34:36-05:00

## Repository

- Branch: `gnhf/i-want-you-to-mnake-26a28a`
- Baseline commit: `a5720b80391277fd8d48da7729b033016ac27975`
- Worktree before runner implementation: clean
- Release target: headless single-node server RC

## Toolchain

| Tool | Baseline |
|---|---|
| Go | `go1.25.5 X:nodwarf5 linux/amd64` |
| Python | `3.14.2` |
| Node | `v22.20.0` |
| npm | `10.9.3` |
| Rust | `rustc 1.92.0` |
| Cargo | `1.92.0` |
| Docker CLI | unavailable |
| Helm CLI | unavailable |

The workspace volume had 202 GiB available (79% used). Unprivileged socket inspection was
restricted by the sandbox, so port ownership is rechecked when a local server is launched.

## Reproducible Checks

The allowlisted runner is [`scripts/hardening_check.sh`](../../scripts/hardening_check.sh).
Receipts and full logs live under ignored `.deepdata-run/checks/` and include the Git commit,
dirty-tree content fingerprint, working directory, exact command, timestamps, and exit code.

| Check | Baseline result |
|---|---|
| `state-json` | Passed |
| `benchmark-unit` | Passed: 8 tests |
| `go-storage` | Passed for storage, collection, and server packages |
| `go-short` | Passed across all Go packages; slowest package about 22 seconds |
| `python-unit` | Passed: 75 tests, 5 live-integration skips |
| `python-mypy` | Failed as expected: 16 `no-any-return` errors |
| `go-vet-cgo0` | Failed as expected: SIMD test imports the CGO-only `cref` package |

The first Go attempts failed because the default build/module caches were read-only. The runner
now uses a workspace-local build cache and the writable task module cache; those infrastructure
failures are retained in the append-only logs and are not misreported as product failures.

## Known Baseline Gaps

- Docker and Helm tooling must be installed or provided before deployment proof.
- UI dependencies are not installed in this checkout; the remote UI build previously passed,
  while Playwright still points at the wrong server binary.
- Five Python tests require a live `DEEPDATA_URL` and are not covered by the unit-only baseline.
- The full short race suite passed before this planning-only checkpoint; it remains a required
  exact-candidate gate and is not inferred from the non-race baseline.
