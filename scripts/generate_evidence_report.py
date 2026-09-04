#!/usr/bin/env python3
"""EVID-01: evidence report for the frozen candidate, assembled from receipts.

The report is a pure function of the frozen tree: the ledger, the receipts
scripts/hardening_check.sh wrote, and git. It invents nothing. A receipt that
is missing, stale against the frozen commit, or taken on a dirty tree is
reported as such rather than omitted -- an evidence report that silently drops
the evidence it could not find is worse than no report.

Stdlib only, matching scripts/gates.py.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import subprocess
import sys

CLEAN_TREE = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def git(root: pathlib.Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True
    )


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent


def is_stale(root: pathlib.Path, commit: str, scope: list[str]) -> bool:
    result = git(root, "diff", "--quiet", commit, "--", *scope)
    return result.returncode == 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="report path (default .deepdata-run/evidence/EVIDENCE-<sha>.md)",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="report on an unfrozen tree; the report says so in its header",
    )
    args = parser.parse_args(argv)

    root = repo_root()
    dirty = git(root, "status", "--porcelain").stdout.strip()
    if dirty and not args.allow_dirty:
        print("evidence: worktree is dirty; a frozen candidate has no uncommitted changes")
        print(dirty)
        return 1

    head = git(root, "rev-parse", "HEAD").stdout.strip()
    short = head[:7]
    output = args.output or root / ".deepdata-run/evidence" / f"EVIDENCE-{short}.md"
    output.parent.mkdir(parents=True, exist_ok=True)

    ledger = json.loads((root / "tasks/gates.json").read_text(encoding="utf-8"))
    gates = ledger["gates"]

    receipts: dict[str, dict] = {}
    checks_dir = root / ".deepdata-run/checks"
    if checks_dir.is_dir():
        for entry in sorted(checks_dir.iterdir()):
            path = entry / "receipt.json"
            if path.is_file():
                try:
                    receipts[entry.name] = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    receipts[entry.name] = {"_unreadable": str(exc)}

    lines: list[str] = []
    add = lines.append
    add(f"# Evidence report — {short}")
    add("")
    add(f"- commit: `{head}`")
    add(f"- generated: {dt.datetime.now().astimezone().isoformat(timespec='seconds')}")
    add(f"- tree: {'DIRTY (--allow-dirty)' if dirty else 'clean'}")
    add(f"- generator: `scripts/generate_evidence_report.py`")
    add("")
    add("Generated from `tasks/gates.json` and the receipts under")
    add("`.deepdata-run/checks/`. Nothing here is hand-written.")
    add("")

    release = [g for g in gates if g["release_gate"]]
    blocking: list[str] = []
    add("## Release gates")
    add("")
    add("| Gate | Status | Evidence commit | Fresh | Tree | Receipt |")
    add("| --- | --- | --- | --- | --- | --- |")
    for gate in sorted(release, key=lambda g: g["id"]):
        ev = gate.get("evidence") or {}
        commit = ev.get("commit")
        if gate["status"] != "pass":
            fresh = "—"
        elif not commit:
            fresh = "no evidence"
        else:
            fresh = "no" if is_stale(root, commit, gate["scope"]) else "yes"
        tree = "clean" if ev.get("tree_fingerprint") == CLEAN_TREE else (
            "dirty" if ev.get("tree_fingerprint") else "—"
        )
        rec = ev.get("receipt")
        if not rec:
            receipt_cell = "—"
        elif (root / rec).exists():
            receipt_cell = f"`{rec}`"
        else:
            receipt_cell = f"**MISSING** `{rec}`"
        if gate["status"] != "pass" or fresh != "yes" or tree != "clean":
            blocking.append(gate["id"])
        add(
            f"| {gate['id']} | {gate['status']} | "
            f"{(commit or '—')[:7]} | {fresh} | {tree} | {receipt_cell} |"
        )
    add("")
    if blocking:
        add(f"**{len(blocking)} of {len(release)} release gates do not meet the bar:** "
            + ", ".join(blocking))
    else:
        add(f"All {len(release)} release gates are pass, fresh, and clean-tree.")
    add("")

    add("## Receipts at this head")
    add("")
    at_head = [
        name
        for name, r in receipts.items()
        if r.get("git_commit") == head and r.get("tree_fingerprint") == CLEAN_TREE
    ]
    if not receipts:
        add("No receipts found under `.deepdata-run/checks/`.")
    else:
        add("| Check | Status | Exit | At frozen head | Tree | Host | Finished |")
        add("| --- | --- | --- | --- | --- | --- | --- |")
        for name, receipt in receipts.items():
            if "_unreadable" in receipt:
                add(f"| {name} | unreadable | — | — | — | — | — |")
                continue
            host = receipt.get("host") or {}
            where = " ".join(
                str(p)
                for p in (host.get("hostname"), host.get("kernel"), f"{host.get('cpus')}cpu")
                if p is not None
            ) or "unrecorded"
            add(
                f"| {name} | {receipt.get('status')} | {receipt.get('exit_code')} | "
                f"{'yes' if receipt.get('git_commit') == head else 'no'} | "
                f"{'clean' if receipt.get('tree_fingerprint') == CLEAN_TREE else 'dirty'} | "
                f"{where} | {receipt.get('finished_at', '—')} |"
            )
    add("")

    add("## Non-release gates")
    add("")
    others = [g for g in gates if not g["release_gate"]]
    by_status: dict[str, list[str]] = {}
    for gate in others:
        by_status.setdefault(gate["status"], []).append(gate["id"])
    for status in sorted(by_status):
        add(f"- **{status}** ({len(by_status[status])}): " + ", ".join(sorted(by_status[status])))
    add("")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"evidence: wrote {output.relative_to(root)}")
    print(f"evidence: {len(release) - len(blocking)}/{len(release)} release gates meet the bar")
    print(f"evidence: {len(at_head)} of {len(receipts)} receipts taken at {short} on a clean tree")

    # The bar is this gate's own statement -- "generated for the frozen
    # candidate with rehearsal receipts" -- and nothing more. Whether every
    # release gate is green is EVID-02's claim, printed above but not enforced
    # here: making it a precondition would put this gate inside the ledger it
    # reports on, and a gate that must already be green to be generated can
    # never be generated the first time.
    #
    # A report carrying no receipt from the frozen candidate is the vacuous
    # case this guard exists to catch.
    if not at_head:
        print(
            f"evidence: no receipt was taken at {short} on a clean tree; "
            "the report cites nothing about the frozen candidate"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
