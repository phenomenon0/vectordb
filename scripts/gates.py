#!/usr/bin/env python3
"""Truth ledger: tasks/gates.json -> docs/PRE_RELEASE_STATUS.md (stdlib only).

check    validate the ledger, evidence binding, freshness, and the committed render
render   regenerate the status page; a pure function of the ledger and git
promote  copy a scripts/hardening_check.sh receipt into a gate's evidence, then render

Status is derived from evidence or it is not status. `stale` is never stored: a
`pass` gate is stale when `git diff --quiet <evidence.commit> -- <scope>` reports a
change between the evidence commit and the current worktree.
"""

from __future__ import annotations

import argparse
import difflib
import json
import pathlib
import re
import subprocess
import sys

EMPTY_TREE_FINGERPRINT = (
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)
ID_RE = re.compile(r"^[A-Z]{2,4}-\d{2}$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# EVID-02 records that `check --release` passed. It is graded by this very
# loop, so requiring it to already be `pass` makes it unreachable: it can
# only be promoted from a run that returned 0, and it cannot return 0 until
# it is promoted. It is excluded here and nowhere else -- every other rule,
# including evidence binding and the freshness check, still applies to it.
RELEASE_META_GATE = "EVID-02"
STATUSES = ("open", "pass", "blocked", "deferred", "retired")
GATE_FIELDS = (
    "id",
    "statement",
    "command",
    "scope",
    "release_gate",
    "status",
    "evidence",
    "note",
)
EVIDENCE_FIELDS = ("commit", "tree_fingerprint", "date", "receipt", "ci_run")
STATEMENT_WIDTH = 96


def git(root: pathlib.Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True
    )


def repo_root() -> pathlib.Path:
    probe = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    if probe.returncode == 0:
        return pathlib.Path(probe.stdout.strip())
    return pathlib.Path(__file__).resolve().parents[1]


def load_ledger(path: pathlib.Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_ledger(path: pathlib.Path, ledger: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(ledger, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def commit_resolves(root: pathlib.Path, sha: str) -> bool:
    return git(root, "cat-file", "-e", f"{sha}^{{commit}}").returncode == 0


def is_stale(root: pathlib.Path, gate: dict) -> bool:
    """True when any scoped path differs between the evidence commit and the worktree."""
    diff = git(
        root, "diff", "--quiet", gate["evidence"]["commit"], "--", *gate["scope"]
    )
    if diff.returncode not in (0, 1):
        raise RuntimeError(f"{gate['id']}: git diff failed: {diff.stderr.strip()}")
    return diff.returncode == 1


def schema_errors(ledger: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(ledger, dict) or ledger.get("schema_version") != 1:
        return ["ledger must be an object with schema_version 1"]
    gates = ledger.get("gates")
    if not isinstance(gates, list) or not gates:
        return ["ledger.gates must be a non-empty list"]
    seen: set[str] = set()
    for index, gate in enumerate(gates):
        label = gate.get("id", f"#{index}") if isinstance(gate, dict) else f"#{index}"
        if not isinstance(gate, dict):
            errors.append(f"{label}: gate must be an object")
            continue
        missing = [field for field in GATE_FIELDS if field not in gate]
        extra = [field for field in gate if field not in GATE_FIELDS]
        if missing:
            errors.append(f"{label}: missing fields {missing}")
        if extra:
            errors.append(f"{label}: unknown fields {extra}")
        if missing:
            continue
        if not isinstance(gate["id"], str) or not ID_RE.match(gate["id"]):
            errors.append(f"{label}: id must match {ID_RE.pattern}")
        if gate["id"] in seen:
            errors.append(f"{label}: duplicate id")
        seen.add(gate["id"])
        for field in ("statement", "command", "note"):
            if not isinstance(gate[field], str):
                errors.append(f"{label}: {field} must be a string")
        if not gate["statement"]:
            errors.append(f"{label}: statement must be non-empty")
        if not isinstance(gate["scope"], list) or not all(
            isinstance(p, str) and p for p in gate["scope"]
        ):
            errors.append(f"{label}: scope must be a list of non-empty pathspecs")
        if not isinstance(gate["release_gate"], bool):
            errors.append(f"{label}: release_gate must be a boolean")
        if gate["status"] not in STATUSES:
            errors.append(f"{label}: status must be one of {list(STATUSES)}")
        evidence = gate["evidence"]
        if evidence is not None:
            if not isinstance(evidence, dict):
                errors.append(f"{label}: evidence must be null or an object")
                continue
            e_missing = [field for field in EVIDENCE_FIELDS if field not in evidence]
            e_extra = [field for field in evidence if field not in EVIDENCE_FIELDS]
            if e_missing or e_extra:
                errors.append(
                    f"{label}: evidence fields must be exactly {list(EVIDENCE_FIELDS)}"
                )
                continue
            if not isinstance(evidence["commit"], str) or not SHA_RE.match(
                evidence["commit"]
            ):
                errors.append(f"{label}: evidence.commit must be a 40-hex sha")
            if not isinstance(
                evidence["tree_fingerprint"], str
            ) or not FINGERPRINT_RE.match(evidence["tree_fingerprint"]):
                errors.append(
                    f"{label}: evidence.tree_fingerprint must be a 64-hex sha256"
                )
            if not isinstance(evidence["date"], str) or not DATE_RE.match(
                evidence["date"]
            ):
                errors.append(
                    f"{label}: evidence.date must be an ISO date (YYYY-MM-DD)"
                )
            if evidence["receipt"] is not None and not isinstance(
                evidence["receipt"], str
            ):
                errors.append(f"{label}: evidence.receipt must be null or a path")
            if evidence["ci_run"] is not None and not (
                isinstance(evidence["ci_run"], str)
                and evidence["ci_run"].startswith("http")
            ):
                errors.append(f"{label}: evidence.ci_run must be null or a URL")
    return errors


def evidence_errors(root: pathlib.Path, gate: dict) -> list[str]:
    """Rule 2: pass needs resolvable evidence and a scope; other statuses may keep history."""
    label = gate["id"]
    evidence = gate["evidence"]
    errors: list[str] = []
    if gate["status"] == "pass":
        if evidence is None:
            return [f"{label}: status pass requires evidence"]
        if not gate["scope"]:
            errors.append(f"{label}: status pass requires a non-empty scope")
    if evidence is not None and not commit_resolves(root, evidence["commit"]):
        errors.append(
            f"{label}: evidence.commit {evidence['commit'][:7]} does not resolve in this repository"
        )
    return errors


def stale_ids(root: pathlib.Path, ledger: dict) -> set[str]:
    stale: set[str] = set()
    for gate in ledger["gates"]:
        if gate["status"] == "pass" and gate["evidence"] and gate["scope"]:
            if commit_resolves(root, gate["evidence"]["commit"]) and is_stale(
                root, gate
            ):
                stale.add(gate["id"])
    return stale


def status_word(gate: dict, stale: set[str]) -> str:
    return "pass (stale)" if gate["id"] in stale else gate["status"]


def cell(text: str, width: int | None = None) -> str:
    text = text.replace("|", "\\|").replace("\n", " ")
    if width is not None and len(text) > width:
        text = text[: width - 1].rstrip() + "…"
    return text


def render(root: pathlib.Path, ledger: dict) -> str:
    stale = stale_ids(root, ledger)
    gates = ledger["gates"]
    release = [g for g in gates if g["release_gate"]]
    fresh = [g for g in release if g["status"] == "pass" and g["id"] not in stale]
    stale_release = [g for g in release if g["id"] in stale]
    lines = [
        "# DeepData Pre-Release Status",
        "",
        "Generated by `python3 scripts/gates.py render` from `tasks/gates.json`. Do not hand-edit; "
        "`gates.py check` fails when this file differs from a fresh render.",
        "",
        f"Readiness: {len(release)} release gates, {len(fresh)} pass and fresh, "
        f"{len(stale_release)} pass (stale), {len(release) - len(fresh) - len(stale_release)} not passed. "
        "A stale pass counts as open for release.",
    ]
    families: dict[str, list[dict]] = {}
    for gate in gates:
        families.setdefault(gate["id"].split("-")[0], []).append(gate)
    for family, members in families.items():
        lines += [
            "",
            f"| {family} | status | commit | date | statement |",
            "|---|---|---|---|---|",
        ]
        for gate in members:
            evidence = gate["evidence"]
            commit = evidence["commit"][:7] if evidence else "—"
            date = evidence["date"] if evidence else "—"
            lines.append(
                f"| {gate['id']} | {status_word(gate, stale)} | {commit} | {date} | "
                f"{cell(gate['statement'], STATEMENT_WIDTH)} |"
            )
    lines += [
        "",
        "Update a gate: run its command (receipt-backed gates: `scripts/hardening_check.sh` with the check "
        "name and `--force`), then `python3 scripts/gates.py promote CHECK GATE`, then commit `tasks/gates.json` "
        "and the regenerated `docs/PRE_RELEASE_STATUS.md` in the same commit. Process: `tasks/PROTOCOL.md`.",
        "",
    ]
    return "\n".join(lines)


def cmd_check(args: argparse.Namespace) -> int:
    root = repo_root()
    failures: list[str] = []
    infos: list[str] = []
    try:
        ledger = load_ledger(args.ledger)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"gates: cannot read ledger {args.ledger}: {exc}")
        return 1
    failures += schema_errors(ledger)
    if failures:
        for line in failures:
            print(f"gates: {line}")
        return 1
    for gate in ledger["gates"]:
        failures += evidence_errors(root, gate)
    if failures:
        for line in failures:
            print(f"gates: {line}")
        return 1
    stale = stale_ids(root, ledger)
    for gate_id in sorted(stale):
        infos.append(f"{gate_id}: pass (stale) — scope changed since evidence commit")
    if args.release:
        for gate in ledger["gates"]:
            if not gate["release_gate"] or gate["id"] == RELEASE_META_GATE:
                continue
            if gate["status"] != "pass":
                failures.append(f"{gate['id']}: release gate is {gate['status']}")
            elif gate["id"] in stale:
                failures.append(f"{gate['id']}: release gate is stale")
            elif gate["evidence"]["tree_fingerprint"] != EMPTY_TREE_FINGERPRINT:
                failures.append(
                    f"{gate['id']}: release gate evidence was not taken on a clean tree"
                )
    fresh = render(root, ledger)
    try:
        committed = args.render_path.read_text(encoding="utf-8")
    except OSError as exc:
        committed = None
        failures.append(f"render {args.render_path} unreadable: {exc}")
    if committed is not None and committed != fresh:
        failures.append(
            f"render out of date: run `python3 scripts/gates.py render` and commit {args.render_path}"
        )
        diff = difflib.unified_diff(
            committed.splitlines(),
            fresh.splitlines(),
            "committed",
            "fresh",
            lineterm="",
            n=0,
        )
        infos += ["diff: " + line for line in list(diff)[:12]]
    for line in infos:
        print(f"gates: info: {line}")
    for line in failures:
        print(f"gates: {line}")
    if not failures:
        print(f"gates: ok ({len(ledger['gates'])} gates, {len(stale)} stale)")
    return 1 if failures else 0


def cmd_render(args: argparse.Namespace) -> int:
    root = repo_root()
    ledger = load_ledger(args.ledger)
    errors = schema_errors(ledger)
    if errors:
        for line in errors:
            print(f"gates: {line}")
        return 1
    args.out.write_text(render(root, ledger), encoding="utf-8")
    print(f"gates: rendered {args.out}")
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    root = repo_root()
    ledger = load_ledger(args.ledger)
    errors = schema_errors(ledger)
    if errors:
        for line in errors:
            print(f"gates: {line}")
        return 1
    gate = next((g for g in ledger["gates"] if g["id"] == args.gate), None)
    if gate is None:
        print(f"gates: unknown gate {args.gate}")
        return 1
    receipt_rel = pathlib.Path(".deepdata-run/checks") / args.check / "receipt.json"
    receipt_path = root / receipt_rel
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"gates: cannot read receipt {receipt_rel}: {exc}")
        return 1
    # Field names are the ones scripts/hardening_check.sh writes; nothing is invented.
    for field in ("status", "git_commit", "tree_fingerprint", "finished_at"):
        if field not in receipt:
            print(f"gates: receipt {receipt_rel} lacks field {field}")
            return 1
    gate["evidence"] = {
        "commit": receipt["git_commit"],
        "tree_fingerprint": receipt["tree_fingerprint"],
        "date": receipt["finished_at"][:10],
        "receipt": str(receipt_rel),
        "ci_run": (gate["evidence"] or {}).get("ci_run"),
    }
    if receipt["status"] == "passed":
        gate["status"] = "pass"
    else:
        print(
            f"gates: receipt status is {receipt['status']!r} (exit {receipt.get('exit_code')}); "
            f"{gate['id']} stays {gate['status']}"
        )
    save_ledger(args.ledger, ledger)
    args.render_path.write_text(render(root, ledger), encoding="utf-8")
    # Receipts are host-blind before schema_version 2; memory and timing evidence
    # means nothing without the machine, so name it rather than let it pass silently.
    host = receipt.get("host") or {}
    if host:
        memory = host.get("mem_total_kb")
        where = " ".join(
            str(part)
            for part in (
                host.get("hostname"),
                host.get("kernel"),
                f"{host.get('cpus')}cpu",
                f"{round(memory / 1048576, 1)}GiB" if memory else None,
            )
            if part is not None
        )
    else:
        where = "host unrecorded (receipt predates schema_version 2)"
    print(
        f"gates: {gate['id']} <- {args.check} @ {receipt['git_commit'][:7]} "
        f"({gate['status']}) on {where}; rendered"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser(
        "check", help="validate ledger, evidence, freshness, committed render"
    )
    check.add_argument(
        "--release",
        action="store_true",
        help="require every release gate pass, fresh, clean-tree",
    )
    check.add_argument("--ledger", type=pathlib.Path)
    check.add_argument("--tasks-dir", type=pathlib.Path, default=root / "tasks")
    check.add_argument(
        "--render-path", type=pathlib.Path, default=root / "docs/PRE_RELEASE_STATUS.md"
    )
    check.set_defaults(func=cmd_check)

    rend = sub.add_parser("render", help="write the status page")
    rend.add_argument("--ledger", type=pathlib.Path, default=root / "tasks/gates.json")
    rend.add_argument(
        "--out", type=pathlib.Path, default=root / "docs/PRE_RELEASE_STATUS.md"
    )
    rend.set_defaults(func=cmd_render)

    prom = sub.add_parser("promote", help="bind a hardening_check.sh receipt to a gate")
    prom.add_argument("check", metavar="check-name")
    prom.add_argument("gate", metavar="gate-id")
    prom.add_argument("--ledger", type=pathlib.Path, default=root / "tasks/gates.json")
    prom.add_argument(
        "--render-path", type=pathlib.Path, default=root / "docs/PRE_RELEASE_STATUS.md"
    )
    prom.set_defaults(func=cmd_promote)

    args = parser.parse_args(argv)
    if args.command == "check" and args.ledger is None:
        args.ledger = args.tasks_dir / "gates.json"
    try:
        return args.func(args)
    except RuntimeError as exc:
        print(f"gates: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
