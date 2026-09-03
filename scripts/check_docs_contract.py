#!/usr/bin/env python3
"""Docs contract linter (stdlib only): prose must agree with code.

  check (default)   report violations as "R<n> <file>:<line>: <message>", exit 1 if any
  --write           regenerate <!-- generated:<name> --> blocks in place (existing markers only)
  --facts           print the facts the rules compare against, as JSON
  --selftest        exercise rule helpers on synthetic input

Rules: R1 gRPC list/count, R2 mutation count, R3 HTTP route table, R4 known-false phrases,
R5 dead relative links, R6 case collisions, R7 phantom backticked file refs, R8 non-goals
claimed live, R9 dashboard metrics, R10 MCP tool list, R11 non-goals block, R12 checkboxes,
R13 fences that do not parse, R14 documented SDK calls the SDK does not have,
R15 documented curl calls the v3 route table does not have.
"""

from __future__ import annotations

import ast
import json
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKED_GLOBS = [
    "README.md",
    "CHANGELOG.md",
    "SECURITY.md",
    "api/contract/v3/CONTRACT.md",
    "docs/**/*.md",
    "internal/collection/API.md",
    "sdk/python/README.md",
    "benchmarks/*.md",  # README.md plus GAP_ANALYSIS.md (spec named only README.md; see notes)
    "tasks/todo.md",
    "tasks/PROTOCOL.md",
]
EXCLUDED_PREFIXES = (
    "tasks/journal/",
    "docs/decisions/",
    "benchmarks/results/",
    "benchmarks/competitive/live/",
)
CHECKBOX_FILES = ("tasks/todo.md", "tasks/PROTOCOL.md", "tasks/lessons.md")
ARCH = "docs/ARCHITECTURE.md"
EXPECTED_BLOCKS = {  # block name -> files that must carry it
    "grpc-rpcs": ("README.md", "internal/collection/API.md"),
    "http-routes": ("internal/collection/API.md",),
    "mcp-tools": ("docs/mcp.md",),
    "non-goals": ("README.md",),
}
GEN_START = re.compile(r"^\s*<!--\s*generated:([a-z0-9-]+)\s*-->\s*$")
GEN_END = re.compile(r"^\s*<!--\s*/generated\s*-->\s*$")
NEGATION = re.compile(
    r"\b(not|no|non-goal|never|out of scope|rejected|unsupported|without|deferred|removed|retired|"
    r"disabled|does not|isn't|is not|non-goals|unavailable|outside|non-rc|future)\b",
    re.I,
)
NON_GOAL_OK = "<!-- non-goal-ok -->"
NUMBER_WORDS = {
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
}
R1_RPC = re.compile(r"\b(gRPC|RPC)s?\b")
R1_NUM = re.compile(r"(?<![\w.:/+-])(nine|ten|eleven|twelve|\d{1,2})(?![\w.:/%+-])")
R2_MUT = re.compile(r"\bmutations?\b", re.I)
R2_NUM = re.compile(r"\b(four|five|six|seven)\b", re.I)
R4_PHRASES = [
    r"/metrics.{0,40}unauthenticated",
    r"does not yet have a license",
    r"(same|exactly) nine",
    r"V3 has no upsert",
    r"title-case",
    r"Playwright",
    r"group-commit.{0,60}(top|first|primary) lever",
    r"[Ee]rrors are (unstructured )?plain text",
]
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
BACKTICK = re.compile(r"`([^`\n]+)`")
KNOWN_EXT = (
    ".go",
    ".py",
    ".md",
    ".sh",
    ".json",
    ".yaml",
    ".yml",
    ".proto",
    ".toml",
    ".txt",
    ".html",
)
BAD_CHARS = set("*?[]{}|\\<>$\"' \t=@")
METRIC_NAME = re.compile(r"\b(?:vectordb|deepdata)_[a-z_]+\b")
MCP_TOOL = re.compile(r'^\s*Name:\s+"([a-z_]+)"', re.M)
ROUTES_CMD = ["env", "GOTOOLCHAIN=go1.25.12", "go", "run", "./cmd/deepdata", "routes"]
FENCE = re.compile(r"^\s*(?:```|~~~)\s*(\w*)")
JSON_ELIDED = re.compile(r"^\s*(?:\.\.\.|//|/\*)", re.M)
SDK_PKG = "sdk/python/deepdata"
# Judged per exact class, never unioned: sync and async are separate promises, and a rename
# in one of them is invisible if the other still carries the old name.
TENANT_OF = {"DeepDataClient": "TenantClient", "AsyncDeepDataClient": "AsyncTenantClient"}
SHELL_LANGS = ("bash", "sh", "shell", "console")
# A curl target is an absolute URL, a host:port, or a /v3 path standing on its own -- never
# a relative path, so `-proto deepdata/v3/deepdata.proto` is not read as a call.
CURL_TARGET = re.compile(
    r"(?:https?://[^\s\"'`\\]+|[\w.-]+:\d+/[^\s\"'`\\]*|(?<![\w./-])/v3/[^\s\"'`\\]*)"
)
CURL_CALL = re.compile(r"\bcurl\b")  # not grpcurl, which speaks proto, not routes
CURL_VERB = re.compile(r"-X\s+([A-Z]+)")


def rel(path: pathlib.Path) -> str:
    return path.relative_to(ROOT).as_posix()


def git_files() -> list[str]:
    out = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line for line in out.splitlines() if line]


def checked_files() -> list[pathlib.Path]:
    seen: dict[str, pathlib.Path] = {}
    for pattern in CHECKED_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            r = rel(path)
            if path.is_file() and not r.startswith(EXCLUDED_PREFIXES):
                seen[r] = path
    return [seen[k] for k in sorted(seen)]


def read_lines(path: pathlib.Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def fenced_mask(lines: list[str]) -> list[bool]:
    """True for lines inside ``` fences (fence lines included)."""
    mask, inside = [], False
    for line in lines:
        if re.match(r"^\s*(```|~~~)", line):
            inside = not inside
            mask.append(True)
        else:
            mask.append(inside)
    return mask


def fences(lines: list[str]):
    """Yield (lang, 1-based line of the opening fence, body) for each fenced block."""
    lang, start, buf = None, 0, []
    for i, line in enumerate(lines, 1):
        m = FENCE.match(line)
        if m is None:
            if lang is not None:
                buf.append(line)
        elif lang is None:
            lang, start, buf = m.group(1) or "none", i, []
        else:
            yield lang, start, "\n".join(buf)
            lang = None


def generated_blocks(lines: list[str]) -> dict[str, tuple[int, int]]:
    """name -> (start index of marker line, end index of closing marker), 0-based."""
    blocks, name, start = {}, None, 0
    for i, line in enumerate(lines):
        m = GEN_START.match(line)
        if m:
            name, start = m.group(1), i
        elif GEN_END.match(line) and name:
            blocks[name] = (start, i)
            name = None
    return blocks


def in_generated(blocks: dict[str, tuple[int, int]], i: int) -> bool:
    return any(s <= i <= e for s, e in blocks.values())


def marker_block(lines: list[str], tag: str) -> tuple[int, int] | None:
    """Range between <!-- tag --> and <!-- /tag --> (0-based marker indices)."""
    start = end = None
    for i, line in enumerate(lines):
        if line.strip() == f"<!-- {tag} -->":
            start = i
        elif line.strip() == f"<!-- /{tag} -->" and start is not None:
            end = i
            break
    return (start, end) if start is not None and end is not None else None


# ---------------------------------------------------------------- facts


def collect_facts() -> dict:
    proto = (ROOT / "api/proto/deepdata/v3/deepdata.proto").read_text(encoding="utf-8")
    service = proto[proto.index("service DeepData") :]
    service = service[: service.index("}")]
    rpcs = re.findall(r"^\s*rpc\s+(\w+)\s*\(", service, re.M)
    mutating = [
        r for r in rpcs if re.match(r"^(Create|Delete|Insert|BatchInsert|Upsert)", r)
    ]
    mcp_src = ROOT / "cmd/deepdata-mcp/main.go"
    mcp_tools = (
        MCP_TOOL.findall(mcp_src.read_text(encoding="utf-8"))
        if mcp_src.exists()
        else []
    )
    metrics_src = ROOT / "cmd/deepdata/metrics.go"
    metrics_names = re.findall(
        r'Name:\s*"((?:vectordb|deepdata)_[a-z_]+)"',
        metrics_src.read_text(encoding="utf-8"),
    )
    dashboard: dict[str, tuple[str, int]] = {}
    for js in sorted((ROOT / "docs/grafana").glob("*.json")):
        for i, line in enumerate(read_lines(js), 1):
            for name in METRIC_NAME.findall(line):
                base = re.sub(r"_(bucket|sum|count)$", "", name)
                dashboard.setdefault(base, (rel(js), i))
    return {
        "rpcs": rpcs,
        "mutating_rpcs": mutating,
        "mcp_tools": mcp_tools,
        "non_goals": non_goals(),
        "metrics_names": metrics_names,
        "dashboard_metric_names": {k: list(v) for k, v in sorted(dashboard.items())},
        "version": (ROOT / "internal/releaseinfo/version.txt")
        .read_text(encoding="utf-8")
        .strip(),
    }


def non_goals() -> list[dict]:
    path = ROOT / ARCH
    if not path.exists():
        return []
    lines = read_lines(path)
    span = marker_block(lines, "non-goals")
    if not span:
        return []
    out = []
    for line in lines[span[0] + 1 : span[1]]:
        if "|" in line:
            label, regex = line.split("|", 1)
            out.append({"label": label.strip(), "regex": regex.strip()})
    return out


def sdk_classes() -> dict[str, dict[str, frozenset | None]] | None:
    """class -> method -> its parameter names. None params means the method takes **kwargs and
    its call sites cannot be judged. None overall if the SDK is gone."""
    pkg = ROOT / SDK_PKG
    if not pkg.is_dir():
        return None
    found: dict[str, dict[str, frozenset | None]] = {}
    for py in sorted(pkg.rglob("*.py")):
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ClassDef):
                continue
            methods: dict[str, frozenset | None] = {}
            for b in node.body:
                if not isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if b.name.startswith("_"):
                    continue
                a = b.args
                methods[b.name] = (
                    None
                    if a.kwarg is not None
                    else frozenset(
                        x.arg
                        for x in a.posonlyargs + a.args + a.kwonlyargs
                        if x.arg != "self"
                    )
                )
            found[node.name] = methods
    needed = set(TENANT_OF) | set(TENANT_OF.values())
    return found if needed <= set(found) else None


def emitted_metric_names() -> set[str]:
    """ponytail: regex call-graph over cmd/deepdata; a metric counts as emitted only when a
    reachable metrics.go function calls WithLabelValues/Inc/Set/Observe/Add on its field."""
    src = (ROOT / "cmd/deepdata/metrics.go").read_text(encoding="utf-8")
    field_to_name = dict(
        re.findall(
            r"(\w+):\s*prometheus\.New\w+\(\s*prometheus\.\w+Opts\{\s*Name:\s*\"([^\"]+)\"",
            src,
        )
    )
    funcs: dict[str, str] = {}
    for m in re.finditer(
        r"^func\s+(?:\([^)]*\)\s*)?(\w+)\s*\(.*?(?=^func\s|\Z)", src, re.M | re.S
    ):
        funcs[m.group(1)] = m.group(0)
    others = ""
    for go in (ROOT / "cmd/deepdata").glob("*.go"):
        if go.name != "metrics.go" and not go.name.endswith("_test.go"):
            others += go.read_text(encoding="utf-8")
    live = {name for name in funcs if re.search(rf"\b{name}\s*\(", others)}
    changed = True
    while changed:
        changed = False
        for name in list(live):
            for callee in funcs:
                if callee not in live and re.search(rf"\b{callee}\s*\(", funcs[name]):
                    live.add(callee)
                    changed = True
    emitted = set()
    for name in live:
        for field in re.findall(
            r"mc\.(\w+)\.(?:WithLabelValues|Inc|Set|Observe|Add)\(", funcs[name]
        ):
            if field in field_to_name:
                emitted.add(field_to_name[field])
    return emitted


# ---------------------------------------------------------------- generated block text


def routes_tsv() -> list[list[str]] | None:
    has_cmd = any(
        re.search(r'"routes"', go.read_text(encoding="utf-8", errors="replace"))
        for go in (ROOT / "cmd/deepdata").glob("*.go")
        if not go.name.endswith("_test.go")
    )
    if not has_cmd:
        return None
    out = subprocess.run(
        ROUTES_CMD, cwd=ROOT, capture_output=True, text=True, timeout=300
    )
    if out.returncode != 0:
        return None
    return [line.split("\t") for line in out.stdout.splitlines() if line.strip()]


def block_text(
    name: str, facts: dict, routes: list[list[str]] | None
) -> list[str] | None:
    if name == "grpc-rpcs":
        return [
            f"`deepdata.v3.DeepData` exposes {len(facts['rpcs'])} unary RPCs: "
            + ", ".join(f"`{r}`" for r in facts["rpcs"])
            + "."
        ]
    if name == "mcp-tools":
        return [f"- `{t}`" for t in facts["mcp_tools"]]
    if name == "non-goals":
        return [f"- {g['label']}" for g in facts["non_goals"]]
    if name == "http-routes":
        if routes is None:
            return None
        header = routes[0]
        body = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        body += ["| " + " | ".join(row) + " |" for row in routes[1:]]
        return body
    return None


# ---------------------------------------------------------------- rules


class Report:
    def __init__(self) -> None:
        self.items: list[tuple[int, str, int, str]] = []

    def add(self, rule: int, file: str, line: int, message: str) -> None:
        self.items.append((rule, file, line, message))


def check_blocks(
    rep: Report, docs: dict[str, list[str]], facts: dict, routes, r3_skip: bool
) -> None:
    for name, files in EXPECTED_BLOCKS.items():
        rule = {"grpc-rpcs": 1, "http-routes": 3, "mcp-tools": 10, "non-goals": 11}[
            name
        ]
        if name == "http-routes" and r3_skip:
            continue
        expected = block_text(name, facts, routes)
        for file in files:
            if file not in docs:
                rep.add(
                    rule, file, 1, f"file missing; expected a generated:{name} block"
                )
                continue
            blocks = generated_blocks(docs[file])
            if name not in blocks:
                rep.add(rule, file, 1, f"missing generated:{name} block")
                continue
            s, e = blocks[name]
            current = [l.rstrip() for l in docs[file][s + 1 : e]]
            if expected is not None and current != expected:
                rep.add(
                    rule,
                    file,
                    s + 1,
                    f"generated:{name} block out of date (run --write)",
                )


def check_prose(
    rep: Report,
    file: str,
    lines: list[str],
    facts: dict,
    goals: list[tuple[str, re.Pattern]],
) -> None:
    fenced = fenced_mask(lines)
    blocks = generated_blocks(lines)
    arch_span = marker_block(lines, "non-goals") if file == ARCH else None
    tracked = TRACKED
    for i, line in enumerate(lines):
        n = i + 1
        if in_generated(blocks, i):
            continue
        for phrase in R4_PHRASES:  # R4 applies everywhere, code included
            if re.search(phrase, line, re.I):
                rep.add(4, file, n, f"known-false phrase: /{phrase}/")
        if arch_span and arch_span[0] <= i <= arch_span[1]:
            continue
        if fenced[i]:
            continue
        prose = BACKTICK.sub(lambda m: " " * len(m.group(0)), line)
        for m in R1_RPC.finditer(prose):
            window = prose[max(0, m.start() - 40) : m.end() + 40]
            for num in R1_NUM.finditer(window):
                value = NUMBER_WORDS.get(num.group(1).lower(), None)
                value = (
                    int(num.group(1))
                    if value is None and num.group(1).isdigit()
                    else value
                )
                if value is not None and value != len(facts["rpcs"]):
                    rep.add(
                        1,
                        file,
                        n,
                        f"says {num.group(1)} RPCs; proto has {len(facts['rpcs'])}",
                    )
                    break
        for m in R2_MUT.finditer(prose):
            window = prose[max(0, m.start() - 40) : m.end() + 40]
            for num in R2_NUM.finditer(window):
                if NUMBER_WORDS[num.group(1).lower()] != len(facts["mutating_rpcs"]):
                    rep.add(
                        2,
                        file,
                        n,
                        f"says {num.group(1)} mutations; proto has {len(facts['mutating_rpcs'])}",
                    )
                    break
        for m in LINK.finditer(line):
            target = m.group(1).split("#", 1)[0]
            if not target or re.match(r"^(https?:|mailto:|/)", target):
                continue
            if not (ROOT / pathlib.Path(file).parent / target).exists():
                rep.add(5, file, n, f"dead link: {m.group(1)}")
        for token in BACKTICK.findall(line):
            if not exists_in_tree(token, file, tracked):
                rep.add(7, file, n, f"backticked path does not exist: `{token}`")
        if file != "docs/security-patterns.md" and not negated(lines, i):
            hits = [label for label, rx in goals if rx.search(line)]
            if hits:
                rep.add(8, file, n, "non-goal claimed live: " + "; ".join(hits))


def call_kind(func: ast.expr, kind: dict[str, str]) -> str | None:
    """The SDK class this call returns, or None when the receiver is not ours."""
    name = getattr(func, "id", None) or getattr(func, "attr", None)
    if name in TENANT_OF or name in TENANT_OF.values():
        return name
    # Only <client>.tenant(...); a bare tenant(...) could be anyone's helper.
    if (
        name == "tenant"
        and isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
    ):
        return TENANT_OF.get(kind.get(func.value.id, ""))
    return None


def check_sdk_calls(rep: Report, file: str, start: int, tree: ast.AST, sdk: dict) -> None:
    """R14: only receivers traceable to an SDK constructor are judged, so a fence that also
    drives chromadb/pinecone/stdlib is left alone instead of reported as drift."""
    kind: dict[str, str] = {}
    for node in ast.walk(tree):
        target = value = None
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            target, value = node.targets[0].id, node.value
        elif isinstance(node, ast.withitem) and isinstance(node.optional_vars, ast.Name):
            target, value = node.optional_vars.id, node.context_expr
        if target and isinstance(value, ast.Call):
            if cls := call_kind(value.func, kind):
                kind[target] = cls
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        recv = node.func.value
        if isinstance(recv, ast.Name):
            cls = kind.get(recv.id)
        elif isinstance(recv, ast.Call):
            cls = call_kind(recv.func, kind)
        else:
            cls = None
        if not cls:
            continue
        methods = sdk[cls]
        if node.func.attr not in methods:
            rep.add(
                14,
                file,
                start,
                f"{cls} has no method .{node.func.attr}(); it has "
                + ", ".join(sorted(methods)),
            )
            continue
        allowed = methods[node.func.attr]
        # A method taking **kwargs accepts anything, and a call splatting **mapping hides its
        # own keys, so neither can be judged.
        if allowed is None or any(k.arg is None for k in node.keywords):
            continue
        for kw in node.keywords:
            if kw.arg not in allowed:
                rep.add(
                    14,
                    file,
                    start,
                    f"{cls}.{node.func.attr}() has no parameter {kw.arg}=; it takes "
                    + ", ".join(sorted(allowed)),
                )


def route_matcher(routes: list[list[str]] | None):
    """(method, path) -> True when the v3 route table has it. A templated segment matches any
    single value, which is exactly what a doc example fills in."""
    if routes is None:
        return None
    table = [(row[0], row[1].split("/")) for row in routes[1:] if len(row) >= 2]

    def known(method: str, path: str) -> bool:
        parts = path.split("/")
        return any(
            m == method
            and len(pat) == len(parts)
            and all(p.startswith("{") or p == q for p, q in zip(pat, parts))
            for m, pat in table
        )

    return known


def check_curl(rep: Report, file: str, lines: list[str], known) -> None:
    """R15: /v3 is the only surface with a generated route table, so it is the only surface a
    curl example can be judged against; /livez, /metrics and clone URLs are left alone."""
    for lang, start, body in fences(lines):
        if lang not in SHELL_LANGS:
            continue
        for offset, line in enumerate(body.replace("\\\n", " ").splitlines()):
            if not CURL_CALL.search(line):
                continue
            verb = CURL_VERB.search(line)
            method = verb.group(1) if verb else "GET"
            for target in CURL_TARGET.findall(line):
                path = re.sub(r"^(?:https?://)?[^/]*", "", target).split("?")[0]
                if path.startswith("/v3/") and not known(method, path):
                    rep.add(15, file, start + offset + 1, f"no v3 route for {method} {path}")


def check_fences(rep: Report, file: str, lines: list[str], sdk: dict | None) -> None:
    """R13: the prose rules mask fences out, so a copy-paste example is the one claim nothing
    reads. Parse them, then judge the SDK calls inside the ones that parsed."""
    for lang, start, body in fences(lines):
        if lang in ("python", "py"):
            try:
                tree = ast.parse(body)
            except SyntaxError as e:
                rep.add(13, file, start + (e.lineno or 0), f"python fence: {e.msg}")
                continue
            if sdk:
                check_sdk_calls(rep, file, start, tree, sdk)
        elif lang == "json" and not JSON_ELIDED.search(body):
            try:
                json.loads(body)
            except ValueError as e:
                rep.add(13, file, start, f"json fence: {e}")


LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+\.)\s")


def negated(lines: list[str], i: int) -> bool:
    """R8 context: a table row stands alone; a list item spans its continuation lines and
    inherits a lead-in paragraph ending with ':' or a heading; prose spans its paragraph."""
    if NON_GOAL_OK in lines[i]:
        return True
    if lines[i].lstrip().startswith("|"):
        return bool(NEGATION.search(lines[i]))
    start = i
    while start > 0 and lines[start - 1].strip() and not LIST_ITEM.match(lines[start]):
        start -= 1
    end = i
    while end + 1 < len(lines) and lines[end + 1].strip() and not LIST_ITEM.match(lines[end + 1]):
        end += 1
    text = "\n".join(lines[start : end + 1])
    if LIST_ITEM.match(lines[start]):
        lead = start - 1
        while lead >= 0 and (not lines[lead].strip() or LIST_ITEM.match(lines[lead]) or lines[lead][:1].isspace()):
            lead -= 1
        if lead >= 0 and (lines[lead].rstrip().endswith(":") or lines[lead].startswith("#")):
            top = lead
            while top > 0 and lines[top - 1].strip():
                top -= 1
            text += "\n" + "\n".join(lines[top : lead + 1])
    return bool(NEGATION.search(text))


def path_like(token: str) -> str | None:
    """Return the normalized path a backticked token refers to, or None if it is not a file ref."""
    if any(c in BAD_CHARS for c in token) or token.startswith(
        ("http", "/", "~", "-", "#", "./..")
    ):
        return None
    token = re.sub(r":\d+(?:-\d+)?$", "", token).rstrip(".,;:)")
    if ":" in token or not token or token.endswith("..."):
        return None
    token = token[2:] if token.startswith("./") else token
    if "/" not in token and not token.endswith(KNOWN_EXT):
        return None
    if not re.match(r"^[\w.][\w./+-]*$", token):
        return None
    if not token.endswith(KNOWN_EXT) and not token.endswith("/"):
        if token.split("/", 1)[0] not in TOP_LEVEL:
            return None  # `bytes/vec`-style units, not repo paths
    return token


def exists_in_tree(token: str, file: str, tracked: set[str]) -> bool:
    path = path_like(token)
    if path is None:
        return True
    directory = path.endswith("/")
    path = path.rstrip("/")
    candidates = {path, (pathlib.Path(file).parent / path).as_posix().lstrip("./")}
    for c in candidates:
        if c in tracked:
            return True
        if directory or "/" in c:
            if any(t.startswith(c + "/") for t in tracked):
                return True
    if "/" not in path:
        return any(t.rsplit("/", 1)[-1] == path for t in tracked)
    return any(t.endswith("/" + path) for t in tracked)


def check_case_collisions(rep: Report, paths: list[str]) -> None:
    groups: dict[str, set[str]] = {}
    for p in paths:
        groups.setdefault(p.lower(), set()).add(p)
    for group in groups.values():
        if len(group) > 1:
            first, *rest = sorted(group)
            rep.add(6, first, 1, "case collision with " + ", ".join(rest))


def check_dashboard(rep: Report, facts: dict, emitted: set[str]) -> None:
    declared = set(facts["metrics_names"])
    for name, (file, line) in facts["dashboard_metric_names"].items():
        if name not in declared:
            rep.add(
                9,
                file,
                line,
                f"panel metric {name} is not declared in cmd/deepdata/metrics.go",
            )
        elif name not in emitted:
            rep.add(
                9,
                file,
                line,
                f"panel metric {name} is declared but never emitted from reachable code",
            )


def check_checkboxes(rep: Report) -> None:
    for file in CHECKBOX_FILES:
        path = ROOT / file
        if not path.exists():
            continue
        for i, line in enumerate(read_lines(path), 1):
            if re.match(r"^\s*- \[[ xX]\]", line):
                rep.add(12, file, i, "checkbox; status lives in tasks/gates.json")


TRACKED: set[str] = set()
TOP_LEVEL: set[str] = set()


def run_check(write: bool) -> int:
    global TRACKED, TOP_LEVEL
    paths = git_files()
    TRACKED = set(paths)
    TOP_LEVEL = {p.split("/", 1)[0] for p in paths}
    facts = collect_facts()
    docs = {rel(p): read_lines(p) for p in checked_files()}
    routes = routes_tsv()
    r3_skip = routes is None
    rep = Report()
    if r3_skip:
        print("R3 SKIP: routes subcommand unavailable (gate DOC-03)")
        if os.environ.get("DOC_ALLOW_R3_SKIP") != "1":
            rep.add(
                3,
                "internal/collection/API.md",
                1,
                "routes subcommand unavailable; set DOC_ALLOW_R3_SKIP=1 to accept",
            )
    if write:
        for file, lines in docs.items():
            blocks = generated_blocks(lines)
            out, changed = list(lines), False
            for name in sorted(blocks, key=lambda k: -blocks[k][0]):
                text = block_text(name, facts, routes)
                if text is not None:
                    s, e = blocks[name]
                    out[s + 1 : e] = text
                    changed = True
            if changed:
                (ROOT / file).write_text("\n".join(out) + "\n", encoding="utf-8")
                docs[file] = out
                print(f"wrote {file}")
    if not facts["non_goals"]:
        rep.add(8, ARCH, 1, "missing <!-- non-goals --> block (label | regex per line)")
    goals = [(g["label"], re.compile(g["regex"], re.I)) for g in facts["non_goals"]]
    check_blocks(rep, docs, facts, routes, r3_skip)
    sdk = sdk_classes()
    known_route = route_matcher(routes)
    for file, lines in docs.items():
        check_prose(rep, file, lines, facts, goals)
        check_fences(rep, file, lines, sdk)
        if known_route:
            check_curl(rep, file, lines, known_route)
    check_case_collisions(rep, paths)
    check_dashboard(rep, facts, emitted_metric_names())
    check_checkboxes(rep)
    rep.items.sort()
    for rule, file, line, message in rep.items:
        print(f"R{rule} {file}:{line}: {message}")
    rules = sorted({r for r, *_ in rep.items})
    print(
        f"docs-contract: {len(rep.items)} violations (rules: {', '.join(f'R{r}' for r in rules) or 'none'})"
    )
    return 1 if rep.items else 0


def selftest() -> int:
    rep = Report()
    check_case_collisions(
        rep, ["docs/BENCHMARKS.md", "docs/benchmarks.md", "README.md"]
    )
    assert [i[0] for i in rep.items] == [6], rep.items
    check_case_collisions(rep, ["a.md", "b.md"])
    assert len(rep.items) == 1
    tracked = {
        "scripts/gates.py",
        "docs/cookbook.md",
        "api/proto/deepdata/v3/deepdata.proto",
        "cmd/deepdata/main.go",
    }
    assert exists_in_tree("scripts/gates.py:12", "README.md", tracked)
    assert exists_in_tree("./cmd/deepdata", "README.md", tracked)
    assert exists_in_tree("deepdata.proto", "README.md", tracked)
    assert exists_in_tree("v3/deepdata.proto", "README.md", tracked)
    assert exists_in_tree("cookbook.md", "docs/security.md", tracked)
    assert exists_in_tree("go test ./...", "README.md", tracked)  # not a path token
    assert exists_in_tree("localhost:8080/health", "README.md", tracked)
    assert not exists_in_tree(
        ".deepdata-run/checks/go-race/receipt.json", "README.md", tracked
    )
    assert not exists_in_tree(
        "benchmarks/competitive/run_comparison.py", "README.md", tracked
    )
    assert not exists_in_tree("missing.md", "README.md", tracked)
    facts = {"rpcs": ["x"] * 11, "mutating_rpcs": ["x"] * 6}
    rep = Report()
    goals = [("cluster", re.compile(r"\bclusters?\b", re.I))]
    lines = [
        "the same nine unary gRPC operations",  # 1: R1 + R4
        "",
        "gRPC on `:50051` and HTTP/h2c",
        "V3 HTTP plus the unary gRPC service",
        "",
        "five canonical mutations",  # 6: R2
        "six mutations",
        "",
        "runs a cluster",  # 9: R8
        "",
        "does not run a cluster",
        "",
        f"a cluster {NON_GOAL_OK}",
        "",
        "The RC does not include:",
        "",
        "- a cluster",
        "  or a second cluster",
        "",
        "## Not a good fit",
        "",
        "- a cluster",
        "",
        "| cluster | Yes |",  # 24: R8 (table rows stand alone)
        "| PQ | No |",
        "",
        "Remnants exist for the cluster.",
        "They are not reachable.",
        "",
        "```",
        "nine RPCs inside a fence",
        "```",
    ]
    check_prose(rep, "x.md", lines, facts, goals)
    got = sorted((r, l) for r, _, l, _ in rep.items)
    assert got == [(1, 1), (2, 6), (4, 1), (8, 9), (8, 24)], got
    sdk = {
        "DeepDataClient": {"tenant": frozenset({"name"}), "close": frozenset()},
        "AsyncDeepDataClient": {"tenant": frozenset({"name"}), "close": frozenset()},
        "TenantClient": {
            "search": frozenset({"collection", "top_k"}),
            "insert": None,  # takes **kwargs
        },
        "AsyncTenantClient": {"insert": None},
    }
    rep = Report()
    lines = [
        "```python",
        "def broken(:",  # 2: R13
        "```",
        "```json",
        '{"a": 1,}',  # 5: R13
        "```",
        "```json",
        "{",  # elided -- not a claim about a whole document
        '  "a": 1,',
        "  ...",
        "}",
        "```",
        "```python",
        "client = DeepDataClient()",
        "client.query('x')",  # R14, reported at the fence (13)
        "client.tenant('t').search(collection='d', top_k=5)",
        "```",
        "```python",
        "a = AsyncDeepDataClient()",
        "a.tenant('t').search(collection='d')",  # R14: sync-only method, async class (18)
        "```",
        "```python",
        "import chromadb",
        "src = chromadb.PersistentClient()",
        "src.get_collection('x').get()",  # foreign receiver, silent
        "```",
        "```text",
        "def also_broken(:",  # not python, silent
        "```",
    ]
    check_fences(rep, "x.md", lines, sdk)
    got = sorted((r, l) for r, _, l, _ in rep.items)
    assert got == [(13, 2), (13, 4), (14, 13), (14, 18)], got

    rep = Report()
    lines = [
        "```python",
        "c = DeepDataClient()",
        "c.tenant('t').search(collection='d', no_such=1)",  # R14: bad kwarg
        "c.tenant('t').search(collection='d', top_k=5)",
        "c.tenant('t').insert(anything=1)",  # **kwargs method, silent
        "c.tenant('t').search(**opts)",  # splatted, silent
        "```",
    ]
    check_fences(rep, "x.md", lines, sdk)
    assert [(r, l, m) for r, _, l, m in rep.items] == [
        (14, 1, "TenantClient.search() has no parameter no_such=; it takes collection, top_k")
    ], rep.items

    known = route_matcher(
        [
            ["method", "path"],
            ["GET", "/v3/tenants/{tenant}/collections"],
            ["POST", "/v3/tenants/{tenant}/collections/{collection}/search"],
        ]
    )
    assert known("GET", "/v3/tenants/acme/collections")
    assert not known("POST", "/v3/tenants/acme/collections")  # verb the route does not serve
    assert not known("GET", "/v3/tenants/acme/collections/x")  # one segment too deep
    rep = Report()
    lines = [
        "```bash",
        "curl http://localhost:8080/v3/tenants/acme/collections",
        "curl -X POST localhost:8080/v3/nope",  # 3: R15
        "curl -X POST localhost:8080/v3/tenants/acme/collections",  # 4: R15, wrong verb
        "curl localhost:8080/livez",  # not v3, silent
        "grpcurl -proto deepdata/v3/deepdata.proto :50051 x/Y",  # not curl, silent
        "curl -o o.json localhost:8080/v3/status && cat api/contract/v3/x.json",  # 7: R15
        "curl \\",
        "  http://localhost:8080/v3/tenants/a/collections?limit=2",  # continued, silent
        "```",
        "```text",
        "curl localhost:8080/v3/nope",  # not a shell fence, silent
        "```",
    ]
    check_curl(rep, "x.md", lines, known)
    assert sorted((r, l) for r, _, l, _ in rep.items) == [(15, 3), (15, 4), (15, 7)], rep.items

    global TOP_LEVEL
    TOP_LEVEL = {"scripts", "docs", "api", "cmd"}
    assert exists_in_tree("bytes/vec", "README.md", tracked)
    TOP_LEVEL.add("internal")
    assert not exists_in_tree("internal/apierror", "README.md", tracked | {"internal/x.go"})
    print("selftest ok")
    return 0


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        return selftest()
    if "--facts" in argv:
        print(json.dumps(collect_facts(), indent=2))
        return 0
    return run_check(write="--write" in argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
