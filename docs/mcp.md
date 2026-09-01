# DeepData MCP server

`cmd/deepdata-mcp` is a stdio JSON-RPC 2.0 server speaking MCP protocol version `2025-06-18`
(`cmd/deepdata-mcp/main.go:38`). It forwards every tool call to a running DeepData server over the HTTP
V3 contract; stdlib only, no engine import (main.go:23-35). It arrived with commit bde4f94 (2026-08-21).

## Build and connect

```bash
go build -o deepdata-mcp ./cmd/deepdata-mcp
```

Claude Desktop configuration:

```json
{
  "mcpServers": {
    "deepdata": {
      "command": "./deepdata-mcp",
      "env": {"DEEPDATA_URL": "http://127.0.0.1:8080",
              "DEEPDATA_TENANT": "acme",
              "DEEPDATA_API_KEY": "replace-me"}
    }
  }
}
```

| Variable | Required | Meaning |
|---|---|---|
| `DEEPDATA_URL` | no | Base URL of the DeepData server; default `http://127.0.0.1:8080` (main.go:98-101). HTTP calls time out after 30 seconds (main.go:110) |
| `DEEPDATA_TENANT` | no | The one tenant this process operates on; default `mcp` (main.go:102-105). The tenant is never a tool argument (main.go:114-116) |
| `DEEPDATA_API_KEY` | by the server | Sent as `Authorization: Bearer` when set (main.go:109, :134-136); a server started without `DEEPDATA_INSECURE_DEV_MODE=1` rejects unauthenticated calls |

## Tools

The server advertises exactly these tools in `tools/list` (main.go:228-286):

<!-- generated:mcp-tools -->
- `search`
- `insert`
- `upsert`
- `get_document`
- `list_collections`
<!-- /generated -->

- `search` — required `collection` and `queries` (field name to a dense array or a sparse
  `{indices, values, dim}` object); optional `top_k`, `ef_search`, `filters`, `score_floor`, `fallback`,
  `usage_boost`, `include_vectors`. POST …/search (main.go:301-323).
- `insert` — required `collection`, positive integer `id`, non-empty `vectors`; optional `metadata`.
  POST …/docs with the id in the body (main.go:324-342).
- `upsert` — same arguments as `insert`. PUT …/docs/{id}, id in the path (main.go:343-365).
- `get_document` — required `collection` and `id`. GET …/docs/{id} (main.go:366-379).
- `list_collections` — no arguments. GET /v3/tenants/{tenant}/collections (main.go:380-385), which the
  server authorizes at the `admin` permission (`cmd/deepdata/collection_http.go:425-429`).

Capabilities advertise tools only (main.go:174). There are no resources, prompts, outputSchema or
annotations today; the schemas for `queries`, `filters` and `fallback` are bare objects (main.go:239-246).

## Errors today

- A non-2xx HTTP response becomes a tool result with `isError: true` whose single text part is the server's
  plain-text body, trimmed (main.go:146-148, :156-161; test TestToolErrorsAreIsErrorNotRPCErrors, `cmd/deepdata-mcp/main_test.go:213`).
- Missing or malformed arguments are JSON-RPC error `-32602`; an unknown tool is `-32601`; an unparsable
  frame is `-32700`; a wrong `jsonrpc` version is `-32600` (main.go:186-201, :424-438).

`GOTOOLCHAIN=go1.25.12 go test ./cmd/deepdata-mcp` runs the five tests in `cmd/deepdata-mcp/main_test.go`
(:86-242) against an httptest stand-in. No job in `.github/workflows/ci.yml` includes the package (gate CI-05).

## Planned

Plans, each tracked by its gate in `tasks/gates.json`; none of this exists at HEAD:

- CTL-01 — structured errors: every API error carries code, hint and docs pointer through a new
  internal/apierror package, and the MCP server forwards that envelope.
- CTL-02 — text in, text out: server-side embeddings so agents send texts instead of vectors.
- CTL-03 — rewrite onto a shared api/contract package: six verbs deepdata_recall, deepdata_remember,
  deepdata_forget, deepdata_get, deepdata_collections, deepdata_create_collection, each with outputSchema
  and annotations, plus the resources deepdata://contract and deepdata://status.
- CI-05 — cmd/deepdata-mcp joins the CI package list and its tests run there.
