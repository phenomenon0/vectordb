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
| `DEEPDATA_URL` | no | Base URL of the DeepData server; default `http://127.0.0.1:8080` (main.go:117-120). HTTP calls time out after 30 seconds (main.go:129) |
| `DEEPDATA_TENANT` | no | The one tenant this process operates on; default `mcp` (main.go:121-124). The tenant is never a tool argument (main.go:133-135) |
| `DEEPDATA_API_KEY` | by the server | Sent as `Authorization: Bearer` when set (main.go:128, :154); a server started without `DEEPDATA_INSECURE_DEV_MODE=1` rejects unauthenticated calls |

## Tools

The server advertises exactly these tools in `tools/list` (main.go:257):

<!-- generated:mcp-tools -->
- `search`
- `insert`
- `upsert`
- `get_document`
- `list_collections`
<!-- /generated -->

- `search` — required `collection` and at least one of `queries` (field name to a dense array or a sparse
  `{indices, values, dim}` object) and `texts` (field name to query text, for fields that bind an embedding; a
  field appears in only one of the two); optional `top_k`, `ef_search`, `filters`, `score_floor`, `fallback`,
  `usage_boost`, `include_vectors`. POST …/search (main.go:341); the result carries the server's `embedded_by`.
- `insert` — required `collection`, positive integer `id`, and `vectors` or `texts` (same rule); optional
  `metadata`. POST …/docs with the id in the body (main.go:363).
- `upsert` — same arguments as `insert`. PUT …/docs/{id}, id in the path (main.go:382).
- `get_document` — required `collection` and `id`. GET …/docs/{id} (main.go:401).
- `list_collections` — no arguments. GET /v3/tenants/{tenant}/collections (main.go:415), which the
  server authorizes at the `admin` permission (`cmd/deepdata/collection_http.go:425-429`).

Capabilities advertise tools only (main.go:203). There are no resources, prompts, outputSchema or
annotations today; the schemas for `queries`, `filters` and `fallback` are bare objects (main.go:268-275).

## Errors

- A non-2xx HTTP response becomes a tool result with `isError: true`. When the body is the server's JSON
  envelope ([API.md#errors](../internal/collection/API.md#errors)) the text part reads
  `<code>: <message> Hint: <hint>` and `structuredContent` carries the envelope unchanged, so a host reads
  `code`, `retryable` and `request_id` without parsing prose (main.go:93-105, :165-172, :180-190; test
  TestToolErrorsAreIsErrorNotRPCErrors, `cmd/deepdata-mcp/main_test.go:213`). A non-JSON body is passed
  through trimmed as the text part, with no `structuredContent`.
- Missing or malformed arguments are JSON-RPC error `-32602`; an unknown tool is `-32601`; an unparsable
  frame is `-32700`; a wrong `jsonrpc` version is `-32600` (main.go:216-230, :457-465).

`GOTOOLCHAIN=go1.25.12 go test ./cmd/deepdata-mcp` runs the six tests in `cmd/deepdata-mcp/main_test.go`
against an httptest stand-in. No job in `.github/workflows/ci.yml` includes the package (gate CI-05).

## Planned

Plans, each tracked by its gate in `tasks/gates.json`; none of this exists at HEAD:

- CTL-03 — rewrite onto a shared api/contract package: six verbs deepdata_recall, deepdata_remember,
  deepdata_forget, deepdata_get, deepdata_collections, deepdata_create_collection, each with outputSchema
  and annotations, plus the resources deepdata://contract and deepdata://status.
- CI-05 — cmd/deepdata-mcp joins the CI package list and its tests run there.
