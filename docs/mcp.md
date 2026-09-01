# DeepData MCP server

`cmd/deepdata-mcp` is a stdio JSON-RPC 2.0 server speaking MCP protocol version `2025-06-18`
(`cmd/deepdata-mcp/main.go:38`). It forwards six memory verbs to a running DeepData server over the HTTP
V3 contract; stdlib plus the dependency-free `api/contract` package, no engine import. The verbs' argument
and result shapes are the JSON Schema files under `api/contract/v3/schemas/` (one `{"input", "output"}` file
per verb plus `error.json`), embedded by `api/contract/contract.go` and served verbatim as `inputSchema` and
`outputSchema`; the agent-facing text is `api/contract/v3/CONTRACT.md`, served as the `deepdata://contract`
resource. `cmd/deepdata-mcp/main_test.go` asserts the tool set equals the schema set and every schema
byte-matches its file (TestToolSchemasAreTheContractFiles), so the contract cannot drift from what agents see.

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
              "DEEPDATA_COLLECTION": "memory",
              "DEEPDATA_API_KEY": "replace-me"}
    }
  }
}
```

| Variable | Required | Meaning |
|---|---|---|
| `DEEPDATA_URL` | no | Base URL of the DeepData server; default `http://127.0.0.1:8080` (main.go:213). HTTP calls time out after 30 seconds (main.go:217) |
| `DEEPDATA_TENANT` | no | The one tenant this process operates on; default `mcp` (main.go:214). The tenant is never a tool argument; passing one is rejected as an unknown field |
| `DEEPDATA_COLLECTION` | no | Collection every verb uses when the call omits `collection`; default `memory` (main.go:215) |
| `DEEPDATA_API_KEY` | by the server | Sent as `Authorization: Bearer` when set (main.go:216); a server started without `DEEPDATA_INSECURE_DEV_MODE=1` rejects unauthenticated calls |

A first session against an empty tenant is two calls: `deepdata_create_collection {"name":"memory","preset":"memory"}`,
then `deepdata_remember {"items":[{"text":"…"}]}`. The preset reads the server's embedder from `/readyz` and binds it
to a dense `text` field beside a sparse `keywords` field (`bm25`, dim 10000), so the agent never names a provider,
model or dimension (main.go:916). A server with `DEEPDATA_EMBEDDER=none` refuses the preset with
`embedder_unavailable` and a hint before anything is created.

## Tools

The server advertises exactly these tools in `tools/list` (main.go:435):

<!-- generated:mcp-tools -->
- `deepdata_recall`
- `deepdata_remember`
- `deepdata_forget`
- `deepdata_get`
- `deepdata_collections`
- `deepdata_create_collection`
<!-- /generated -->

Every tool carries `title`, `description`, `inputSchema`, `outputSchema` and annotations; every successful
result carries the JSON both as the text part and as `structuredContent`. Annotations: recall, get and
collections are `readOnlyHint` + `idempotentHint`; remember and create_collection declare
`destructiveHint: false`; forget is `destructiveHint: true` + `idempotentHint: true`; all six are
`openWorldHint: false`.

- `deepdata_recall` — exactly one of `query` (text) or `queries` (field → vector); optional `collection`,
  `top_k` (1–50, default 10), `filters` (the 15 operators plus `$and`/`$or`/`$not`, listed in the schema's
  `$defs.filter`), `score_floor`, `fallback`, `usage_boost`, `response_format` (`concise`, the default, or
  `detailed`), `max_chars` (default 8000, minimum 500). Text is embedded on every field of the collection that
  binds an embedding; with exactly one dense and one sparse binding and no caller `fallback`, the request gets
  `fallback {primary: <dense>, secondary: <sparse>}`. A collection with no binding answers `invalid_argument` with
  a hint naming `deepdata_create_collection`; more than two bindings answers a hint to pass `queries`. POST
  …/search with `include_vectors: false` (main.go:555). Output: `hits [{id, score, metadata}]`, `best_score`,
  `score_direction` (`lower_is_better` on dense distances, `higher_is_better` on sparse and fused scores, naming
  the field that actually answered), `weak_match`, `fell_back_to`, `embedded_by`, `truncated`, `hint`. Concise mode cuts every metadata string
  over 300 runes to 300 + `…` (main.go:681); both modes then drop tail hits until the JSON fits `max_chars`, set
  `truncated: true` and name the dropped ids in `hint`. Vectors are never returned.
- `deepdata_remember` — `items` (1–100), each exactly one of `text` or `vectors`, optional `metadata` and `id`;
  optional `collection`. Text is embedded for every bound field and kept at `metadata.text`. Items with an `id`
  go to PUT …/docs/{id} (idempotent, first); one fresh item goes to POST …/docs; several go to POST
  …/docs/batch, atomically (main.go:713). Output: `ids` in item order, `count`.
- `deepdata_forget` — `id`; optional `collection`. DELETE …/docs `{doc_id}`; a document that is already gone
  is a success, so the verb is idempotent (main.go:813). Output: `{deleted: id}`.
- `deepdata_get` — `ids` (1–50); optional `collection`. GET …/docs/{id} per id; unknown ids land in `missing`
  instead of failing the call (main.go:833). Output: `documents [{id, metadata}]`, `missing`. Vectors are stripped.
- `deepdata_collections` — optional `name`. With a name, GET …/collections/{name}; without, GET
  /v3/tenants/{tenant}/collections. Both need only the `read` permission, so a least-privilege agent can find
  out what it may search. Output: `collections [{name, fields [{name, type, dim, index, score_direction,
  embedding}], doc_count, description}]`; `score_direction` says which way that field's scores read
  (main.go:874).
- `deepdata_create_collection` — `name` (`[A-Za-z0-9_-]{1,64}`) and exactly one of `preset: "memory"` or
  explicit `fields`; optional `description`. POST …/collections, then GET the result so the output carries the
  server-resolved `fields` (main.go:916).

Collection schemas are cached per process; any 4xx for a collection drops its entry, so a deleted or recreated
collection is re-read on the next text call (main.go:272; test TestSchemaCacheDropsOn4xx).

## Resources

`resources/list` returns two entries (main.go:380); `resources/read` (main.go:387) serves:

- `deepdata://contract` — `api/contract/v3/CONTRACT.md` as `text/markdown`.
- `deepdata://status` — the server's `GET /v3/status` body as `application/json` (version, the HTTP/gRPC/MCP
  operation lists, the embedder, limits, capabilities); when the server is unreachable the error text is the
  content.
- Any other URI is JSON-RPC error `-32002`.

## Errors

- A non-2xx HTTP response becomes a tool result with `isError: true`. When the body is the server's JSON
  envelope ([API.md#errors](../internal/collection/API.md#errors)) the text part reads
  `<code>: <message> Hint: <hint>` and `structuredContent` carries the envelope unchanged, so a host reads
  `code`, `retryable` and `request_id` without parsing prose (main.go:330; test
  TestToolErrorsAreIsErrorNotRPCErrors). A non-JSON body is passed through trimmed as the text part, with no
  `structuredContent`. Failures the server never sees (text against an unbound field, the `memory` preset
  without an embedder) use the same envelope shape with `retryable: false` (main.go:160).
- Argument-shape violations — an unknown field, a missing required field, both or neither of an
  exactly-one pair, `top_k` out of range — are JSON-RPC error `-32602` and never reach the network
  (main.go:484, `DisallowUnknownFields`); an unknown tool is `-32601`; an unparsable frame is `-32700`; a
  wrong `jsonrpc` version is `-32600`.

`GOTOOLCHAIN=go1.25.12 go test ./cmd/deepdata-mcp ./api/contract` runs the tests in
`cmd/deepdata-mcp/main_test.go` and `api/contract/contract_test.go` against an httptest stand-in; both packages
are in the vet and test lists of `.github/workflows/ci.yml`, and `scripts/hardening_check.sh go-mcp` is the
receipt for gates CTL-03 and CI-05.

## Not exposed

Tenant and admin operations, `delete_collection`, `ef_search`, `hybrid_params`, `include_vectors`, cursors,
prompts and sampling. Tenant is process-bound; the rest is either not a memory operation or is covered by
`max_chars` truncation with a steering hint (journal 2026-09-01, section 5.7).
