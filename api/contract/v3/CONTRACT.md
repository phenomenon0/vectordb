# DeepData agent contract (v3)

DeepData is durable memory: store text or vectors, get ranked documents back. You reach it
through six MCP verbs. Your tenant is fixed by the server process (`DEEPDATA_TENANT`); you
never pass it. `collection` defaults to `DEEPDATA_COLLECTION` (default `memory`) on every verb.

## The six verbs

| Verb | What it does | Safety |
|---|---|---|
| `deepdata_recall` | Search a collection; ranked hits with confidence signals | read-only, idempotent |
| `deepdata_remember` | Store 1–100 items (text or vectors); returns their ids | additive |
| `deepdata_forget` | Delete one document by id | destructive, idempotent |
| `deepdata_get` | Fetch 1–50 documents by id; unknown ids land in `missing` | read-only, idempotent |
| `deepdata_collections` | Describe one collection, or list all of them | read-only, idempotent |
| `deepdata_create_collection` | Create a collection from the `memory` preset or explicit fields | additive |

Exact argument and result shapes are the JSON Schemas beside this file (one per verb, e.g.
`schemas/deepdata_recall.json`, each `{"input": …, "output": …}`) — the same bytes the MCP
server serves as `inputSchema`/`outputSchema`.

## Texts vs vectors

A collection field may carry an `embedding` binding (`provider`, `model`). Bound fields accept
plain text: the server embeds it on write and on query, and reports `embedded_by`
(field → `provider:model`) in responses. Unbound fields require vectors you computed yourself.
Never send the same field as both text and vector. `deepdata_remember` with `text` also stores
the raw string at `metadata.text`, so recall gives the words back, not just ids.

With `query` text and two bound fields (the `memory` preset: dense `text` + sparse `keywords`),
recall searches the dense field first and falls back to the sparse field automatically,
reporting `fell_back_to` when that happened. Pass your own `fallback` to override.

## Reading a recall result

- `hits` are best-first; `score` is the raw engine score. `score_direction` names how to
  read it: dense fields score by distance (`lower_is_better`), sparse and fused scores are
  similarities (`higher_is_better`). It describes the field that actually answered, so a
  fallback answer reports the secondary field's direction.
- `best_score` calibrates future `score_floor` values, in the same direction.
- `weak_match: true` means your `score_floor` filtered everything — treat as "no confident
  answer", not an error.
- `truncated: true` means the `max_chars` budget dropped tail hits; the `hint` names them.
  Re-recall with a filter, a smaller `top_k`, or a larger `max_chars` if you need the rest.
- Vectors are never returned over MCP.

## Errors are prompts

Failures return `isError` with a text line `code: message Hint: hint` and the full envelope as
`structuredContent` (shape: `schemas/error.json`). Follow the `hint` — it names the next call.

| code | meaning | retry? |
|---|---|---|
| `invalid_argument` | malformed or out-of-range argument; `field` says which | no — fix the call |
| `not_found` | collection or document does not exist | no |
| `already_exists` | collection name taken | no |
| `unauthenticated` / `permission_denied` | credential absent / lacks the permission | no |
| `quota_exceeded` | tenant or collection limit; permanent for this process | no |
| `embedding_mismatch` | field binding disagrees with the server embedder | no |
| `payload_too_large` | request over the journal bound; split the batch | no |
| `rate_limited` | over RPS; wait `retry_after_ms` | yes |
| `embedder_unavailable` | text sent but no embedder is configured | yes, or send vectors |
| `unavailable` | store fault-latched or starting | yes |
| `internal` | server bug; report it | no |

## Limits

- `top_k` ≤ 50 over MCP; identifiers (collections, fields) match `[A-Za-z0-9_-]{1,64}`.
- `deepdata_remember` ≤ 100 items per call; `deepdata_get` ≤ 50 ids.
- At most 2 fields per search; recall output is budgeted by `max_chars` (default 8000).
- Listing all collections and describing one by name both need only the read permission.

## Resources

- `deepdata://contract` — this document.
- `deepdata://status` — the server's `GET /v3/status`: version, the operation list, the
  embedder it will use, its limits, its capabilities, and `signals.usage.loaded`, which is
  false only when persisted usage records were discarded.
