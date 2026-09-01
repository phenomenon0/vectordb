# Prometheus and Grafana for the release candidate

DeepData serves `GET /metrics` on the HTTP listener (default `PORT` 8080,
`cmd/deepdata/main.go:3284`). The RC binary is canonical-only
(`const canonicalOnly = true`, `cmd/deepdata/main.go:3166`);
`canonicalRCSurface` (`cmd/deepdata/server.go:2554-2565`) lets `/metrics`
through together with `/v3/tenants/`, `/healthz`, `/readyz` and `/livez`,
and answers 404 for every other registered route.

## Authentication

`/metrics` is registered as `mux.Handle("/metrics", guard(...))`
(`cmd/deepdata/server.go:1324`). `guard` is the same closure that fronts the
API routes (`cmd/deepdata/server.go:160`): when authentication is required
(`REQUIRE_AUTH=1`, or a JWT secret or API token is configured,
`cmd/deepdata/runtime.go:35`) a scrape needs the same bearer token as an API
call. The comment at `cmd/deepdata/server.go:1320-1323` records why:
request-volume and operation detail leak through this endpoint, so it is
gated exactly like the API. The gate landed in 043ad5d.

## Scrape configuration

```yaml
scrape_configs:
  - job_name: deepdata
    scrape_interval: 15s
    metrics_path: /metrics
    authorization:
      type: Bearer
      credentials: <the token the API accepts>
    static_configs:
      - targets:
          - 127.0.0.1:8080
```

## What the binary emits

`/metrics` serves a private registry (`cmd/deepdata/metrics.go:52`, handler
`:212-214`); Go runtime metrics are not on it. Fifteen `vectordb_*` families
are registered (`cmd/deepdata/metrics.go:190-206`), but only two have a
write path outside test files:

| Family | Type | Labels | Written by |
|---|---|---|---|
| `vectordb_http_requests_total` | counter | `method`, `endpoint`, `status` | `RecordHTTPRequest` (`cmd/deepdata/metrics.go:316-324`) |
| `vectordb_http_request_duration_seconds` | histogram | `method`, `endpoint` | same |

`RecordHTTPRequest` is reached only through `withMetrics`
(`cmd/deepdata/metrics.go:338-350`), which wraps the historical root routes
registered in `cmd/deepdata/server.go` (`/insert` at `:342`, `/query` at
`:687`, `/delete` at `:1238`, ... `/api/index/create` at `:2709`). Two facts
follow for the RC binary:

- `canonicalRCSurface` answers 404 for those root routes before the mux sees
  them, and the canonical handler (`cmd/deepdata/collection_http.go:344`) is
  registered with `guard` alone, not `withMetrics`. V3 traffic therefore
  produces no samples in either family today.
- The `endpoint` label is the literal string passed to `withMetrics`
  (`"query"`, `"insert"`, ...), not a URL path. `HTTPMiddleware`
  (`cmd/deepdata/metrics.go:262-277`), which would label by normalized path,
  has no caller outside `cmd/deepdata/metrics_test.go`.

Instrumenting the V3 handlers is future work and is not tracked by a gate.

## Dashboard

`vectordb-dashboard.json` keeps only panels whose PromQL reads the two
families above:

| Row | Panel | Expression reads |
|---|---|---|
| Overview | Queries/sec (id 3) | `vectordb_http_requests_total{endpoint="query"}` |
| Overview | Error Rate (id 6) | 5xx share of `vectordb_http_requests_total` |
| Query Performance | Requests/sec by Endpoint (id 11) | `vectordb_http_requests_total` by `endpoint` |
| Query Performance | HTTP Latency by Endpoint (id 13) | P95 of `vectordb_http_request_duration_seconds_bucket` |

Until V3 is instrumented, all four panels read "No data" against an RC
deployment (see the previous section). Import the JSON as a starting point;
it is an experimental compatibility asset, not a supported UI or a release
gate. Linter rule R9 (`scripts/check_docs_contract.py`, the command of gate
DOC-01 in `tasks/gates.json`) reports a violation if a panel references a
family that no reachable code writes; `.github/workflows/ci.yml` runs that script
in the Linux RC Go contract job.

## Declared but not emitted

The remaining families in `cmd/deepdata/metrics.go` are registered and have
no writer outside test files, so they never appear in the exposition and no
panel reads them:

- `vectordb_vectors_total`, `vectordb_vectors_deleted` (`:58-72`)
- `vectordb_operations_total`, `vectordb_operation_duration_seconds`,
  `vectordb_operation_errors_total` (`:74-97`; writer `RecordOperation`
  `:217`, called only from tests)
- `vectordb_query_duration_seconds`, `vectordb_query_results`,
  `vectordb_query_shards_fanout` (`:100-125`; writer `RecordQuery` `:229`,
  called only from tests)
- `vectordb_shard_health_status`, `vectordb_shard_replication_lag_operations`,
  `vectordb_shard_nodes`, `vectordb_failover_total`,
  `vectordb_failover_duration_seconds` (`:128-168`; writer
  `UpdateShardHealth` `:241`, called only from tests). Sharding,
  replication and failover are non-goals of the RC; these families are not
  served and must not be used to claim distributed health.

The `deepdata_*` families in `internal/telemetry/metrics.go` are registered
on the default Prometheus registry (`internal/telemetry/metrics.go:78`), which
`/metrics` does not serve.

## RC health signals

Use the probes as the release signals; `/metrics` reachability is secondary.

| Endpoint | Meaning |
|---|---|
| `/livez`, `/healthz` | The process answers its liveness check (`cmd/deepdata/server.go:1390-1391`) |
| `/readyz` | Named checks `collection_snapshot`, `mutation_journal`, `lifetime_lock` (`cmd/deepdata/server.go:1432`) |
| `/metrics` | Exposition endpoint reachable with credentials |

A `200` from `/livez` is not sufficient for traffic admission. Alert on a
non-`200` `/readyz` and remove the instance from service; a persistence fault
is intentionally fail-closed.

The RC does not ship a supported Grafana dashboard, web UI, or alert pack.
Treat any customized dashboard and thresholds as deployment-owned
configuration tested against workload-specific objectives.
