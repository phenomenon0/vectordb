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
are registered (`cmd/deepdata/metrics.go:190-206`) and none has a writer:
the two HTTP families were fed by `withMetrics`, which wrapped the legacy
root routes deleted together with the v1 engine, and the canonical V3
handler is registered with `guard` alone. A scrape therefore returns the
registered families with no samples.

Instrumenting the V3 handlers is future work and is not tracked by a gate.

## Dashboard

No dashboard JSON ships in this directory. The one that used to live here
kept only panels reading `vectordb_http_requests_total` and
`vectordb_http_request_duration_seconds`; once nothing wrote those families
every panel read "No data", and linter rule R9 (`scripts/check_docs_contract.py`,
the command of gate DOC-01 in `tasks/gates.json`) rejects a panel whose family
no reachable code writes. `.github/workflows/ci.yml` runs that script in the
Linux RC Go contract job, so a dashboard has to wait for the V3 handlers to
be instrumented.

## Declared but not emitted

Every family in `cmd/deepdata/metrics.go` is registered and has no writer,
so none carries samples in the exposition:

- `vectordb_http_requests_total`, `vectordb_http_request_duration_seconds`
  (`:169-183`; the `RecordHTTPRequest`/`withMetrics` writers went with the
  v1 root routes)
- `vectordb_vectors_total`, `vectordb_vectors_deleted` (`:58-72`)
- `vectordb_operations_total`, `vectordb_operation_duration_seconds`,
  `vectordb_operation_errors_total` (`:74-97`)
- `vectordb_query_duration_seconds`, `vectordb_query_results`,
  `vectordb_query_shards_fanout` (`:100-125`)
- `vectordb_shard_health_status`, `vectordb_shard_replication_lag_operations`,
  `vectordb_shard_nodes`, `vectordb_failover_total`,
  `vectordb_failover_duration_seconds` (`:128-168`). Sharding,
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
