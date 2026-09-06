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

`/metrics` serves a private registry (`cmd/deepdata/metrics.go:33`, handler
`:86-88`) with four `vectordb_tenant_*` families, every one labelled by
tenant:

- `vectordb_tenant_requests_total{transport,tenant,operation,code}` — a
  counter, incremented once per completed request.
- `vectordb_tenant_request_duration_seconds{transport,tenant,operation}` —
  a histogram (default buckets) observed on the same request.
- `vectordb_tenant_documents{tenant}` and `vectordb_tenant_bytes{tenant}` —
  gauges holding the tenant's current usage.

The counter and histogram are written by `RecordTenantRequest`
(`cmd/deepdata/metrics.go:92-98`), called from both transports:
`instrumentCanonicalHTTP` wraps every canonical V3 HTTP handler
(`cmd/deepdata/collection_http.go:327-339`), and the gRPC unary interceptor
times and records every call (`cmd/deepdata/main.go:591-597`). `tenant` is
`canonicalRateLimitTenant(tenantCtx, target)` — the caller's own tenant, or
the addressed tenant only when the caller is a server admin — never the raw
path/request segment, so one tenant's credentials can't attribute load to
another tenant's label; it is `"unknown"` when no tenant context resolved.
`operation` is `normalizeMetricsPath(path)` for HTTP or `info.FullMethod` for
gRPC, which collapses the `{tenant}`, `{collection}` and `{doc_id}` path
segments to `:id`, `:name` and `:doc_id` so cardinality tracks route shape,
not tenant or document count.

The two gauges are written by `RefreshTenantUsage`
(`cmd/deepdata/metrics.go:102-112`), which resets both and re-sets them from
`TenantManager.ListTenantInfos()` on every scrape
(`cmd/deepdata/server.go:120-127`; `// ponytail: O(tenants) per scrape`).

Go runtime metrics are not on this registry.

## Dashboard

No dashboard JSON ships in this directory. A dashboard added here must only
read the four `vectordb_tenant_*` families above: linter rule R9
(`scripts/check_docs_contract.py`, the command of gate DOC-01 in
`tasks/gates.json`) rejects a panel whose family isn't declared and written
from reachable code in `cmd/deepdata/metrics.go`. `.github/workflows/ci.yml`
runs that script in the Linux RC Go contract job.

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
