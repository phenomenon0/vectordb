# Prometheus and Grafana status for the release candidate

DeepData exposes `GET /metrics` on the HTTP listener. The endpoint is useful
for scrape discovery, but the bundled `vectordb-dashboard.json` predates the
narrowed single-node RC contract and is an **experimental compatibility
asset**, not a supported UI or release gate.

## Scrape configuration

```yaml
scrape_configs:
  - job_name: deepdata
    scrape_interval: 15s
    metrics_path: /metrics
    static_configs:
      - targets:
          - 127.0.0.1:8080
```

`/metrics` is unauthenticated. Bind it to a protected network or restrict it at
the reverse proxy/service mesh; do not expose it directly to the internet.

## Dashboard limitations

The JSON dashboard contains panels for historical root/V2 traffic and for
shards, replication lag, and failover. Those are outside the RC and may remain
empty. In particular, these panels must not be used to claim distributed
health:

- Shard Node Health
- Replication Lag
- Failover Events
- Failover Duration

Some query and operation panels also depend on legacy instrumentation rather
than canonical V3/gRPC traffic. Import the dashboard only as a starting point,
hide unsupported panels, and verify every PromQL expression against the series
actually emitted by the exact release binary.

To inspect the raw contract before building alerts:

```bash
curl --fail --silent http://127.0.0.1:8080/metrics
```

## RC health signals

Use the process probes as the primary release signals:

| Endpoint | Meaning |
|---|---|
| `/livez` | The process can answer its liveness check |
| `/readyz` | Snapshot, mutation journal, and lifetime lock are healthy |
| `/metrics` | Prometheus exposition endpoint is reachable |

A `200` from `/livez` is not sufficient for traffic admission. Alert on a
non-`200` `/readyz` and remove the instance from service; a persistence fault
is intentionally fail-closed.

The RC does not ship a supported Grafana dashboard, web UI, replication
dashboard, or predefined production alert pack. Treat any customized dashboard
and alert thresholds as deployment-owned configuration that must be tested
against workload-specific objectives.
