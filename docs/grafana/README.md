# Grafana Dashboard for VectorDB

Pre-built dashboard for monitoring VectorDB via Prometheus.

## Setup

1. Ensure VectorDB is running — metrics are exposed at `GET /metrics`
2. Point Prometheus at your VectorDB instance:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: vectordb
    scrape_interval: 10s
    static_configs:
      - targets: ['localhost:8080']
```

3. Import the dashboard:
   - Grafana → Dashboards → Import → Upload `vectordb-dashboard.json`
   - Select your Prometheus data source when prompted

## Panels

### Overview Row
- **Total Vectors** — gauge of stored vectors across all shards
- **Deleted (Tombstones)** — vectors pending compaction
- **Queries/sec** — current query throughput
- **Query P50 / P99** — latency percentiles
- **Error Rate** — 5xx responses as fraction of total

### Query Performance Row
- **Query Latency Percentiles** — P50/P95/P99 over time
- **Requests/sec by Endpoint** — stacked area per endpoint
- **Results per Query** — distribution of result set sizes
- **HTTP Latency by Endpoint** — P95 per endpoint

### Operations Row
- **Operations by Type** — insert/query/delete rates
- **Errors by Type** — error breakdown by operation

### Shard Health Row (Distributed Mode)
- **Shard Node Health** — UP/DOWN per node
- **Replication Lag** — ops behind primary per replica
- **Failover Events** — failover rate by shard
- **Failover Duration** — P95 failover time

## Metrics Reference

All metrics are prefixed with `vectordb_`:

| Metric | Type | Labels |
|--------|------|--------|
| `vectordb_vectors_total` | Gauge | shard_id, collection, node_id |
| `vectordb_vectors_deleted` | Gauge | shard_id, collection, node_id |
| `vectordb_operations_total` | Counter | operation, shard_id, status |
| `vectordb_operation_duration_seconds` | Histogram | operation, shard_id |
| `vectordb_operation_errors_total` | Counter | operation, shard_id, error_type |
| `vectordb_query_duration_seconds` | Histogram | mode, collections |
| `vectordb_query_results` | Histogram | mode |
| `vectordb_http_requests_total` | Counter | method, endpoint, status |
| `vectordb_http_request_duration_seconds` | Histogram | method, endpoint |
| `vectordb_shard_health_status` | Gauge | shard_id, node_id, role |
| `vectordb_shard_replication_lag_operations` | Gauge | shard_id, node_id |
| `vectordb_failover_total` | Counter | shard_id, status |
| `vectordb_failover_duration_seconds` | Histogram | shard_id |

## Alerting Rules (Optional)

```yaml
# prometheus-alerts.yml
groups:
  - name: vectordb
    rules:
      - alert: HighQueryLatency
        expr: histogram_quantile(0.99, rate(vectordb_query_duration_seconds_bucket[5m])) > 0.5
        for: 5m
        labels: { severity: warning }
        annotations: { summary: "VectorDB P99 query latency > 500ms" }

      - alert: HighErrorRate
        expr: sum(rate(vectordb_http_requests_total{status=~"5.."}[5m])) / sum(rate(vectordb_http_requests_total[5m])) > 0.05
        for: 5m
        labels: { severity: critical }
        annotations: { summary: "VectorDB error rate > 5%" }

      - alert: ShardDown
        expr: vectordb_shard_health_status == 0
        for: 1m
        labels: { severity: critical }
        annotations: { summary: "VectorDB shard {{ $labels.shard_id }} node {{ $labels.node_id }} is down" }

      - alert: HighReplicationLag
        expr: vectordb_shard_replication_lag_operations > 1000
        for: 5m
        labels: { severity: warning }
        annotations: { summary: "Replica lag > 1000 ops for shard {{ $labels.shard_id }}" }
```
