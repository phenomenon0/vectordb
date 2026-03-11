# Distributed Architecture

## Overview

DeepData scales horizontally using collection-based sharding and WAL streaming replication. Single-node handles most workloads. Distributed mode is for when you need HA failover or dataset sizes beyond one machine.

## Design Principles

### 1. Collection-Based Sharding

Each collection maps to a shard using consistent hashing. Natural fit for multi-tenant (customer → collection → shard). Queries within a collection hit a single shard. Cross-collection queries fan out.

### 2. Replication for High Availability

Primary-Replica pattern (1 primary + N replicas per shard). Writes go to the primary, then async-replicate to followers via WAL streaming. Reads can go to any replica. Automatic failover when a primary fails.

### 3. WAL Streaming Replication

Replicas poll the primary's WAL endpoint (`/wal/stream?since=N`) to catch up. Adaptive polling: fast when behind, slow when idle. When a replica falls too far behind (WAL entries trimmed), it falls back to streaming snapshot sync.

### 4. Streaming Snapshots

New replicas or far-behind followers bootstrap via streaming snapshots. The primary gzip-compresses its state to a temp file on disk and streams it to the follower. This avoids loading the entire snapshot into memory — critical for large datasets.

- **Endpoints**: `/internal/snapshot/stream/download` (GET), `/internal/snapshot/stream/upload` (POST)
- **Backward compat**: Streaming-first with fallback to legacy `/internal/snapshot` on 404

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       DeepData Gateway                          │
│                  (HTTP + gRPC Entry Point)                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  DistributedVectorDB                             │
│        (Coordinator — Routes & Aggregates Queries)              │
└───────────────┬────────────────┬────────────────┬───────────────┘
                │                │                │
                ▼                ▼                ▼
        ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
        │  Shard 0      │ │  Shard 1      │ │  Shard N      │
        │  (Primary)    │ │  (Primary)    │ │  (Primary)    │
        │  Collections  │ │  Collections  │ │  Collections  │
        │  [A, D, G]    │ │  [B, E, H]    │ │  [C, F, I]    │
        └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
                │ WAL stream      │ WAL stream      │ WAL stream
        ┌───────▼────────┐ ┌──────▼────────┐ ┌─────▼─────────┐
        │ Replica 0-1    │ │ Replica 1-1   │ │ Replica N-1   │
        │ (Read replica) │ │ (Read replica) │ │ (Read replica) │
        └────────────────┘ └───────────────┘ └───────────────┘
```

## Routing

### Collection-Based (Default)
```
Collection → Hash(collection_name) % num_shards → Shard
```

Single shard per collection query. Cross-collection queries fan out to all relevant shards, aggregate top-K results, and re-rank.

### Broadcast Queries
Query without collection filter → broadcast to all shards → merge top-K globally.

## Replication Protocol

### Write Path
1. Primary validates + writes to WAL
2. Primary updates in-memory index
3. Primary returns success to client
4. Primary's WAL entries become available to replicas

### Read Path (Configurable)
- **Primary-only**: Strongest consistency, higher load on primary
- **Replica-prefer**: Eventual consistency (~100ms lag), default
- **Balanced**: 20% primary, 80% replicas

### Failover
When the health manager detects a primary failure:
1. Pick the replica with least WAL lag
2. Promote to primary via `/promote`
3. Update routing table
4. Remaining replicas re-point to new primary

## Configuration

### Example: 3 Shards, 2 Replicas

```yaml
coordinator:
  num_shards: 3
  replication_factor: 2
  read_strategy: "replica-prefer"
  consistency: "eventual"
  sync_interval_ms: 100
  health_check_interval_s: 10
```

Each shard node:
```yaml
vectordb_node:
  shard_id: 0
  role: primary          # or "replica"
  primary_addr: ""       # set for replicas
  http_addr: ":9000"
  grpc_addr: ":50051"
  storage_path: "/data/shard0"
  capacity: 5000000
  dimension: 384
```

## Capacity Planning

```
Shard capacity = RAM available / memory per vector

Example (64GB RAM):
- 384d vectors: ~2.5GB per 1M vectors
- Overhead: ~30% (HNSW, metadata, docs)
- Usable: ~19M vectors per shard
```

| Shards | Replicas | Vectors/Shard | Total Capacity | Fault Tolerance |
|--------|----------|---------------|----------------|-----------------|
| 3 | 2 | 10M | 30M | 1 node loss |
| 6 | 2 | 10M | 60M | 1 node loss |
| 10 | 3 | 10M | 100M | 2 node loss |

For datasets exceeding RAM, DiskANN indexes can memory-map graph data to disk within each shard.

## Performance

| Query Type | Latency | Notes |
|------------|---------|-------|
| Single-collection | 10-50ms | 1 shard |
| Multi-collection | 20-100ms | Parallel fan-out |
| Broadcast | 50-200ms | All shards |

Throughput per coordinator: ~5-10K insert/s, ~10-20K query/s (load-balanced across replicas).

## Monitoring

Each shard exposes `/metrics` (Prometheus) and optional OpenTelemetry traces:
- Per-shard: vector count, query latency, throughput
- Per-replica: replication lag, sync errors, snapshot status
- Per-coordinator: route success rate, query fanout

Health probes: `/healthz` (liveness), `/readyz` (readiness).

## Status

Implemented:
- Collection-based routing and shard management
- Primary-replica WAL streaming replication
- Streaming snapshot sync for bootstrap / catch-up
- Automatic failover with health monitoring
- gRPC + HTTP per shard

Planned:
- Automatic rebalancing (hot shard splitting)
- Geo-distribution and cross-region replication
- Quorum reads/writes for strong consistency
- Query result caching
