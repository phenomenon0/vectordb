# Distributed architecture status

DeepData does **not** have a supported distributed architecture in the release
candidate.

The production contract is one persistent Linux process, one locked data
directory, and one canonical tenant-aware collection engine. HTTP and gRPC may
both be enabled on that process, but they are two transports over the same
state—not separate nodes.

## Unsupported distributed behaviors

The RC does not support or qualify:

- collection sharding or cross-node query fan-out;
- primary/follower replication or WAL streaming;
- read replicas or follower restore;
- automatic leader election, promotion, or failover;
- quorum reads/writes or configurable consistency levels;
- snapshot upload/download or snapshot streaming; or
- multiple processes sharing or concurrently mounting the same data directory.

Do not place several DeepData processes behind a load balancer as if they were
replicas. Their memory indexes and journals do not form a cluster, and a shared
volume does not turn the lifetime lock into a distributed consensus protocol.

## Experimental source

The repository may still contain packages, metrics, tests, or design remnants
for sharding, replication, and failover. They are experimental, non-RC source.
They are not reachable through the canonical production surface, are not
included in release claims, and must not be used as deployment instructions.

## Supported topology

```text
clients
   |
external TLS / authentication-aware network boundary
   |
one DeepData Linux process
   |
one exclusive local or single-writer persistent data directory
```

Availability for this RC means supervised restart and a tested stopped-backup
recovery procedure for that one state directory. It does not mean transparent
failover. Operators who require zero-downtime HA or online replica recovery
should defer adoption or place DeepData behind an application architecture
that does not claim DeepData instances are one consistent cluster.

## Requirements before distributed support can be claimed

Future distributed work needs an explicit consistency model, authenticated
node protocol, durable membership and fencing, replication fault tests,
snapshot/bootstrap compatibility, split-brain tests, rolling-upgrade rules,
and recovery objectives. Until those gates exist and pass, distributed code
remains experimental.
