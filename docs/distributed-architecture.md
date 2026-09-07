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
- automatic leader election, promotion, or failover;
- quorum reads/writes or configurable consistency levels;
- durable membership, fencing, or split-brain protection; or
- multiple processes sharing or concurrently mounting the same data directory.

Do not place several DeepData processes behind a load balancer as if they were
interchangeable. Their memory indexes and journals do not form a cluster, and a
shared volume does not turn the lifetime lock into a distributed consensus
protocol.

## The node surface is off unless a node credential is set

An experimental single-leader journal transport is present but disabled. It is
not part of the RC contract and is not covered by a release claim.

Setting `DEEPDATA_REPLICATION_TOKEN` turns it on. There is no separate boolean:
the credential is the switch, so a process cannot be serving these routes
without one. When it is unset the routes return 404 like any other path outside
the RC surface.

What it does, and only this: a leader exposes its own journal and a snapshot of
its own state, and `deepdata replicate` keeps a second tenant directory in step
by applying that journal in LSN order (one tenant at a time; omitting
`--tenant` follows every tenant the leader lists, each into its own
subdirectory of the tenant store tree). Stop that sync and a plain `deepdata
serve` against the same tenant directory serves it read-only. Nothing tells it
to: the sync writes a marker beside that tenant's store, so the tenant
directory is the evidence and there is no flag to forget. Reads answer
normally, every write is refused with `403` `permission_denied` before it
reaches the journal, and `GET /readyz` and `GET /v3/status` both report
`read_only` so a load balancer stops sending it writes.

`deepdata replicate`'s exclusive lock is not a turn of phrase: the collection
store takes it per tenant for the lifetime of the process holding it, so
`deepdata replicate` and `deepdata serve` cannot both hold the tenant it is
syncing -- whichever opens that tenant's store second is refused with
`collection store is already open`. Moving a tenant directory between the two
roles means stopping one process and starting the other.

Serving reads while the sync continues does not need a second process: setting
`DEEPDATA_LEADER_URL` (with the same node credential) turns `deepdata serve`
itself into a standby. It follows every tenant the leader lists, in-process,
into its own tenant store tree -- the same seed-then-tail sync `deepdata
replicate` performs, sharing its re-list and retry loop -- and answers reads
for what it has synced the whole time it follows, not only once it stops.

`GET /readyz` reports this as a `following` block:
`{"leader": "...", "tenants": {"<id>": {"state", "applied_lsn", "leader_lsn",
"lag"}}}`. `state` is `bootstrapping` while a tenant seeds from a snapshot,
`streaming` while it tails the leader's journal, `reconnecting` while a
dropped stream retries on its own, or `stopped` on a terminal fault (see
[docs/troubleshooting.md](troubleshooting.md)). `applied_lsn` is this tenant's
own durable cursor; `leader_lsn` is the leader's position as of the preamble
the stream opened with; `lag` is the difference between the two measured
against that same snapshot, so a caught-up stream reports `lag: 0` -- it is
not a live delta against whatever the leader does next.

Following is per tenant: a tenant already on the standby's disk that the
leader does not list is untouched, exactly as ordinary and writable as
before. The marker a sync writes beside a tenant's store is still the
evidence once the standby stops: a plain `deepdata serve` against that
directory afterward, without `DEEPDATA_LEADER_URL`, still opens that tenant
read-only. `deepdata replicate` is unchanged and remains the sync-only mode:
it never serves.

What it does not do: it never elects, promotes, fences, or fails over; it has no
membership list; it does not resync itself when it falls too far behind, because
discarding a replica directory is an operator decision, not a retry policy; and
it does not make the leader highly available. A dropped stream is retried from
the replica's own durable cursor. Nothing else is automatic.

### The node credential is not the API token

`DEEPDATA_REPLICATION_TOKEN` authenticates a peer server, not a tenant. One
request on this surface transfers a snapshot of every tenant in the store, so it
must never be an API token issued to an application, and the routes must not be
reachable from an untrusted network. A tenant credential, and the anonymous
server-admin context that `DEEPDATA_INSECURE_DEV_MODE=1` grants, are both
rejected here.

## Experimental source

The repository may still contain packages, metrics, tests, or design remnants
for sharding, replication, and failover beyond the transport described above.
They are experimental, non-RC source. They are not reachable through the
canonical production surface, are not included in release claims, and must not
be used as deployment instructions.

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
and recovery objectives. The node surface above supplies two of those -- an
authenticated node protocol and a bootstrap path with fault tests -- and none
of the rest. Until every one of those gates exists and passes, this code
remains experimental and the RC claim above does not change.
