# Distributed VectorDB Deployment Guide

## Quick Start

### Option 1: Local Development (Single Machine)

Run coordinator + 3 shards with 2 replicas each on localhost:

```bash
# Terminal 1: Start Coordinator
go run . -mode=coordinator \
  -addr=:8080 \
  -shards=3 \
  -replication=2

# Terminal 2: Shard 0 Primary
go run . -mode=shard \
  -node-id=shard-0-primary \
  -shard-id=0 \
  -role=primary \
  -addr=:9000 \
  -coordinator=http://localhost:8080 \
  -capacity=1000000 \
  -dim=384 \
  -index-path=/tmp/vectordb/shard0/primary

# Terminal 3: Shard 0 Replica
go run . -mode=shard \
  -node-id=shard-0-replica1 \
  -shard-id=0 \
  -role=replica \
  -addr=:9001 \
  -primary=http://localhost:9000 \
  -coordinator=http://localhost:8080 \
  -capacity=1000000 \
  -dim=384 \
  -index-path=/tmp/vectordb/shard0/replica1

# Terminal 4: Shard 1 Primary
go run . -mode=shard \
  -node-id=shard-1-primary \
  -shard-id=1 \
  -role=primary \
  -addr=:9002 \
  -coordinator=http://localhost:8080 \
  -capacity=1000000 \
  -dim=384 \
  -index-path=/tmp/vectordb/shard1/primary

# Terminal 5: Shard 1 Replica
go run . -mode=shard \
  -node-id=shard-1-replica1 \
  -shard-id=1 \
  -role=replica \
  -addr=:9003 \
  -primary=http://localhost:9002 \
  -coordinator=http://localhost:8080 \
  -capacity=1000000 \
  -dim=384 \
  -index-path=/tmp/vectordb/shard1/replica1

# Terminal 6: Shard 2 Primary
go run . -mode=shard \
  -node-id=shard-2-primary \
  -shard-id=2 \
  -role=primary \
  -addr=:9004 \
  -coordinator=http://localhost:8080 \
  -capacity=1000000 \
  -dim=384 \
  -index-path=/tmp/vectordb/shard2/primary

# Terminal 7: Shard 2 Replica
go run . -mode=shard \
  -node-id=shard-2-replica1 \
  -shard-id=2 \
  -role=replica \
  -addr=:9005 \
  -primary=http://localhost:9004 \
  -coordinator=http://localhost:8080 \
  -capacity=1000000 \
  -dim=384 \
  -index-path=/tmp/vectordb/shard2/replica1
```

### Option 2: Docker Compose

```yaml
version: '3.8'

services:
  coordinator:
    build: .
    command: >
      -mode=coordinator
      -addr=:8080
      -shards=3
      -replication=2
    ports:
      - "8080:8080"
    networks:
      - vectordb

  shard-0-primary:
    build: .
    command: >
      -mode=shard
      -node-id=shard-0-primary
      -shard-id=0
      -role=primary
      -addr=:9000
      -coordinator=http://coordinator:8080
      -capacity=5000000
      -dim=384
      -index-path=/data/index
    volumes:
      - shard0-primary:/data
    networks:
      - vectordb

  shard-0-replica1:
    build: .
    command: >
      -mode=shard
      -node-id=shard-0-replica1
      -shard-id=0
      -role=replica
      -addr=:9000
      -primary=http://shard-0-primary:9000
      -coordinator=http://coordinator:8080
      -capacity=5000000
      -dim=384
      -index-path=/data/index
    volumes:
      - shard0-replica1:/data
    networks:
      - vectordb

  # ... repeat for other shards

networks:
  vectordb:
    driver: bridge

volumes:
  shard0-primary:
  shard0-replica1:
  # ... other volumes
```

### Option 3: Kubernetes

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: vectordb-config
data:
  NUM_SHARDS: "3"
  REPLICATION_FACTOR: "2"
  CAPACITY: "5000000"
  DIMENSION: "384"

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vectordb-coordinator
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vectordb-coordinator
  template:
    metadata:
      labels:
        app: vectordb-coordinator
    spec:
      containers:
      - name: coordinator
        image: vectordb:latest
        args:
          - "-mode=coordinator"
          - "-addr=:8080"
          - "-shards=$(NUM_SHARDS)"
          - "-replication=$(REPLICATION_FACTOR)"
        ports:
        - containerPort: 8080
        envFrom:
        - configMapRef:
            name: vectordb-config

---
apiVersion: v1
kind: Service
metadata:
  name: vectordb-coordinator
spec:
  selector:
    app: vectordb-coordinator
  ports:
  - port: 8080
    targetPort: 8080
  type: LoadBalancer

---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: vectordb-shard-0
spec:
  serviceName: "vectordb-shard-0"
  replicas: 2  # 1 primary + 1 replica
  selector:
    matchLabels:
      app: vectordb-shard
      shard-id: "0"
  template:
    metadata:
      labels:
        app: vectordb-shard
        shard-id: "0"
    spec:
      containers:
      - name: shard
        image: vectordb:latest
        args:
          - "-mode=shard"
          - "-node-id=shard-0-$(HOSTNAME)"
          - "-shard-id=0"
          - "-role=primary"  # Set via pod index
          - "-addr=:9000"
          - "-coordinator=http://vectordb-coordinator:8080"
          - "-capacity=$(CAPACITY)"
          - "-dim=$(DIMENSION)"
          - "-index-path=/data/index"
        ports:
        - containerPort: 9000
        volumeMounts:
        - name: data
          mountPath: /data
        envFrom:
        - configMapRef:
            name: vectordb-config
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 100Gi
```

## Testing the Deployment

### 1. Insert Data
```bash
# Insert into collection "docs"
curl -X POST http://localhost:8080/insert \
  -H "Content-Type: application/json" \
  -d '{
    "doc": "Machine learning is a subset of artificial intelligence",
    "collection": "docs",
    "meta": {"category": "AI", "source": "manual"}
  }'

# Response:
# {"id":"doc-1","collection":"docs"}
```

### 2. Batch Insert
```bash
curl -X POST http://localhost:8080/batch_insert \
  -H "Content-Type: application/json" \
  -d '{
    "docs": [
      {"doc": "Deep learning uses neural networks", "collection": "docs", "meta": {"category": "AI"}},
      {"doc": "Natural language processing analyzes text", "collection": "docs", "meta": {"category": "NLP"}},
      {"doc": "Computer vision processes images", "collection": "docs", "meta": {"category": "CV"}}
    ]
  }'
```

### 3. Query Collection
```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "artificial intelligence and neural networks",
    "top_k": 5,
    "collections": ["docs"],
    "mode": "hybrid"
  }'

# Response:
# {
#   "ids": ["doc-1", "doc-2"],
#   "docs": ["...", "..."],
#   "scores": [0.95, 0.87],
#   "meta": [{...}, {...}],
#   "count": 2
# }
```

### 4. Cross-Collection Query
```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning techniques",
    "top_k": 10,
    "collections": ["docs", "research", "tutorials"]
  }'
```

### 5. Check Cluster Status
```bash
curl http://localhost:8080/admin/cluster_status | jq .

# Response:
# {
#   "num_shards": 3,
#   "replication_factor": 2,
#   "read_strategy": "replica-prefer",
#   "total_nodes": 6,
#   "shards": [
#     {
#       "shard_id": 0,
#       "nodes": [
#         {"node_id": "shard-0-primary", "role": "primary", "healthy": true, ...},
#         {"node_id": "shard-0-replica1", "role": "replica", "healthy": true, ...}
#       ]
#     },
#     ...
#   ]
# }
```

## Performance Testing

### Load Test with Apache Bench
```bash
# Insert test
ab -n 10000 -c 50 -p insert.json -T application/json \
  http://localhost:8080/insert

# Query test
ab -n 10000 -c 50 -p query.json -T application/json \
  http://localhost:8080/query
```

### Multi-Tenant Load Test
```bash
# Simulate 100 tenants with 1000 docs each
for tenant in {1..100}; do
  for doc in {1..1000}; do
    curl -X POST http://localhost:8080/insert \
      -H "Content-Type: application/json" \
      -d "{\"doc\":\"Document $doc for tenant $tenant\",\"collection\":\"tenant-$tenant\"}" &
  done
  wait
done
```

## Monitoring

### Health Checks
```bash
# Coordinator health
curl http://localhost:8080/health

# Individual shard health
curl http://localhost:9000/health
curl http://localhost:9002/health
curl http://localhost:9004/health
```

### Metrics (if Prometheus enabled)
```bash
curl http://localhost:8080/metrics
```

## Scaling Operations

### Adding a New Shard
```bash
# 1. Start new shard primary
go run . -mode=shard \
  -node-id=shard-3-primary \
  -shard-id=3 \
  -role=primary \
  -addr=:9006 \
  -coordinator=http://localhost:8080 \
  -capacity=1000000 \
  -dim=384 \
  -index-path=/tmp/vectordb/shard3/primary

# 2. Start replica
go run . -mode=shard \
  -node-id=shard-3-replica1 \
  -shard-id=3 \
  -role=replica \
  -addr=:9007 \
  -primary=http://localhost:9006 \
  -coordinator=http://localhost:8080 \
  -capacity=1000000 \
  -dim=384 \
  -index-path=/tmp/vectordb/shard3/replica1

# 3. Trigger rebalancing (future feature)
# curl -X POST http://localhost:8080/admin/rebalance
```

### Removing a Shard
```bash
# 1. Mark shard as draining (future feature)
# curl -X POST http://localhost:8080/admin/drain_shard -d '{"shard_id": 3}'

# 2. Wait for migration to complete

# 3. Unregister nodes
curl -X POST http://localhost:8080/admin/unregister_shard \
  -d '{"node_id": "shard-3-primary"}'

curl -X POST http://localhost:8080/admin/unregister_shard \
  -d '{"node_id": "shard-3-replica1"}'

# 4. Shutdown nodes
```

## Production Recommendations

### Hardware Sizing
**Coordinator:**
- 2-4 CPU cores
- 4-8 GB RAM
- Minimal storage (logs only)

**Shard Node (for 5M vectors @ 384 dims):**
- 8-16 CPU cores (for HNSW indexing)
- 32-64 GB RAM (12.5 GB vectors + HNSW + overhead)
- 50-100 GB SSD storage (snapshots + WAL)

### High Availability
- Run coordinator with 3 replicas behind load balancer
- Minimum 2 replicas per shard (prefer 3 for production)
- Use StatefulSets in Kubernetes for stable network identities
- Configure readiness/liveness probes

### Backup Strategy
```bash
# Per-shard snapshot backup (automated via cron)
0 2 * * * rsync -avz /data/shard0/primary/ backup-server:/backups/shard0/$(date +\%Y\%m\%d)/
```

### Security
```bash
# Run with API token
-api-token=your-secret-token

# Use TLS
-tls-cert=/path/to/cert.pem
-tls-key=/path/to/key.pem

# Network policies (Kubernetes)
# Only allow coordinator → shards
# Block external access to shards
```

## Troubleshooting

### Shard Not Registering
```bash
# Check coordinator logs
tail -f /var/log/coordinator.log

# Verify network connectivity
curl http://coordinator:8080/health

# Check shard logs
tail -f /var/log/shard-0-primary.log
```

### High Replication Lag
```bash
# Check shard status
curl http://localhost:8080/admin/cluster_status | jq '.shards[0].nodes'

# Increase sync frequency (shard config)
-sync-interval=500ms

# Check network bandwidth between primary and replica
iperf3 -c replica-host
```

### Query Performance Issues
```bash
# Check query distribution across shards
curl http://localhost:8080/admin/query_stats

# Verify load balancing to replicas
curl http://localhost:8080/admin/cluster_status | jq '.read_strategy'

# Optimize: switch to replica-prefer
curl -X POST http://localhost:8080/admin/set_read_strategy \
  -d '{"strategy": "replica-prefer"}'
```

## Next Steps

- [ ] Set up monitoring with Prometheus + Grafana
- [ ] Configure alerts for shard failures
- [ ] Implement automated backup strategy
- [ ] Plan capacity for expected growth
- [ ] Test failover scenarios
- [ ] Document runbook for common operations

## Support

For issues or questions:
- GitHub Issues: https://github.com/your-org/agentscope-go/issues
- Documentation: See DISTRIBUTED_ARCHITECTURE.md
