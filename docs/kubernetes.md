# Kubernetes Deployment Guide

## Quick Start with kubectl

### Single-Node Deployment

```yaml
# deepdata-deployment.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: deepdata
  labels:
    app: deepdata
spec:
  serviceName: deepdata
  replicas: 1
  selector:
    matchLabels:
      app: deepdata
  template:
    metadata:
      labels:
        app: deepdata
    spec:
      containers:
        - name: deepdata
          image: ghcr.io/phenomenon0/vectordb:latest
          ports:
            - containerPort: 8080
              name: http
            - containerPort: 50051
              name: grpc
          env:
            - name: PORT
              value: "8080"
            - name: GRPC_PORT
              value: "50051"
            - name: DATA_DIR
              value: /data
            - name: LOG_LEVEL
              value: info
          volumeMounts:
            - name: data
              mountPath: /data
          resources:
            requests:
              cpu: 250m
              memory: 512Mi
            limits:
              cpu: "2"
              memory: 4Gi
          livenessProbe:
            httpGet:
              path: /healthz
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 15
          readinessProbe:
            httpGet:
              path: /readyz
              port: 8080
            initialDelaySeconds: 3
            periodSeconds: 5
          startupProbe:
            httpGet:
              path: /healthz
              port: 8080
            failureThreshold: 30
            periodSeconds: 2
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 10Gi
---
apiVersion: v1
kind: Service
metadata:
  name: deepdata
spec:
  selector:
    app: deepdata
  ports:
    - port: 8080
      targetPort: 8080
      name: http
    - port: 50051
      targetPort: 50051
      name: grpc
  type: ClusterIP
```

```bash
kubectl apply -f deepdata-deployment.yaml
```

### With Authentication

```yaml
# deepdata-secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: deepdata-auth
type: Opaque
stringData:
  jwt-secret: "your-jwt-secret-at-least-32-chars-long"
```

Add to the container env:
```yaml
env:
  - name: JWT_SECRET
    valueFrom:
      secretKeyRef:
        name: deepdata-auth
        key: jwt-secret
  - name: JWT_REQUIRED
    value: "true"
```

### With Encryption at Rest

```yaml
# deepdata-encryption-secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: deepdata-encryption
type: Opaque
stringData:
  passphrase: "your-encryption-passphrase"
```

Add to the container env:
```yaml
env:
  - name: ENCRYPTION_ENABLED
    value: "true"
  - name: ENCRYPTION_PASSPHRASE
    valueFrom:
      secretKeyRef:
        name: deepdata-encryption
        key: passphrase
```

### Ingress

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: deepdata
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "60"
spec:
  rules:
    - host: deepdata.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: deepdata
                port:
                  number: 8080
  tls:
    - hosts:
        - deepdata.example.com
      secretName: deepdata-tls
```

## Resource Sizing Guide

| Dataset Size | Dimension | RAM Request | RAM Limit | CPU | PVC |
|-------------|-----------|-------------|-----------|-----|-----|
| <100K vectors | 384 | 256Mi | 1Gi | 250m | 2Gi |
| 100K-500K | 384 | 1Gi | 4Gi | 500m | 10Gi |
| 500K-1M | 768 | 4Gi | 8Gi | 1 | 20Gi |
| 1M-5M | 768 | 8Gi | 16Gi | 2 | 50Gi |
| 5M+ | 768 | 16Gi+ | 32Gi+ | 4+ | 100Gi+ |

**Formula**: ~(dim × 4 + 200) bytes per vector for HNSW index.
- 384d: ~1.7KB/vector → 1M vectors ≈ 1.7GB RAM
- 768d: ~3.3KB/vector → 1M vectors ≈ 3.3GB RAM

With DiskANN, memory requirements drop significantly since graph data is memory-mapped from disk.

## Backup & Restore

### Manual Snapshot

```bash
# Create snapshot
kubectl exec deepdata-0 -- curl -s http://localhost:8080/export > backup.bin

# Restore
kubectl cp backup.bin deepdata-0:/data/backup.bin
kubectl exec deepdata-0 -- curl -X POST http://localhost:8080/import \
  --data-binary @/data/backup.bin
```

### CronJob for Automated Backups

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: deepdata-backup
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: backup
              image: curlimages/curl:latest
              command:
                - sh
                - -c
                - |
                  curl -s http://deepdata:8080/export > /backup/deepdata-$(date +%Y%m%d).bin
                  # Keep last 7 days
                  find /backup -name "deepdata-*.bin" -mtime +7 -delete
              volumeMounts:
                - name: backup
                  mountPath: /backup
          volumes:
            - name: backup
              persistentVolumeClaim:
                claimName: deepdata-backup
          restartPolicy: OnFailure
```

## Monitoring

### Prometheus ServiceMonitor

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: deepdata
spec:
  selector:
    matchLabels:
      app: deepdata
  endpoints:
    - port: http
      path: /metrics
      interval: 15s
```

### OpenTelemetry Collector

To export traces to an OTEL collector:
```yaml
env:
  - name: OTEL_EXPORTER_OTLP_ENDPOINT
    value: "http://otel-collector:4317"
```

### Key Metrics to Alert On

| Metric | Warning | Critical | Description |
|--------|---------|----------|-------------|
| `deepdata_search_duration_seconds` (P99) | >100ms | >500ms | Query latency |
| `deepdata_insert_duration_seconds` (P99) | >50ms | >200ms | Insert latency |
| `deepdata_vectors_total` | - | >capacity×0.9 | Approaching capacity |
| `deepdata_search_requests_total` (error rate) | >1% | >5% | Error rate |
| `deepdata_memory_bytes` | >limit×0.8 | >limit×0.9 | Memory pressure |

See [Grafana Dashboard](grafana/) for a pre-built dashboard with 40+ panels and alerting rules.

## Troubleshooting

### Pod stuck in Pending
Check PVC binding: `kubectl describe pvc data-deepdata-0`

### OOMKilled
Increase memory limits. Check vector count vs RAM sizing guide above. Consider DiskANN or PQ quantization to reduce memory footprint.

### Slow startup with large dataset
Increase `startupProbe.failureThreshold`. Loading 1M+ vectors can take 30-60 seconds.

### Data loss after pod restart
Ensure PVC is `ReadWriteOnce` and the StatefulSet `volumeClaimTemplates` is configured. Data should persist across restarts.

### gRPC not reachable
Ensure the Service exposes port 50051 and your Ingress or load balancer routes gRPC traffic correctly. For gRPC through nginx ingress, add:
```yaml
annotations:
  nginx.ingress.kubernetes.io/backend-protocol: "GRPC"
```
