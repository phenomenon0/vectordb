# Kubernetes Deployment Guide

## Quick Start with kubectl

### Single-Node Deployment

```yaml
# vectordb-deployment.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: vectordb
  labels:
    app: vectordb
spec:
  serviceName: vectordb
  replicas: 1
  selector:
    matchLabels:
      app: vectordb
  template:
    metadata:
      labels:
        app: vectordb
    spec:
      containers:
        - name: vectordb
          image: ghcr.io/phenomenon0/vectordb:latest
          ports:
            - containerPort: 8080
              name: http
          env:
            - name: PORT
              value: "8080"
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
  name: vectordb
spec:
  selector:
    app: vectordb
  ports:
    - port: 8080
      targetPort: 8080
      name: http
  type: ClusterIP
```

```bash
kubectl apply -f vectordb-deployment.yaml
```

### With Authentication

```yaml
# vectordb-secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: vectordb-auth
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
        name: vectordb-auth
        key: jwt-secret
  - name: JWT_REQUIRED
    value: "true"
```

### Ingress

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: vectordb
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "60"
spec:
  rules:
    - host: vectordb.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: vectordb
                port:
                  number: 8080
  tls:
    - hosts:
        - vectordb.example.com
      secretName: vectordb-tls
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

## Backup & Restore

### Manual Snapshot

```bash
# Create snapshot
kubectl exec vectordb-0 -- curl -s http://localhost:8080/export > backup.bin

# Restore
kubectl cp backup.bin vectordb-0:/data/backup.bin
kubectl exec vectordb-0 -- curl -X POST http://localhost:8080/import \
  --data-binary @/data/backup.bin
```

### CronJob for Automated Backups

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: vectordb-backup
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
                  curl -s http://vectordb:8080/export > /backup/vectordb-$(date +%Y%m%d).bin
                  # Keep last 7 days
                  find /backup -name "vectordb-*.bin" -mtime +7 -delete
              volumeMounts:
                - name: backup
                  mountPath: /backup
          volumes:
            - name: backup
              persistentVolumeClaim:
                claimName: vectordb-backup
          restartPolicy: OnFailure
```

## Monitoring

### Prometheus ServiceMonitor

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: vectordb
spec:
  selector:
    matchLabels:
      app: vectordb
  endpoints:
    - port: http
      path: /metrics
      interval: 15s
```

### Key Metrics to Alert On

| Metric | Warning | Critical | Description |
|--------|---------|----------|-------------|
| `vectordb_query_latency_p99` | >100ms | >500ms | Query latency |
| `vectordb_insert_latency_p99` | >50ms | >200ms | Insert latency |
| `vectordb_wal_bytes` | >50MB | >100MB | WAL size growth |
| `vectordb_active_vectors` | - | >capacity×0.9 | Approaching capacity |
| `vectordb_error_rate` | >1% | >5% | Error rate |

## Troubleshooting

### Pod stuck in Pending
Check PVC binding: `kubectl describe pvc data-vectordb-0`

### OOMKilled
Increase memory limits. Check vector count vs RAM sizing guide above.

### Slow startup with large dataset
Increase `startupProbe.failureThreshold`. Loading 1M+ vectors can take 30-60 seconds.

### Data loss after pod restart
Ensure PVC is `ReadWriteOnce` and the StatefulSet `volumeClaimTemplates` is configured. Data should persist across restarts.
