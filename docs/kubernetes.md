# Kubernetes Deployment Guide

This guide deploys the Linux amd64, persistent, single-node release candidate.
Keep `replicas: 1`; the RC has no supported clustering or failover path.

## StatefulSet

Replace `<RC_VERSION>`, `<RC_DIGEST>`, storage class, and token before applying.
The image reference must resolve to the exact candidate image.

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: deepdata-auth
type: Opaque
stringData:
  api-token: "replace-with-a-long-random-token"
---
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
      nodeSelector:
        kubernetes.io/os: linux
        kubernetes.io/arch: amd64
      terminationGracePeriodSeconds: 60
      securityContext:
        runAsNonRoot: true
        runAsUser: 10001
        runAsGroup: 10001
        fsGroup: 10001
        fsGroupChangePolicy: OnRootMismatch
        seccompProfile:
          type: RuntimeDefault
      containers:
        - name: deepdata
          image: "ghcr.io/phenomenon0/deepdata:<RC_VERSION>@sha256:<RC_DIGEST>"
          imagePullPolicy: IfNotPresent
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop: ["ALL"]
          ports:
            - name: http
              containerPort: 8080
            - name: grpc
              containerPort: 50051
          env:
            - name: PORT
              value: "8080"
            - name: GRPC_PORT
              value: "50051"
            - name: VECTORDB_MODE
              value: local
            - name: VECTORDB_BASE_DIR
              value: /data
            - name: VECTORDB_DATA_DIR
              value: local
            - name: REQUIRE_AUTH
              value: "1"
            - name: API_TOKEN
              valueFrom:
                secretKeyRef:
                  name: deepdata-auth
                  key: api-token
            - name: LOG_FORMAT
              value: json
          volumeMounts:
            - name: data
              mountPath: /data
            - name: tmp
              mountPath: /tmp
          resources:
            requests:
              cpu: 250m
              memory: 512Mi
            limits:
              cpu: "2"
              memory: 4Gi
          startupProbe:
            httpGet:
              path: /livez
              port: http
            failureThreshold: 60
            periodSeconds: 2
          livenessProbe:
            httpGet:
              path: /livez
              port: http
            periodSeconds: 15
            timeoutSeconds: 3
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /readyz
              port: http
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 3
      volumes:
        - name: tmp
          emptyDir: {}
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        storageClassName: YOUR_STORAGE_CLASS
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
    - name: http
      port: 8080
      targetPort: http
    - name: grpc
      port: 50051
      targetPort: grpc
  type: ClusterIP
```

```bash
kubectl apply -f deepdata.yaml
kubectl rollout status statefulset/deepdata --timeout=5m
kubectl exec deepdata-0 -- curl -fsS http://localhost:8080/livez
kubectl exec deepdata-0 -- curl -fsS http://localhost:8080/readyz
```

The official image and pod security context use the stable numeric identity
`10001:10001`. Confirm that the storage driver honors `fsGroup`; otherwise
pre-provision ownership outside the application pod. Do not run DeepData as
root or grant privileged mode.

`VECTORDB_DATA_DIR=local` resolves below `VECTORDB_BASE_DIR=/data`, so the exact
primary directory is `/data/local`. The backup boundary is the whole `/data`
PVC, not selected files beneath the primary directory.

## Helm invariants

The chart must render the same contract as the manifest above:

- exactly one replica and a `Recreate` strategy;
- an immutable image digest and Linux/amd64 node selection;
- UID/GID/fsGroup `10001`, dropped capabilities, read-only root filesystem,
  and a writable `/tmp` `emptyDir`;
- `/data/local` as primary state on a persistent `ReadWriteOnce` volume;
- both HTTP and gRPC ports;
- public liveness/readiness probes; and
- `REQUIRE_AUTH=1` with an existing Secret.

By default the chart reads `API_TOKEN` from the existing Secret
`deepdata-auth`, key `api-token`. Create that Secret before installing (the
manifest above shows its required shape). To use an existing JWT Secret
instead, set `auth.existingSecret`, `auth.existingSecretType=jwtSecret`, and
the matching `auth.existingSecretKey`.

The chart deliberately does not accept secret material through Helm values,
because Helm would retain it in the release record. It rejects a missing
Secret name or key and any auth type other than `apiToken` or `jwtSecret`.

Replace the digest placeholder below with the candidate's lowercase 64-digit
SHA-256 digest. Set `persistence.verifiedPOSIXSemantics=true` only after
verifying that the selected volume supports advisory locks, atomic
same-directory rename, file `fsync`, and directory `fsync`.

```bash
export DEEPDATA_IMAGE_REPOSITORY='ghcr.io/phenomenon0/deepdata'
export DEEPDATA_IMAGE_DIGEST='sha256:<64-lowercase-hex-digest>'
export DEEPDATA_API_TOKEN='replace-with-a-long-random-token'

kubectl create secret generic deepdata-auth \
  --from-literal="api-token=$DEEPDATA_API_TOKEN" \
  --dry-run=client -o yaml | kubectl apply -f -

helm lint deploy/helm/deepdata --strict \
  --set-string image.repository="$DEEPDATA_IMAGE_REPOSITORY" \
  --set-string image.digest="$DEEPDATA_IMAGE_DIGEST" \
  --set persistence.verifiedPOSIXSemantics=true \
  --set-string auth.existingSecret=deepdata-auth \
  --set-string auth.existingSecretType=apiToken \
  --set-string auth.existingSecretKey=api-token

helm template deepdata deploy/helm/deepdata \
  --set-string image.repository="$DEEPDATA_IMAGE_REPOSITORY" \
  --set-string image.digest="$DEEPDATA_IMAGE_DIGEST" \
  --set persistence.verifiedPOSIXSemantics=true \
  --set-string auth.existingSecret=deepdata-auth \
  --set-string auth.existingSecretType=apiToken \
  --set-string auth.existingSecretKey=api-token \
  > /tmp/deepdata-rendered.yaml
kubectl apply --dry-run=server -f /tmp/deepdata-rendered.yaml

helm upgrade --install deepdata deploy/helm/deepdata \
  --set-string image.repository="$DEEPDATA_IMAGE_REPOSITORY" \
  --set-string image.digest="$DEEPDATA_IMAGE_DIGEST" \
  --set persistence.verifiedPOSIXSemantics=true \
  --set-string auth.existingSecret=deepdata-auth \
  --set-string auth.existingSecretType=apiToken \
  --set-string auth.existingSecretKey=api-token \
  --wait --timeout=5m
```

Do not set `DEEPDATA_INSECURE_DEV_MODE` in a pod. It exists only for explicit
credentialless local development and is not a supported persistent deployment
mode.

## Ingress and storage security

Terminate TLS at a trusted ingress or load balancer. Use encrypted PVCs and
restrict network access to intended clients. HTTP and gRPC require separate
ingress routing rules unless the selected ingress supports both protocols on a
shared listener.

### Network isolation

The Helm chart renders a default-deny `NetworkPolicy` for the DeepData pod
(`networkPolicy.enabled=true`): ingress is limited to the advertised HTTP and
gRPC ports, and all pod egress is denied. The RC pod makes no outbound
connections; if `telemetry.enabled=true`, you must also set
`networkPolicy.egressTo` to the OTLP exporter's CIDR or the chart refuses to
render.

Policy enforcement requires a CNI with NetworkPolicy support (Calico, Cilium,
OVN-Kubernetes, etc.). Non-enforcing CNIs such as kindnet accept the manifest
but apply no traffic policy, so the operator must provide equivalent isolation
externally; `networkPolicy.enabled=false` suppresses the chart policy when a
cluster-level mechanism is used. Inspect the rendered contract with:

```bash
helm template <release> deploy/helm/deepdata \
  --set persistence.verifiedPOSIXSemantics=true \
  --set-string image.digest=sha256:REPLACE_WITH_YOUR_DIGEST \
  | grep -A 40 'kind: NetworkPolicy'
```

## Offline backup

Stop the only pod before capturing the whole PVC:

```bash
kubectl scale statefulset/deepdata --replicas=0
kubectl wait --for=delete pod/deepdata-0 --timeout=180s
```

Create a CSI snapshot of `data-deepdata-0`:

```yaml
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: deepdata-state-20260718
spec:
  volumeSnapshotClassName: YOUR_SNAPSHOT_CLASS
  source:
    persistentVolumeClaimName: data-deepdata-0
```

```bash
kubectl apply -f deepdata-snapshot.yaml
kubectl wait volumesnapshot/deepdata-state-20260718 \
  --for=jsonpath='{.status.readyToUse}'=true --timeout=10m
kubectl scale statefulset/deepdata --replicas=1
kubectl rollout status statefulset/deepdata --timeout=5m
```

If CSI snapshots are unavailable, keep the StatefulSet at zero and use a
one-shot maintenance pod that mounts both the state PVC and a backup PVC. Copy
`/data/.` recursively with ownership, permissions, and hidden files preserved.

## Offline restore

Restore a snapshot into a new PVC; never merge it into the current PVC:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: deepdata-restore-candidate-20260718
spec:
  storageClassName: YOUR_STORAGE_CLASS
  dataSource:
    apiGroup: snapshot.storage.k8s.io
    kind: VolumeSnapshot
    name: deepdata-state-20260718
  accessModes: ["ReadWriteOnce"]
  resources:
    requests:
      storage: 10Gi
```

The example StatefulSet uses `volumeClaimTemplates`, so ordinal zero remains
bound to `data-deepdata-0`. It cannot attach the candidate by name. Do not
delete the original PVC or assume scaling up will select the candidate.

Production cutover therefore requires a rehearsed CSI/operator procedure or a
manifest with an explicit existing-claim setting. Before accepting cutover,
start the candidate as the only writer and require an executable assertion that
checks the expected V3 tenant, schema, document count, representative search,
and gRPC result. Retain the original PVC and snapshot until those checks pass.

## Monitoring

Prometheus metrics are available at `/metrics`; liveness is `/livez` and
readiness is `/readyz`. Alert on readiness failures, request error rate,
high-latency searches, memory pressure, and PVC capacity. A ready response is
not a substitute for V3/gRPC data validation after restore or upgrade.
