# DeepData RC Installation

## Supported runtime

The persistent release candidate supports Linux amd64 only. It is a headless,
single-process, single-node server. Linux arm64 and non-Linux cross-builds are
compile proofs, not persistence-supported release artifacts.

The server exposes tenant-aware HTTP V3 on port 8080 and the matching eleven
unary gRPC methods on port 50051. Clients provide vectors; when `DEEPDATA_EMBEDDER`
names an embedder they may instead send `texts` for fields that bind an
`embedding`. One embedder per process; the default `none` refuses `texts` with
`503 embedder_unavailable`.

## Container quick start

Use an immutable release tag and digest. Replace the placeholders with values
from the candidate manifest; do not deploy a floating tag.

```bash
export DEEPDATA_VERSION='0.2.0-rc.1'
export DEEPDATA_IMAGE="ghcr.io/phenomenon0/deepdata:${DEEPDATA_VERSION}@sha256:<RC_DIGEST>"
export DEEPDATA_API_TOKEN='replace-with-a-long-random-token'
docker compose pull deepdata
docker compose up -d --no-build deepdata
curl -fsS http://localhost:8080/livez
curl -fsS http://localhost:8080/readyz
```

Compose refuses to render without `DEEPDATA_API_TOKEN`; it passes that value as
the server's only `API_TOKEN`. Keep the variable in a protected operator
environment or secret injection mechanism. Do not add `JWT_SECRET` alongside
it: server startup rejects simultaneous static-token and JWT configuration.

The official container runs as numeric UID/GID `10001:10001` and stores its
primary state under `/data/local`. Back up the whole `/data` volume while the
container is stopped.

## Build from source

Go 1.25.12 or newer is required.

```bash
git clone https://github.com/phenomenon0/vectordb.git
cd vectordb
CGO_ENABLED=1 GOOS=linux GOARCH=amd64 go build -trimpath -o deepdata ./cmd/deepdata
export API_TOKEN='replace-with-a-long-random-token'
./deepdata serve
```

Source-built persistent deployments are supported only on Linux amd64. A
production build should be tied to an exact source commit and the evidence
generated for that commit.

## Canonical Python client

```bash
pip install deepdata-client
```

```python
from deepdata import DeepDataClient

with DeepDataClient(
    "http://localhost:8080",
    api_token="replace-with-token",
) as client:
    tenant = client.tenant("org-123")
    tenant.create_collection(
        "docs",
        fields=[{
            "name": "embedding",
            "type": "dense",
            "dim": 3,
            "index": {"type": "hnsw"},
        }],
    )
    tenant.insert("docs", vectors={"embedding": [0.1, 0.2, 0.3]})
    results = tenant.search(
        "docs",
        queries={"embedding": [0.1, 0.2, 0.3]},
        top_k=5,
    )
```

See the [Python SDK guide](../sdk/python/README.md#canonical-v3-quick-start)
for the complete typed sync and async contract.

## Supported configuration

| Variable | Default | RC meaning |
|---|---|---|
| `PORT` | `8080` | HTTP listen port |
| `GRPC_PORT` | `50051` | gRPC listen port; set `0` only when intentionally testing HTTP alone |
| `VECTORDB_MODE` | `local` | Must be `local` in canonical startup |
| `VECTORDB_BASE_DIR` | `~/.vectordb` | Parent used when the data directory is relative or unset |
| `VECTORDB_DATA_DIR` | empty | Exact primary directory if absolute; otherwise relative to the base directory |
| `DEEPDATA_BIND_HOST` | empty (all interfaces with auth) | Optional IP literal to bind both HTTP and gRPC; use `127.0.0.1` for a host-local service |
| `API_TOKEN` | unset | Static bearer token with server-wide administrative access; at least 32 bytes with no surrounding whitespace; configure this or `JWT_SECRET`, never both |
| `JWT_SECRET` | unset | HS256 JWT verification secret; at least 32 bytes with no surrounding whitespace; configure this or `API_TOKEN`, never both |
| `REQUIRE_AUTH` | `0` | Compatibility/defense-in-depth switch set to `1` by shipped deployments; it does not relax the exact-one-credential startup requirement |
| `DEEPDATA_INSECURE_DEV_MODE` | `0` | Set to `1` only for explicit credentialless local development; never for persistent or network-accessible deployments |
| `TRUST_PROXY` | `0` | Trust `X-Forwarded-For`/`X-Real-IP`; enable only when direct backend access is blocked and the proxy overwrites both headers |
| `LOG_LEVEL` | `info` | `debug`, `info`, `warn`, or `error` |
| `LOG_FORMAT` | `json` | `json` or `text` |
| `MAX_COLLECTIONS` | `10000` | Collection limit |
| `DEEPDATA_EMBEDDER` | `none` | Process text embedder for fields that bind an `embedding`: `none`, `ollama`, `openai`, `onnx` (build tag `onnx`) or `hash` (deterministic test embedder, never implicit). `none` answers `texts` with `503 embedder_unavailable`; a configured embedder is probed once at startup and refuses to start when unreachable (`cmd/deepdata/embed_text.go:43`) |
| `DEEPDATA_EMBED_DIM` | `384` | Vector dimension for `hash` and `onnx`; Ollama and OpenAI report their own |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama base URL for `DEEPDATA_EMBEDDER=ollama` |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model; bindings name it as `provider:model` |
| `OPENAI_API_KEY` | unset | Required by `DEEPDATA_EMBEDDER=openai` |
| `ONNX_EMBED_MODEL` / `ONNX_EMBED_TOKENIZER` | `vectordb/models/bge-small-en-v1.5/model.onnx` and the tokenizer.json beside it | ONNX model and tokenizer paths (`scripts/fetch_bge_small.sh`) |
| `ONNX_EMBED_MAX_LEN` | `512` | ONNX tokenizer truncation length |
| `MAX_TENANTS` | `100000` | Tenant limit |
| `TENANT_RPS` | `100` | Per-tenant requests per second |
| `TENANT_BURST` | `100` | Per-tenant burst allowance |
| `AUTH_FAILURE_RPS` | `1` | Failed-auth attempts replenished per peer IP per second |
| `AUTH_FAILURE_BURST` | `5` | Failed-auth attempts allowed per peer IP before temporary throttling |
| `MAX_RATE_LIMIT_KEYS` | `100000` | Maximum keys tracked by each rate limiter; a full failed-auth map rejects unseen peers until capacity recovers |

Terminate TLS at a trusted proxy or ingress and use encrypted storage. Do not
place bearer tokens in URLs or command history.

HTTP and gRPC share a failure-only authentication throttle keyed by normalized
peer IP. Successful authentication does not spend this budget. When
`TRUST_PROXY=1`, the trusted proxy must overwrite forwarding headers and be the
only path to the backend; otherwise clients can forge the throttle key.

Authentication is fail-closed before persistent state is opened: normal
startup requires exactly one of `API_TOKEN` or `JWT_SECRET`; neither and both
are configuration errors. Generate either credential with at least 32 random
bytes (for example, `openssl rand -hex 32`). Query-string bearer tokens are not
accepted. `DEEPDATA_INSECURE_DEV_MODE=1` is an explicit local
development exception, not a production configuration.

## systemd service

Create a dedicated account and state root:

```bash
getent group deepdata >/dev/null || sudo groupadd --system deepdata
getent passwd deepdata >/dev/null || \
  sudo useradd --system --home-dir /var/lib/deepdata \
    --gid deepdata --shell /usr/sbin/nologin deepdata
sudo install -d -o deepdata -g deepdata -m 0750 /var/lib/deepdata
```

Install the exact candidate binary at `/usr/local/bin/deepdata`, then create:

```ini
# /etc/systemd/system/deepdata.service
[Unit]
Description=DeepData single-node vector server
After=network.target

[Service]
Type=simple
User=deepdata
Group=deepdata
WorkingDirectory=/var/lib/deepdata
ExecStart=/usr/local/bin/deepdata serve
Environment=PORT=8080
Environment=GRPC_PORT=50051
Environment=VECTORDB_MODE=local
Environment=VECTORDB_BASE_DIR=/var/lib/deepdata
Environment=VECTORDB_DATA_DIR=local
Environment=REQUIRE_AUTH=1
EnvironmentFile=-/etc/deepdata/deepdata.env
Restart=on-failure
RestartSec=5
UMask=0027
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

Store exactly one of `API_TOKEN` or `JWT_SECRET` in
`/etc/deepdata/deepdata.env`, readable only by root and the service account.
Do not set `DEEPDATA_INSECURE_DEV_MODE` in the service environment. Then start
and verify:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now deepdata
curl -fsS http://localhost:8080/livez
curl -fsS http://localhost:8080/readyz
sudo systemctl show deepdata --property=MainPID --value
sudo test -d /var/lib/deepdata/local
```

The exact primary directory is `/var/lib/deepdata/local`; the stopped backup
boundary is the whole `/var/lib/deepdata` state root. Use the
[offline backup and restore procedure](cookbook.md#offline-backup), including
its mandatory V3/gRPC data assertion.

## Canonical surface check

After authentication is configured, run the repository smoke test from the
exact candidate checkout:

```bash
./tests/smoke_test.sh
```

It exercises HTTP V3 and every unary gRPC method except `Upsert` and `GetDoc`
(`tests/smoke_test.sh:129-201`), restarts the process, and checks that
unsupported legacy routes remain unavailable.

At HEAD the post-restart assertion at `tests/smoke_test.sh:363` still expects
the pre-bc1fa27 PascalCase key `DocCount`, while the server has emitted
`doc_count` since bc1fa27 (`internal/collection/manager.go:186`), so that
`jq -e` check exits non-zero against the current wire form. The script fix
belongs to the tests owner; gates SDK-02 and CI-06 track the script.
