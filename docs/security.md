# Security guide for the release candidate

DeepData's supported security boundary is intentionally explicit:

| Layer | RC responsibility |
|---|---|
| Request authentication/authorization | Static bearer token or HMAC JWT |
| Tenant/collection scope | Enforced by canonical V3 HTTP and unary gRPC |
| TLS and mTLS | External reverse proxy, ingress, or service mesh |
| Disk and backup encryption | External filesystem, volume, or cloud KMS |
| Network policy and denial-of-service controls | Deployment environment |
| Compliance audit trail | External proxy/platform logging; not an RC server claim |

## Require authentication

Normal startup is fail-closed and requires exactly one authentication mode:
`API_TOKEN` or `JWT_SECRET`. Setting neither or both is a configuration error
detected before the persistent data directory is opened. `REQUIRE_AUTH=1`
remains set in the shipped deployment manifests as defense in depth; it does
not turn credentialless startup into a supported mode. Either credential must
contain at least 32 bytes and must not have leading or trailing whitespace.

`DEEPDATA_INSECURE_DEV_MODE=1` is the sole credentialless escape hatch. Use it
only for an explicit local, disposable development process. Never use it with
persistent data or a listener reachable by another machine.

### Static administrative token

```bash
export API_TOKEN='replace-with-a-long-random-token'
./deepdata serve
```

Clients send:

```http
Authorization: Bearer replace-with-a-long-random-token
```

The static token has full administrative access to every tenant and
collection. Use it for tightly controlled service-to-service deployments, not
as a per-user credential. Generate one with `openssl rand -hex 32`; do not use
the literal example value.

### Tenant JWTs

Configure one sufficiently random HMAC secret and an optional issuer:

```bash
export JWT_SECRET="$(openssl rand -hex 32)"
export JWT_ISSUER='deepdata-production'
./deepdata serve
```

Generate short-lived tokens offline with the same secret and issuer (`deepdata
token` reads `JWT_SECRET` and `JWT_ISSUER` from the environment, same as
`serve`):

```bash
./deepdata token \
  -tenant=acme \
  -permissions=read,write \
  -collections=docs \
  -ttl=24h
```

JWTs are HMAC-signed and carry `tenant_id`, `permissions`, `collections`, an
optional `server_admin` flag, issuer, issued-at, and expiry claims. An empty
collection list means all collections within the token's permitted tenant. An
`admin` JWT is administrative only inside its declared tenant; it never
becomes the server-wide administrator represented by `API_TOKEN`. A
non-empty collection claim continues to restrict an admin JWT, and
tenant-wide list/info operations reject collection-scoped tokens.

`deepdata token -server-admin` mints a JWT with the `server_admin` claim set,
the same global standing as `API_TOKEN`: it crosses every tenant and
collection boundary. Mint it only for genuine operator credentials.

There is no canonical HTTP token-issuance API. Generate and distribute tokens
through a protected operator workflow. Replacing `JWT_SECRET` and restarting
invalidates tokens signed with the old secret.

## Permission map

| Canonical operation | Permission |
|---|---|
| Tenant lifecycle (create / list / update / delete tenant) | `server_admin` |
| Tenant info | `admin` |
| List collections | `read` |
| Create collection | `admin` |
| Delete collection | `admin` |
| Get collection | `read` |
| Search | `read` |
| Server status (`GET /v3/status`) | `read` |
| Insert / atomic batch insert | `write` |
| Delete document | `write` |

For every JWT, including an admin JWT, `tenant_id` must match the tenant in the
V3 path or gRPC request. A non-empty collection claim restricts
collection-level operations and cannot authorize tenant-wide list/info.
HTTP and gRPC use the same decisions; send the bearer token as HTTP
`Authorization` or gRPC `authorization` metadata. Query-string bearer tokens
are rejected so credentials do not enter URLs, access logs, or browser history.
JWT verification accepts HS256 only.

## Transport security is external

The RC server listens on cleartext HTTP/h2c (default port 8080) and cleartext
gRPC (default port 50051). Terminate TLS at a reverse proxy, ingress, or service
mesh and keep the backend listeners on a private network or loopback interface.
Do not rely on legacy `TLS_*` settings as part of the production RC contract.

At the TLS boundary:

- require modern TLS policy and a certificate from your managed CA;
- use mTLS when workload identity is required;
- forward `Authorization` without logging its value;
- cap request sizes and connection rates; and
- expose only the canonical paths that clients need.

## Persistence encryption is external

Canonical snapshots and mutation journals are not application-encrypted.
Place the entire data directory on an encrypted filesystem or volume, manage
keys outside DeepData, and encrypt stopped backups separately. Restrict the
directory to the service account and test recovery with the encryption layer
enabled.

Do not copy individual live journal/snapshot files as a consistency claim.
Use the documented stopped-backup procedure and validate application-specific
queries after restore.

## Operational endpoints

The canonical allowlist includes:

- `/v3/tenants/...`
- `/livez` and `/healthz`
- `/readyz`
- `/metrics`

The health probes (`/healthz`, `/readyz`, `/livez`) are intentionally
unauthenticated so orchestrators can probe them; they reveal availability, so
restrict them with firewall, ingress, or service-mesh policy. `/metrics` exposes
every tenant's usage, so it requires the server-administrator credential when
`REQUIRE_AUTH=1`, not just any authenticated caller
(`cmd/deepdata/server.go:1503-1507`). A durable-store
fault makes `/readyz` return `503`; liveness alone is not proof that persisted
data is safe to serve.

## Production checklist

- [ ] Run the persistent server only on Linux under a dedicated unprivileged account.
- [ ] Configure exactly one of `API_TOKEN` or `JWT_SECRET`; verify that missing or conflicting credentials prevent startup.
- [ ] Keep `REQUIRE_AUTH=1` in deployment configuration and never set `DEEPDATA_INSECURE_DEV_MODE` outside disposable local development.
- [ ] Store `API_TOKEN` or `JWT_SECRET` in a secret manager, not source control or command history.
- [ ] Use short-lived, tenant-scoped JWTs and the smallest required permissions.
- [ ] Terminate TLS externally and isolate cleartext backend ports.
- [ ] Restrict the unauthenticated health probes by network policy; keep `/metrics` behind `REQUIRE_AUTH=1`.
- [ ] Encrypt the data directory and backups outside the process.
- [ ] Protect file ownership and deny a second process access to the data path.
- [ ] Alert on readiness failure, authentication failures at the edge, and restart loops.
- [ ] Test crash recovery and stopped-backup restore before accepting traffic.
- [ ] Do not claim built-in audit logging, at-rest encryption, replication, or mTLS for this RC.
