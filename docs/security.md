# Security Guide

## Authentication

### JWT Authentication

DeepData uses JWT tokens for multi-tenant authentication.

#### Enable Authentication

```bash
export JWT_SECRET="your-very-long-secret-key-at-least-32-chars"
export JWT_REQUIRED=true
./deepdata serve
```

#### Generate Tokens

**Using the CLI:**
```bash
# Admin token
deepdata-cli gentoken --tenant admin --permissions admin --secret "$JWT_SECRET"

# Read-only token
deepdata-cli gentoken --tenant viewer --permissions read --secret "$JWT_SECRET"

# Scoped to specific collections
deepdata-cli gentoken --tenant partner --permissions read,write \
  --collections public,shared --expires 168h --secret "$JWT_SECRET"
```

**Using the Admin API:**
```bash
curl -X POST http://localhost:8080/admin/tokens \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "customer-1",
    "permissions": ["read", "write"],
    "collections": ["customer-1-docs"],
    "expires_in": "720h"
  }'
```

#### Token Claims

```json
{
  "tenant_id": "customer-1",
  "permissions": ["read", "write"],
  "collections": ["docs", "images"],
  "iss": "deepdata",
  "exp": 1735689600
}
```

### API Key Authentication

For simpler setups, use a static API token:
```bash
export API_TOKEN="your-api-key"
./deepdata serve
```

Clients include it as a header:
```bash
curl -H "Authorization: Bearer your-api-key" http://localhost:8080/health
```

## TLS/HTTPS

### Self-Signed Certificate (Development)

DeepData can generate self-signed certificates:
```bash
export TLS_ENABLED=true
export TLS_CERT_FILE=cert.pem
export TLS_KEY_FILE=key.pem
export TLS_AUTO_CERT=true  # auto-generate if files don't exist
./deepdata serve
```

### Production TLS

Use real certificates from Let's Encrypt or your CA:
```bash
export TLS_ENABLED=true
export TLS_CERT_FILE=/etc/deepdata/tls/tls.crt
export TLS_KEY_FILE=/etc/deepdata/tls/tls.key
export TLS_MIN_VERSION=1.2
./deepdata serve
```

### Mutual TLS (mTLS)

For service-to-service authentication:
```bash
export TLS_ENABLED=true
export TLS_CERT_FILE=server.crt
export TLS_KEY_FILE=server.key
export TLS_CLIENT_CA=ca.crt
export TLS_CLIENT_AUTH=require
./deepdata serve
```

## Authorization (RBAC)

### Permission Model

| Permission | Allows |
|------------|--------|
| `read` | Query, scroll, health check |
| `write` | Insert, delete, upsert |
| `admin` | All operations + tenant management + audit access |

### Collection-Level Access

Tokens can be scoped to specific collections:
```json
{
  "tenant_id": "partner",
  "permissions": ["read", "write"],
  "collections": ["shared-docs", "public"]
}
```

Requests to other collections return `403 Forbidden`.

### Per-Tenant Rate Limiting

```bash
# Global rate limit (all tenants)
export TENANT_RPS=100
export TENANT_BURST=100

# Or set per-tenant via admin API
curl -X POST http://localhost:8080/admin/ratelimit/set \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"tenant_id": "customer-1", "rps": 50}'
```

### Storage Quotas

```bash
curl -X POST http://localhost:8080/admin/quota/set \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"tenant_id": "customer-1", "max_vectors": 100000, "max_bytes": 1073741824}'
```

## Encryption at Rest

DeepData supports AES-256-GCM and ChaCha20-Poly1305 encryption for all persisted data:

```bash
export ENCRYPTION_ENABLED=true
export ENCRYPTION_PASSPHRASE="your-encryption-passphrase"
export ENCRYPTION_ALGORITHM=aes-256-gcm  # or chacha20-poly1305
./deepdata serve
```

- **AES-256-GCM**: Default. Hardware-accelerated on modern CPUs (AES-NI).
- **ChaCha20-Poly1305**: Better performance on hardware without AES-NI.
- **Key derivation**: Argon2id with configurable parameters. Keys are never stored directly — they're derived from the passphrase at startup.

Encrypted files use a binary header with magic bytes `VDBE`, making it easy to identify encrypted vs. plaintext data files.

## Audit Logging

Track all write, admin, and auth operations:

```bash
export AUDIT_LOG=true
export AUDIT_LOG_FILE=/var/log/deepdata/audit.log
./deepdata serve
```

Audit log format:
```json
{
  "timestamp": "2026-03-01T10:30:00Z",
  "tenant_id": "customer-1",
  "action": "insert",
  "collection": "docs",
  "doc_id": "abc123",
  "outcome": "success",
  "ip": "10.0.0.5"
}
```

25+ event types across categories: authentication, RBAC, vector operations, admin actions, cluster events, and system events.

## gRPC Security

gRPC on port 50051 inherits the same JWT/TLS configuration as HTTP. When TLS is enabled, gRPC uses the same certificate. Requests without valid tokens are rejected at the interceptor level.

## Network Security Checklist

- [ ] Enable TLS in production (`TLS_ENABLED=true`)
- [ ] Use strong JWT secret (32+ characters, randomly generated)
- [ ] Enable `JWT_REQUIRED=true` to block unauthenticated access
- [ ] Set per-tenant rate limits to prevent abuse
- [ ] Set storage quotas for multi-tenant deployments
- [ ] Enable encryption at rest for sensitive data
- [ ] Enable audit logging for compliance
- [ ] Restrict network access (firewall, security groups)
- [ ] Use mTLS for service-to-service communication
- [ ] Rotate JWT secrets periodically
- [ ] Monitor `/metrics` endpoint for anomalies
- [ ] Review audit logs regularly
