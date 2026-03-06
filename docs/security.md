# Security Guide

## Authentication

### JWT Authentication

VectorDB uses JWT tokens for multi-tenant authentication.

#### Enable Authentication

```bash
# Set a strong secret (32+ characters)
export JWT_SECRET="your-very-long-secret-key-at-least-32-chars"
export JWT_REQUIRED=true
./vectordb-server
```

#### Generate Tokens

**Using the CLI:**
```bash
# Admin token
vectordb-cli gentoken --tenant admin --permissions admin --secret "$JWT_SECRET"

# Read-only token
vectordb-cli gentoken --tenant viewer --permissions read --secret "$JWT_SECRET"

# Scoped to specific collections
vectordb-cli gentoken --tenant partner --permissions read,write \
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
  "iss": "vectordb",
  "exp": 1735689600
}
```

### API Key Authentication

For simpler setups, use a static API token:
```bash
export API_TOKEN="your-api-key"
./vectordb-server
```

Clients include it as a header:
```bash
curl -H "Authorization: Bearer your-api-key" http://localhost:8080/health
```

## TLS/HTTPS

### Self-Signed Certificate (Development)

VectorDB can generate self-signed certificates:
```bash
export TLS_ENABLED=true
export TLS_CERT_FILE=cert.pem
export TLS_KEY_FILE=key.pem
export TLS_AUTO_CERT=true  # auto-generate if files don't exist
./vectordb-server
```

### Production TLS

Use real certificates from Let's Encrypt or your CA:
```bash
export TLS_ENABLED=true
export TLS_CERT_FILE=/etc/vectordb/tls/tls.crt
export TLS_KEY_FILE=/etc/vectordb/tls/tls.key
export TLS_MIN_VERSION=1.2
./vectordb-server
```

### Mutual TLS (mTLS)

For service-to-service authentication:
```bash
export TLS_ENABLED=true
export TLS_CERT_FILE=server.crt
export TLS_KEY_FILE=server.key
export TLS_CLIENT_CA=ca.crt
export TLS_CLIENT_AUTH=require
./vectordb-server
```

## Authorization (RBAC)

### Permission Model

| Permission | Allows |
|------------|--------|
| `read` | Query, health check |
| `write` | Insert, delete, upsert |
| `admin` | All operations + tenant management |

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
# Global rate limit
export TENANT_RPS=100

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

VectorDB supports AES-256-GCM and ChaCha20-Poly1305 encryption:

```bash
export ENCRYPTION_ENABLED=true
export ENCRYPTION_PASSPHRASE="your-encryption-passphrase"
export ENCRYPTION_ALGORITHM=aes-256-gcm  # or chacha20-poly1305
./vectordb-server
```

Key derivation uses Argon2id with configurable parameters.

## Audit Logging

All write and admin operations are logged:

```bash
export AUDIT_LOG=true
export AUDIT_LOG_FILE=/var/log/vectordb/audit.log
./vectordb-server
```

Audit log format:
```json
{
  "timestamp": "2025-12-01T10:30:00Z",
  "tenant_id": "customer-1",
  "action": "insert",
  "collection": "docs",
  "doc_id": "abc123",
  "ip": "10.0.0.5"
}
```

## Network Security Checklist

- [ ] Enable TLS in production (`TLS_ENABLED=true`)
- [ ] Use strong JWT secret (32+ characters, randomly generated)
- [ ] Enable `JWT_REQUIRED=true` to block unauthenticated access
- [ ] Set per-tenant rate limits to prevent abuse
- [ ] Set storage quotas for multi-tenant deployments
- [ ] Enable audit logging for compliance
- [ ] Restrict network access (firewall, security groups)
- [ ] Use mTLS for service-to-service communication
- [ ] Rotate JWT secrets periodically
- [ ] Monitor `/metrics` endpoint for anomalies
