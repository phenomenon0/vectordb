# Security policy

DeepData is preparing its first defensible single-node release candidate. No
currently published repository tag is a supported DeepData production
release, and the unreleased candidate may still change before its exact-SHA
evidence review is complete.

## Reporting a vulnerability

Do not open a public issue containing exploit details, credentials, private
data, or recovery artifacts. Use the repository's private GitHub security
advisory reporting flow when it is available. If private reporting is not
enabled, contact the repository owner through a private channel and disclose
only enough initially to arrange an encrypted handoff.

Include the affected commit, deployment shape, canonical HTTP or gRPC method,
reproduction steps, impact, and whether persistent data or tenant isolation is
involved. Remove bearer tokens, JWT secrets, user data, and full state files
from logs and examples.

## Candidate security boundary

- Supported production surface: persistent, headless, single-node Linux amd64
  server; tenant-aware HTTP V3; unary `deepdata.v3.DeepData`; static bearer or
  scoped HS256 JWT authentication.
- External controls: TLS termination, encrypted disks/PVCs, secret management,
  ingress/network policy, backup encryption, and compliance audit retention.
- Not supported by this candidate: legacy root/V2 mutations, built-in TLS or
  encryption at rest, distributed/HA operation, advanced LLM/recommendation
  APIs, the web UI, or desktop packaging.

Operators should preserve the complete state root before recovery work and
must never attach secrets or customer data to a public report. See
[`docs/security.md`](docs/security.md) and
[`docs/troubleshooting.md`](docs/troubleshooting.md) for the current runtime
and recovery contracts.
