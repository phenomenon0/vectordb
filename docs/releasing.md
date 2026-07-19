# DeepData release process

The canonical candidate version comes from
`internal/releaseinfo/version.txt`. For this tree it is `0.2.0-rc.1`; Python
uses the PEP 440 spelling `0.2.0rc1`. Run:

```bash
python3 scripts/check_version_contract.py
```

before building artifacts. That gate checks the server source, Python package,
Helm chart/image metadata, and Docker image label default.

## Unsigned dry run

The `DeepData RC artifacts` workflow runs on `deepdata-v*` tags and can also be
started manually. Its default behavior is read-only outside the workflow run:
it executes the release-surface tests and builds an unsigned, temporary
artifact bundle containing:

- a reproducible Linux amd64 headless-server archive;
- the `deepdata-client` sdist and wheel;
- the Helm chart and OCI container archive;
- CycloneDX source/container SBOMs;
- Go/Python dependency and build provenance manifests; and
- SHA-256 checksums tied to the workflow commit.

The bundle is retained as a CI artifact for review. A tag-triggered run does
not publish it.

## Explicit publication gate

Publication is a separate manual dispatch with `publish=true`, selected on the
exact `deepdata-v0.2.0-rc.1` tag. The job refuses to proceed unless:

- the tag exactly matches the embedded version;
- an approved root `LICENSE` file exists;
- the repository has a configured `PYPI_API_TOKEN`; and
- GitHub grants the workflow its scoped contents, packages, and OIDC
  permissions.

Only that path keyless-signs the checksum manifest, pushes the container and
Helm chart to GHCR, uploads `deepdata-client` to PyPI, and creates an immutable
GitHub release. Private credentials are never used by dry-run builds.

The selected Python distribution name was unclaimed when checked on
2026-07-19, but availability is not ownership. The project owner must confirm
the name and trusted-publishing configuration immediately before publication.

## Required evidence

Publication must point to the exact commit recorded in
`docs/PRE_RELEASE_STATUS.md` and its evidence report. A green artifact workflow
does not override failed durability, authorization, deployment, correctness,
security, or soak gates, and it cannot substitute for the legal license
decision.
