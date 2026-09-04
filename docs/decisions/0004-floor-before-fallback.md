# 0004 — Score floor applies before the fallback decision

- Date: 2026-08-21
- Status: accepted
- Supersedes: the pre-commit draft FallbackParams doc comment that called the decision "independent of ScoreFloor" (tasks/lessons.md:89-91); no committed revision carried it — bde4f94 landed the corrected wording
- Evidence: bde4f94 (feat(collection): agent retrieval contract across engine, HTTP, gRPC, SDK, MCP); internal/collection/types.go:412-426 (FallbackParams); tasks/lessons.md:87-96

## Context
A two-field request runs two post-processing stages on the primary answer: score_floor filters it, then the fallback ladder decides whether to search the secondary field. The draft doc comment said the two were independent while the code applied the floor first.

## Decision
The floor runs first. "No confident result" — zero hits, every hit wiped out by score_floor, or (with threshold > 0) a best score worse than the threshold in the primary field's own score direction — routes to the secondary field. The two mechanisms compose; neither bypasses the other.

## Consequences
The doc comment at internal/collection/types.go:412-421 states the order explicitly. Any later stage that consumes the answer (usage boost; the per-result confidence tracked by CTL-04) follows the same rule: state the order in the doc comment and decide on the post-filter answer (lessons.md:94-96).
