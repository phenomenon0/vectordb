// Package testutil holds test-only helpers shared across the repository: today
// a loopback httptest server that binds IPv4 127.0.0.1 and skips the test when
// the environment forbids opening a listener.
//
// Tower layer: none — test support, never linked into a shipped binary.
package testutil
