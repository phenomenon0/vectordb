// Command gentoken mints JWT bearer tokens for DeepData's multi-tenant
// authentication using internal/security. It signs claims offline and never
// contacts a server, so it is a development and operator helper only.
//
// Tower layer: L3 transports — it produces the credentials they verify.
package main
