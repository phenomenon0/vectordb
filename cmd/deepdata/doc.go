// Command deepdata is the single-node DeepData server: the L3 transports
// (HTTP /v3 and gRPC deepdata.v3) over the L1 engine in internal/collection
// and its L0 durable store, plus the runtime that authenticates and rate-limits
// them. It also carries the offline legacy-snapshot migration path and the
// routes subcommand that prints the L2 operation table.
package main
