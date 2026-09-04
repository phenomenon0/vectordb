// Package replication is the experimental single-leader journal transport: a
// leader that serves its own journal and a snapshot of its own state, and a
// follower that keeps a second directory in step by applying that journal in
// LSN order.
//
// It is off unless DEEPDATA_REPLICATION_TOKEN is set, is not part of the RC
// contract, and never elects, promotes, fences, or fails over. See
// docs/distributed-architecture.md.
//
// Tower layer: L3 transports.
package replication
