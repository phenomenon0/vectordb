package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	vcollection "github.com/phenomenon0/vectordb/internal/collection"
	"github.com/phenomenon0/vectordb/internal/logging"
	"github.com/phenomenon0/vectordb/internal/replication"
)

// replicationTokenEnv enables and authenticates the node surface. There is no
// boolean toggle beside it on purpose: an operator cannot turn replication on
// and forget the credential, because the credential IS the switch.
const replicationTokenEnv = "DEEPDATA_REPLICATION_TOKEN"

// canonicalReplicationSurface mounts the leader's node routes in front of the
// client handler when the node credential is set, and returns next unchanged
// when it is not.
//
// In front, not inside: the client surface is an allowlist the contract tests
// pin, and every route on it is authorized against a tenant. These routes are
// authorized against a different credential and export the whole store, so they
// stay outside that middleware rather than becoming an exception inside it.
// They also stream, and the client chain's request timeout would cut a tail.
func canonicalReplicationSurface(next http.Handler, collections *CollectionHTTPServer, statePath string, logger *logging.Logger) (http.Handler, error) {
	token := os.Getenv(replicationTokenEnv)
	if token == "" {
		return next, nil
	}
	store := collections.DurableStore()
	if store == nil {
		// A memory-only process has no journal to stream. Failing here rather
		// than serving an empty surface keeps a misconfigured leader from
		// looking healthy to a follower that will never receive a record.
		return nil, fmt.Errorf("%s is set but this process has no durable store to replicate", replicationTokenEnv)
	}
	// The spool sits beside the state it copies, so a snapshot transfer cannot
	// succeed on a filesystem that has no room for the state itself.
	spool := filepath.Dir(statePath)
	node, err := replication.NewLeaderHandler(store, replication.LeaderConfig{
		Token:    token,
		SpoolDir: spool,
		Logger:   log.New(replicationLogWriter{logger}, "", 0),
	})
	if err != nil {
		return nil, err
	}
	logger.Warn("node replication surface enabled; it exports every tenant's state to any caller holding the node token",
		"prefix", replication.PathPrefix, "spool_dir", spool)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, replication.PathPrefix) {
			node.ServeHTTP(w, r)
			return
		}
		next.ServeHTTP(w, r)
	}), nil
}

// replicationLogWriter routes the node surface's transport faults into the
// server's own structured logger instead of stderr, so they land in the same
// place an operator is already reading.
type replicationLogWriter struct{ logger *logging.Logger }

func (w replicationLogWriter) Write(p []byte) (int, error) {
	w.logger.Warn(strings.TrimRight(string(p), "\n"))
	return len(p), nil
}

// bindReplicaReadOnly re-applies the replica binding a directory carries.
//
// DurableStore.IsReplica is in-memory, so a plain `deepdata serve` learns a
// directory is a replica the only way it can: from the marker the follower
// left beside the store. Serve calls this on every open, and there is no flag
// to forget -- the data directory is the evidence.
//
// A marker that cannot be honored is a startup failure, never a shrug. Serving
// the directory as an ordinary store is the one outcome that must not happen:
// a single local write consumes the LSN the leader's next record needs and
// forks the two histories at the same position.
func bindReplicaReadOnly(collections *CollectionHTTPServer, basePath string) error {
	leaderID, isReplica, err := replication.ReplicaLeaderID(basePath)
	if err != nil {
		return err
	}
	if !isReplica {
		return nil
	}
	store := collections.DurableStore()
	if store == nil {
		return fmt.Errorf("%q is a read replica of leader %x but this process has no durable store to serve it read-only", basePath, leaderID)
	}
	if err := store.MakeReplica(leaderID); err != nil {
		return fmt.Errorf("serve %q read-only as a replica of leader %x: %w", basePath, leaderID, err)
	}
	return nil
}

// runReplicate is the `deepdata replicate` subcommand: keep a local replica
// directory in step with a leader.
//
// It only syncs, and it holds the directory for as long as it does: the
// collection store takes an exclusive lock, so a `deepdata serve` against the
// same path is refused with "collection store is already open" while this
// command runs. A replica directory is therefore either tailing its leader or
// being served, never both at once, and switching between them means stopping
// one process and starting the other. Serving it is a plain `deepdata serve`
// against the same path: that process finds the replica marker, answers reads
// normally, refuses every write with a 403, and reports read_only on /readyz so
// a load balancer stops sending it writes.
func runReplicate(args []string, logger *logging.Logger) int {
	fs := flag.NewFlagSet("replicate", flag.ExitOnError)
	leaderURL := fs.String("leader", "", "leader base URL, e.g. http://leader.internal:8080")
	retry := fs.Duration("retry", 5*time.Second, "wait before reconnecting after the stream drops")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *leaderURL == "" {
		fmt.Fprintln(os.Stderr, "replicate: --leader is required")
		return 2
	}
	token := os.Getenv(replicationTokenEnv)
	if token == "" {
		fmt.Fprintf(os.Stderr, "replicate: %s must be set to the leader's node token\n", replicationTokenEnv)
		return 2
	}

	cfg, errs := loadServerConfig(nil, os.Getenv)
	if len(errs) > 0 {
		fmt.Fprintf(os.Stderr, "replicate: %s\n", strings.Join(errs, "; "))
		return 2
	}
	// The same path the server would open, so a replica directory and the
	// leader directory are configured identically and an operator can promote
	// one by changing the subcommand, not the layout.
	base := cfg.IndexPath + ".collections"
	if err := os.MkdirAll(filepath.Dir(base), 0o750); err != nil {
		logger.Error("cannot create the replica state directory", "path", filepath.Dir(base), "error", err)
		return 1
	}

	// No whole-request timeout: a follow stream is open-ended by design, and a
	// Client.Timeout would cut it at a fixed interval forever.
	follower := &replication.Follower{LeaderURL: *leaderURL, Token: token, Client: &http.Client{}}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	store, err := follower.Open(ctx, base, base)
	if err != nil {
		logger.Error("cannot open a replica of this leader", "path", base, "leader", *leaderURL, "error", err)
		return 1
	}
	defer func() {
		if closeErr := store.Close(); closeErr != nil {
			logger.Error("failed to close the replica store", "error", closeErr)
		}
	}()
	logger.Info("replica open", "path", base, "leader", *leaderURL, "applied_lsn", store.ReplicaCursor().LSN)

	for {
		err := follower.Follow(ctx, store)
		switch {
		case ctx.Err() != nil:
			logger.Info("replication stopped", "applied_lsn", store.ReplicaCursor().LSN)
			return 0
		case errors.Is(err, replication.ErrResyncRequired):
			// Deliberately terminal. Recovering means discarding this
			// directory, and that is an operator's decision -- it may be the
			// one a read fleet is serving from.
			logger.Error("this replica is behind the leader's retained journal; discard the replica directory and start again to re-bootstrap",
				"path", base, "applied_lsn", store.ReplicaCursor().LSN, "error", err)
			return 1
		case errors.Is(err, vcollection.ErrJournalStoreMismatch):
			logger.Error("this replica does not belong to that leader", "path", base, "leader", *leaderURL, "error", err)
			return 1
		}
		logger.Warn("replication stream dropped; reconnecting", "applied_lsn", store.ReplicaCursor().LSN, "retry_in", *retry, "error", err)
		select {
		case <-ctx.Done():
			return 0
		case <-time.After(*retry):
		}
	}
}
