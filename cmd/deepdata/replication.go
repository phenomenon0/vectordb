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
	"sync"
	"sync/atomic"
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
func canonicalReplicationSurface(next http.Handler, collections *CollectionHTTPServer, statePath, token string, logger *logging.Logger) (http.Handler, error) {
	if token == "" {
		return next, nil
	}
	set := collections.Stores()
	if set == nil {
		// A memory-only process has no journal to stream. Failing here rather
		// than serving an empty surface keeps a misconfigured leader from
		// looking healthy to a follower that will never receive a record.
		return nil, fmt.Errorf("%s is set but this process has no durable store to replicate", replicationTokenEnv)
	}
	// The spool sits beside the state it copies, so a snapshot transfer cannot
	// succeed on a filesystem that has no room for the state itself.
	spool := filepath.Dir(statePath)
	node, err := replication.NewTenantLeaderHandler(replication.LeaderConfig{
		Token:    token,
		SpoolDir: spool,
		Epoch:    func(id string) (replication.Epoch, error) { return replication.ReadEpoch(set.Base(id)) },
		Logger:   log.New(replicationLogWriter{logger}, "", 0),
	}, set.Tenants, func(id string) (replication.Source, bool) { return set.Store(id) })
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

// bindReplicaReadOnly re-applies the replica binding each tenant's directory
// carries.
//
// DurableStore.IsReplica is in-memory, so a plain `deepdata serve` learns a
// tenant is a replica the only way it can: from the marker the follower left
// beside its store. Serve calls this on every open, and there is no flag to
// forget -- the data directory is the evidence.
//
// A marker that cannot be honored is a startup failure, never a shrug. Serving
// a tenant as an ordinary store is the one outcome that must not happen: a
// single local write consumes the LSN the leader's next record needs and
// forks the two histories at the same position.
func bindReplicaReadOnly(collections *CollectionHTTPServer) error {
	set := collections.Stores()
	if set == nil {
		return nil
	}
	for _, id := range set.Tenants() {
		store, ok := set.Store(id)
		if !ok {
			return fmt.Errorf("tenant %q vanished while binding its replica marker", id)
		}
		if err := bindReplicaMarker(set.Base(id))(store); err != nil {
			return fmt.Errorf("tenant %q: %w", id, err)
		}
	}
	return nil
}

// bindReplicaMarker reads the replica marker at base, if any, and applies it
// to store -- the AdoptTenant bind hook a standby uses for a tenant Seed just
// bootstrapped (whose marker Seed already wrote), and the body
// bindReplicaReadOnly re-applies to every tenant already on disk at boot.
func bindReplicaMarker(base string) func(*vcollection.DurableStore) error {
	return func(store *vcollection.DurableStore) error {
		leaderID, isReplica, err := replication.ReplicaLeaderID(base)
		if err != nil {
			return err
		}
		if !isReplica {
			return nil
		}
		if err := store.MakeReplica(leaderID); err != nil {
			return fmt.Errorf("serve read-only as a replica of leader %x: %w", leaderID, err)
		}
		return nil
	}
}

// runReplicate is the `deepdata replicate` subcommand: keep a local tenant
// directory in step with one tenant on a leader.
//
// It only syncs, and it holds the tenant directory for as long as it does:
// the collection store takes an exclusive lock, so a `deepdata serve` against
// the same tenant store directory is refused with "collection store is
// already open" while this command runs. A replica directory is therefore
// either tailing its leader or being served, never both at once, and
// switching between them means stopping one process and starting the other.
// Serving it is a plain `deepdata serve` against the parent tenant store
// directory: that process finds the replica marker, answers this tenant's
// reads normally, refuses its writes with a 403, and reports read_only on
// /readyz so a load balancer stops sending it writes.
//
// --tenant selects one tenant; omitting it follows every tenant the leader
// lists (replicateAll), each into its own subdirectory of the tenant store
// tree.
func runReplicate(args []string, logger *logging.Logger) int {
	fs := flag.NewFlagSet("replicate", flag.ExitOnError)
	leaderURL := fs.String("leader", "", "leader base URL, e.g. http://leader.internal:8080")
	tenant := fs.String("tenant", "", "tenant ID to replicate; omit to follow every tenant the leader lists")
	retry := fs.Duration("retry", 5*time.Second, "wait before reconnecting after the stream drops")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *leaderURL == "" {
		fmt.Fprintln(os.Stderr, "replicate: --leader is required")
		return 2
	}
	if *tenant != "" && !vcollection.ValidTenantID(*tenant) {
		fmt.Fprintln(os.Stderr, "replicate: --tenant must be a valid tenant ID")
		return 2
	}
	// loadServerConfig's errs cover the serve surface (PORT, rate limits,
	// timeouts, ...) that replicate never reads, so they are not this
	// subcommand's concern; VECTORDB_MODE is, and it is the one thing the
	// historical mode loader validated for replicate.
	cfg, _ := loadServerConfig(nil, os.Getenv)
	token := cfg.ReplicationToken
	if token == "" {
		fmt.Fprintf(os.Stderr, "replicate: %s must be set to the leader's node token\n", replicationTokenEnv)
		return 2
	}
	if cfg.mode != "" && cfg.mode != "local" {
		fmt.Fprintf(os.Stderr, "replicate: unknown mode: %s (valid: local)\n", cfg.mode)
		return 2
	}
	// The same directory layout the server would open, so a replica's tenant
	// store directory and the leader's are configured identically and an
	// operator can promote one by changing the subcommand, not the layout.
	tenantsDir := cfg.IndexPath + ".tenants"
	if err := os.MkdirAll(tenantsDir, 0o750); err != nil {
		logger.Error("cannot create the replica state directory", "path", tenantsDir, "error", err)
		return 1
	}

	// No whole-request timeout: a follow stream is open-ended by design, and a
	// Client.Timeout would cut it at a fixed interval forever.
	follower := replication.Follower{LeaderURL: *leaderURL, Token: token, Client: &http.Client{}}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if *tenant == "" {
		return replicateAll(ctx, follower, tenantsDir, logger, func(ctx context.Context, f *replication.Follower, base string) int {
			return followTenant(ctx, f, base, *retry, logger)
		})
	}
	follower.Tenant = *tenant
	return followTenant(ctx, &follower, filepath.Join(tenantsDir, *tenant), *retry, logger)
}

// relistInterval is how often replicateAll asks the leader for its current
// tenant list.
//
// ponytail: fixed re-list interval, no shared backoff, no worker pool.
var relistInterval = 30 * time.Second

// replicateAll follows every tenant template.Tenants lists, re-listing every
// relistInterval to pick up tenants created after it started. A tenant is
// started once and never restarted: if run ends (ctx canceled, or one of the
// terminal errors followStream already refuses to retry -- resync required,
// store mismatch, or the leader has since deleted the tenant), that is an
// operator decision, not something this loop second-guesses by starting it
// again next re-list.
//
// run is the per-tenant action: the replicate subcommand passes a followTenant
// closure that opens its own store under dir and closes it on exit; a standby
// passes one that adopts a store its StoreSet already owns instead (see
// standby.go). base is filepath.Join(dir, tenant); a caller with its own
// notion of a tenant's path (a standby has StoreSet.Base) may ignore it.
//
// Returns 0 if ctx was canceled and every tenant loop it started exited 0,
// else 1.
func replicateAll(ctx context.Context, template replication.Follower, dir string, logger *logging.Logger, run func(ctx context.Context, f *replication.Follower, base string) int) int {
	var (
		mu      sync.Mutex
		started = make(map[string]bool)
		wg      sync.WaitGroup
		failed  atomic.Bool
	)

	list := func() {
		tenants, err := template.Tenants(ctx)
		if err != nil {
			logger.Warn("cannot list tenants from the leader; retrying next interval", "leader", template.LeaderURL, "error", err)
			return
		}
		mu.Lock()
		defer mu.Unlock()
		for _, tenant := range tenants {
			if started[tenant] {
				continue
			}
			started[tenant] = true
			if !vcollection.ValidTenantID(tenant) {
				// The leader's /tenants response is not a trust boundary: an ID
				// from it must pass the same check a --tenant flag would before
				// it becomes a path component (filepath.Join(dir, tenant)).
				logger.Error("leader listed an invalid tenant ID; refusing to use it as a path", "leader", template.LeaderURL, "tenant", tenant)
				continue
			}
			follower := template
			follower.Tenant = tenant
			base := filepath.Join(dir, tenant)
			wg.Add(1)
			go func() {
				defer wg.Done()
				if rc := run(ctx, &follower, base); rc != 0 {
					failed.Store(true)
				}
			}()
		}
	}

	list()
	ticker := time.NewTicker(relistInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			wg.Wait()
			if failed.Load() {
				return 1
			}
			return 0
		case <-ticker.C:
			list()
		}
	}
}

// followTenant opens base as a replica of follower's tenant and streams
// records into it until ctx is canceled or the leader tells it to stop for
// good, retrying transient drops every retry.
func followTenant(ctx context.Context, follower *replication.Follower, base string, retry time.Duration, logger *logging.Logger) int {
	store, err := follower.Open(ctx, base, base)
	if err != nil {
		logger.Error("cannot open a replica of this leader", "path", base, "tenant", follower.Tenant, "leader", follower.LeaderURL, "error", err)
		return 1
	}
	defer func() {
		if closeErr := store.Close(); closeErr != nil {
			logger.Error("failed to close the replica store", "tenant", follower.Tenant, "error", closeErr)
		}
	}()
	logger.Info("replica open", "path", base, "tenant", follower.Tenant, "leader", follower.LeaderURL, "applied_lsn", store.ReplicaCursor().LSN)
	return followStream(ctx, follower, store, base, retry, logger, nil)
}

// followStream applies follower's leader journal to store until ctx is
// canceled or the leader tells it to stop for good, retrying transient drops
// every retry. observe, when non-nil, is called on every state change
// ("streaming", "reconnecting", "stopped") -- a standby's window into a
// tenant it never closes itself (see standby.go).
func followStream(ctx context.Context, follower *replication.Follower, store *vcollection.DurableStore, base string, retry time.Duration, logger *logging.Logger, observe func(state, errText string)) int {
	notify := func(state, errText string) {
		if observe != nil {
			observe(state, errText)
		}
	}
	for {
		notify("streaming", "")
		err := follower.Follow(ctx, store, base)
		switch {
		case ctx.Err() != nil:
			logger.Info("replication stopped", "tenant", follower.Tenant, "applied_lsn", store.ReplicaCursor().LSN)
			notify("stopped", "")
			return 0
		case errors.Is(err, replication.ErrResyncRequired):
			// Deliberately terminal. Recovering means discarding this
			// directory, and that is an operator's decision -- it may be the
			// one a read fleet is serving from.
			logger.Error("this replica is behind the leader's retained journal; discard the replica directory and start again to re-bootstrap",
				"path", base, "tenant", follower.Tenant, "applied_lsn", store.ReplicaCursor().LSN, "error", err)
			notify("stopped", err.Error())
			return 1
		case errors.Is(err, replication.ErrStaleLeader):
			logger.Error("leader is stale; point this standby at the current leader",
				"path", base, "tenant", follower.Tenant, "leader", follower.LeaderURL, "applied_lsn", store.ReplicaCursor().LSN, "error", err)
			notify("stopped", err.Error())
			return 1
		case errors.Is(err, vcollection.ErrJournalStoreMismatch):
			logger.Error("this replica does not belong to that leader", "path", base, "tenant", follower.Tenant, "leader", follower.LeaderURL, "error", err)
			notify("stopped", err.Error())
			return 1
		case errors.Is(err, replication.ErrUnknownTenant):
			logger.Error("leader no longer knows this tenant; not reconnecting", "path", base, "tenant", follower.Tenant, "leader", follower.LeaderURL, "error", err)
			notify("stopped", err.Error())
			return 1
		}
		logger.Warn("replication stream dropped; reconnecting", "tenant", follower.Tenant, "applied_lsn", store.ReplicaCursor().LSN, "retry_in", retry, "error", err)
		notify("reconnecting", err.Error())
		select {
		case <-ctx.Done():
			notify("stopped", "")
			return 0
		case <-time.After(retry):
		}
	}
}
