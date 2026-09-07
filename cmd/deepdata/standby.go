package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"sync"
	"time"

	vcollection "github.com/phenomenon0/vectordb/internal/collection"
	"github.com/phenomenon0/vectordb/internal/logging"
	"github.com/phenomenon0/vectordb/internal/replication"
)

// standbyRetry is how long a standby's per-tenant stream waits before
// reconnecting after a drop -- the same default `deepdata replicate --retry`
// uses. There is no env var for it: DEEPDATA_LEADER_URL has no companion
// flags (see config.go).
const standbyRetry = 5 * time.Second

// standby follows every tenant a leader lists into a StoreSet, in-process, so
// a `deepdata serve` pointed at DEEPDATA_LEADER_URL both answers reads
// locally and keeps its state in step with the leader -- no separate
// `replicate` process, no separate data directory.
//
// It shares replicateAll's re-list loop with the replicate subcommand; only
// the per-tenant action differs: followTenant opens its own store and closes
// it on exit, while a standby adopts a store its StoreSet already owns and
// never closes it -- server shutdown does, through the set (see stop).
type standby struct {
	leader string

	mu      sync.Mutex
	tenants map[string]*standbyTenant

	cancel context.CancelFunc
	wg     sync.WaitGroup

	// promoteMu serializes promote against a second call. It is not mu: the
	// halt step below waits on the follow goroutines, which themselves take
	// mu (setState, setLeaderLSN), so holding mu across that wait would
	// deadlock against them.
	promoteMu sync.Mutex
}

// standbyTenant is one tenant's last known following state, reported on
// /readyz.
type standbyTenant struct {
	state     string // "bootstrapping", "streaming", "reconnecting", "stopped"
	leaderLSN uint64
	err       string
}

// startStandby starts following every tenant leaderURL lists into set and
// returns immediately; the follow loops run in the background until the
// returned standby's stop is called.
func startStandby(ctx context.Context, leaderURL, token string, retry time.Duration, set *vcollection.StoreSet, logger *logging.Logger) *standby {
	ctx, cancel := context.WithCancel(ctx)
	st := &standby{leader: leaderURL, tenants: make(map[string]*standbyTenant), cancel: cancel}
	template := replication.Follower{LeaderURL: leaderURL, Token: token, Client: &http.Client{}}
	logger.Info("following leader", "leader", leaderURL)
	st.wg.Add(1)
	go func() {
		defer st.wg.Done()
		// dir is unused here: this run func ignores replicateAll's
		// filepath.Join(dir, tenant) base and asks the StoreSet for the
		// tenant's own path instead, so it can never drift from what
		// AdoptTenant actually opens.
		replicateAll(ctx, template, "", logger, func(ctx context.Context, f *replication.Follower, _ string) int {
			return st.followTenant(ctx, f, set, retry, logger)
		})
	}()
	return st
}

// followTenant seeds, adopts and binds one tenant into set, then streams the
// leader's journal into it until ctx is canceled or the stream ends for good.
// Unlike the replicate subcommand's followTenant, it never closes the store:
// set owns it, and an HTTP request may be reading from it right now.
func (st *standby) followTenant(ctx context.Context, f *replication.Follower, set *vcollection.StoreSet, retry time.Duration, logger *logging.Logger) int {
	id := f.Tenant
	base := set.Base(id)
	st.setState(id, "bootstrapping", "")
	if _, err := f.Seed(ctx, base, base); err != nil {
		logger.Error("cannot seed standby tenant from the leader", "tenant", id, "leader", f.LeaderURL, "error", err)
		st.setState(id, "stopped", err.Error())
		return 1
	}
	store, err := set.AdoptTenant(id, bindReplicaMarker(base))
	if err != nil {
		logger.Error("cannot adopt standby tenant", "tenant", id, "error", err)
		st.setState(id, "stopped", err.Error())
		return 1
	}
	if err := f.Bind(ctx, store, base); err != nil {
		// A tenant already on disk under a different StoreID (someone's local
		// tenant of the same name, not this leader's) lands here: bind never
		// touches the store on a mismatch, so it keeps serving as ordinary
		// local state, not as this leader's replica.
		logger.Error("cannot bind standby tenant to its leader", "tenant", id, "leader", f.LeaderURL, "error", err)
		st.setState(id, "stopped", err.Error())
		return 1
	}
	f.OnPreamble = func(pre replication.Preamble) { st.setLeaderLSN(id, pre.LatestLSN) }
	logger.Info("standby tenant bound", "tenant", id, "leader", f.LeaderURL, "applied_lsn", store.ReplicaCursor().LSN)
	return followStream(ctx, f, store, base, retry, logger, func(state, errText string) { st.setState(id, state, errText) })
}

func (st *standby) tenant(id string) *standbyTenant {
	t, ok := st.tenants[id]
	if !ok {
		t = &standbyTenant{}
		st.tenants[id] = t
	}
	return t
}

func (st *standby) setState(id, state, errText string) {
	st.mu.Lock()
	defer st.mu.Unlock()
	t := st.tenant(id)
	t.state = state
	t.err = errText
}

func (st *standby) setLeaderLSN(id string, lsn uint64) {
	st.mu.Lock()
	defer st.mu.Unlock()
	st.tenant(id).leaderLSN = lsn
}

// snapshot returns a point-in-time copy of every tenant this standby has
// started, keyed by tenant ID, for /readyz.
func (st *standby) snapshot() map[string]standbyTenant {
	st.mu.Lock()
	defer st.mu.Unlock()
	out := make(map[string]standbyTenant, len(st.tenants))
	for id, t := range st.tenants {
		out[id] = *t
	}
	return out
}

// stop cancels every follow loop and waits for them to exit. Must be called
// before the StoreSet closes, so no Follow is applying into a closing store.
func (st *standby) stop() {
	st.cancel()
	st.wg.Wait()
}

// promote is the online counterpart to the offline `deepdata promote`
// command (see promote.go): it halts this standby's follow loops, then
// fences every tenant currently bound as a replica in set into a leader of
// its own history -- same three artifacts promote.go writes (bumped epoch
// sidecar, dropped replica marker), just against stores this process already
// has open instead of ones it opens for the occasion.
//
// It stops at the first tenant that fails and returns what it got through:
// promoted lists, in order, every tenant that completed all of it, and those
// stay promoted -- there is no rollback, same as the offline command past
// its own point of no return. Call it at most once per standby: the caller
// (handlePromote) discards a standby once this returns without error.
func (st *standby) promote(set *vcollection.StoreSet) (promoted []string, epoch map[string]uint64, err error) {
	st.promoteMu.Lock()
	defer st.promoteMu.Unlock()
	st.stop()

	epoch = map[string]uint64{}
	for _, id := range set.ReplicaTenants() {
		base := set.Base(id)
		store, ok := set.Store(id)
		if !ok {
			return promoted, epoch, fmt.Errorf("tenant %q vanished during promotion", id)
		}
		if err := store.Promote(); err != nil {
			return promoted, epoch, fmt.Errorf("tenant %q: %w", id, err)
		}
		pos, err := store.JournalStatus()
		if err != nil {
			return promoted, epoch, fmt.Errorf("tenant %q: %w", id, err)
		}
		local, err := replication.ReadEpoch(base)
		if err != nil {
			return promoted, epoch, fmt.Errorf("tenant %q: %w", id, err)
		}
		next := replication.Epoch{Number: local.Number + 1, StartLSN: pos.LatestLSN}
		if err := replication.WriteEpoch(base, next); err != nil {
			return promoted, epoch, fmt.Errorf("tenant %q: %w", id, err)
		}
		if err := os.Remove(base + promoteMarkerSuffix); err != nil {
			return promoted, epoch, fmt.Errorf("tenant %q: remove replica marker: %w", id, err)
		}
		promoted = append(promoted, id)
		epoch[id] = next.Number
	}
	set.SetReadOnly(false)
	return promoted, epoch, nil
}
