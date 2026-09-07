package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	vcollection "github.com/phenomenon0/vectordb/internal/collection"
	"github.com/phenomenon0/vectordb/internal/logging"
	"github.com/phenomenon0/vectordb/internal/replication"
)

const (
	promoteHelperEnv = "DEEPDATA_PROMOTE_HELPER"
	promoteArgsEnv   = "DEEPDATA_PROMOTE_ARGS"
)

// TestPromoteProcessHelper is the re-exec target runPromoteProcess launches:
// same trick as TestMigrateTenantsProcessHelper, for the promote subcommand.
func TestPromoteProcessHelper(t *testing.T) {
	if os.Getenv(promoteHelperEnv) != "1" {
		return
	}
	os.Args = append([]string{"deepdata", "promote"}, strings.Split(os.Getenv(promoteArgsEnv), "\x1f")...)
	main()
}

// runPromoteProcess runs `deepdata promote args...` as a real subprocess and
// returns its exit code and combined output.
func runPromoteProcess(t *testing.T, args ...string) (rc int, output string) {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^TestPromoteProcessHelper$")
	cmd.Env = processEnv(map[string]string{
		promoteHelperEnv: "1",
		promoteArgsEnv:   strings.Join(args, "\x1f"),
	})
	out, err := cmd.CombinedOutput()
	if err == nil {
		return 0, string(out)
	}
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		return exitErr.ExitCode(), string(out)
	}
	t.Fatalf("run promote helper: %v\n%s", err, out)
	return -1, string(out)
}

var promoteFixtureSchema = vcollection.CollectionSchema{
	Name: "docs",
	Fields: []vcollection.VectorField{{
		Name: "dense", Type: vcollection.VectorTypeDense, Dim: 2,
		Index: vcollection.IndexConfig{Type: vcollection.IndexTypeFLAT},
	}},
}

// buildPromoteFixture syncs a standby copy of tenants acme and globex the way
// `deepdata replicate` does -- Follower.Open against a real node surface,
// then Close (replicaDirectoryForTest's shape, generalized to two tenants
// sharing one tenants directory) -- and returns the data directory `deepdata
// promote` runs against plus the LSN each standby is at, which is exactly
// what promote must fence at.
func buildPromoteFixture(t *testing.T) (dataDir string, lsn map[string]uint64) {
	t.Helper()
	ctx := context.Background()
	dir := t.TempDir()

	leaderDir := filepath.Join(dir, "leader.gob.tenants")
	leader, err := vcollection.OpenStoreSet(leaderDir, vcollection.StoreLimits{MaxTenants: 8, MaxCollections: 8})
	if err != nil {
		t.Fatalf("open leader store set: %v", err)
	}
	defer func() {
		if err := leader.Close(); err != nil {
			t.Errorf("close leader store set: %v", err)
		}
	}()

	for _, tenant := range []string{"acme", "globex"} {
		if _, err := leader.CreateCollection(ctx, tenant, promoteFixtureSchema); err != nil {
			t.Fatalf("create collection for %s: %v", tenant, err)
		}
		doc := &vcollection.Document{Vectors: map[string]vcollection.Vector{"dense": {Dense: []float32{1, 0}}}}
		if err := leader.AddDocument(ctx, tenant, "docs", doc); err != nil {
			t.Fatalf("insert on %s: %v", tenant, err)
		}
	}

	node, err := replication.NewTenantLeaderHandler(replication.LeaderConfig{Token: "node-token", SpoolDir: dir},
		leader.Tenants, func(id string) (replication.Source, bool) { return leader.Store(id) })
	if err != nil {
		t.Fatalf("build node surface: %v", err)
	}
	srv := httptest.NewServer(node)
	defer srv.Close()

	dataDir = filepath.Join(dir, "standby")
	tenantsDir := filepath.Join(dataDir, "index.gob.tenants")
	if err := os.MkdirAll(tenantsDir, 0o750); err != nil {
		t.Fatalf("create standby tenant store directory: %v", err)
	}
	lsn = map[string]uint64{}
	for _, tenant := range []string{"acme", "globex"} {
		base := filepath.Join(tenantsDir, tenant)
		follower := &replication.Follower{LeaderURL: srv.URL, Token: "node-token", Tenant: tenant}
		replica, err := follower.Open(ctx, base, base)
		if err != nil {
			t.Fatalf("sync standby %s: %v", tenant, err)
		}
		pos, err := replica.JournalStatus()
		if err != nil {
			t.Fatalf("standby %s journal status: %v", tenant, err)
		}
		lsn[tenant] = pos.LatestLSN
		if err := replica.Close(); err != nil {
			t.Fatalf("close standby %s: %v", tenant, err)
		}
	}
	return dataDir, lsn
}

// A fresh promote must fence every standby it finds, in one shot, and hand
// back a directory that is no longer a standby at all: markers gone, an
// epoch sidecar recording where each tenant was fenced, and a directory
// `deepdata serve` treats as an ordinary writable leader.
func TestPromotePromotesEveryStandbyAndTheDirectoryServes(t *testing.T) {
	dataDir, lsn := buildPromoteFixture(t)
	tenantsDir := filepath.Join(dataDir, "index.gob.tenants")

	rc, out := runPromoteProcess(t, dataDir)
	if rc != 0 {
		t.Fatalf("promote rc = %d: %s", rc, out)
	}
	for _, tenant := range []string{"acme", "globex"} {
		if !strings.Contains(out, tenant) {
			t.Errorf("promote output missing %s: %s", tenant, out)
		}
		if _, err := os.Stat(filepath.Join(tenantsDir, tenant+"-replica")); !os.IsNotExist(err) {
			t.Errorf("%s replica marker still present after promote: %v", tenant, err)
		}
		got, err := replication.ReadEpoch(filepath.Join(tenantsDir, tenant))
		want := replication.Epoch{Number: 1, StartLSN: lsn[tenant]}
		if err != nil || got != want {
			t.Errorf("%s epoch = %+v, %v; want %+v, nil", tenant, got, err, want)
		}
	}

	indexPath := filepath.Join(dataDir, "index.gob")
	handler := newCanonicalSurfaceTestHandlerAt(t, indexPath)
	create := `{"vectors":{"dense":[0,1]},"metadata":{"origin":"new-leader"}}`
	if resp := canonicalCall(t, handler, http.MethodPost, "/v3/tenants/acme/collections/docs/docs", create); resp.Code != http.StatusOK {
		t.Fatalf("write on the promoted directory = %d, want 200: %s", resp.Code, resp.Body.String())
	}

	resp := canonicalCall(t, handler, http.MethodGet, "/readyz", "")
	var ready map[string]any
	if err := json.Unmarshal(resp.Body.Bytes(), &ready); err != nil {
		t.Fatalf("/readyz body is not JSON: %s", resp.Body.String())
	}
	if ready["read_only"] != false {
		t.Errorf("/readyz read_only = %v on a promoted directory, want false", ready["read_only"])
	}
	if replicas, _ := ready["replica_tenants"].([]any); len(replicas) != 0 {
		t.Errorf("/readyz replica_tenants = %v, want none", ready["replica_tenants"])
	}
}

// A directory a running serve or follower still holds must fail loud, and
// nothing may change: an operator who forgot to stop one process must be
// able to just fix that and retry, not clean up a half-fenced directory.
func TestPromoteRefusesWhenATenantIsHeldOpenAndChangesNothing(t *testing.T) {
	dataDir, _ := buildPromoteFixture(t)
	tenantsDir := filepath.Join(dataDir, "index.gob.tenants")
	acmeBase := filepath.Join(tenantsDir, "acme")
	held, err := vcollection.OpenDurableStore(acmeBase, acmeBase)
	if err != nil {
		t.Fatalf("hold acme open: %v", err)
	}
	defer func() { _ = held.Abort() }()

	var stdout, stderr bytes.Buffer
	if rc := runPromote([]string{dataDir}, &stdout, &stderr, logging.Default()); rc != 1 {
		t.Fatalf("promote rc = %d, want 1: %s", rc, stderr.String())
	}
	for _, tenant := range []string{"acme", "globex"} {
		if _, err := os.Stat(filepath.Join(tenantsDir, tenant+"-replica")); err != nil {
			t.Errorf("%s replica marker missing after a failed promote: %v", tenant, err)
		}
		if got, err := replication.ReadEpoch(filepath.Join(tenantsDir, tenant)); err != nil || got != (replication.Epoch{}) {
			t.Errorf("%s epoch = %+v, %v after a failed promote; want untouched zero value", tenant, got, err)
		}
	}
}

// Nothing to promote is an operator mistake (wrong directory, or a directory
// that was never a standby), not a crash.
func TestPromoteWithNoStandbyMarkersFailsLoud(t *testing.T) {
	dataDir := filepath.Join(t.TempDir(), "state")
	var stdout, stderr bytes.Buffer
	if rc := runPromote([]string{dataDir}, &stdout, &stderr, logging.Default()); rc != 2 {
		t.Fatalf("promote on an empty dir rc = %d, want 2: %s", rc, stderr.String())
	}
}

// --tenant scopes promotion to one tenant and leaves every other tenant's
// marker and epoch exactly as they were.
func TestPromoteTenantFlagPromotesOnlyThatTenant(t *testing.T) {
	dataDir, lsn := buildPromoteFixture(t)
	tenantsDir := filepath.Join(dataDir, "index.gob.tenants")

	var stdout, stderr bytes.Buffer
	if rc := runPromote([]string{"--tenant", "acme", dataDir}, &stdout, &stderr, logging.Default()); rc != 0 {
		t.Fatalf("promote --tenant acme rc = %d: %s", rc, stderr.String())
	}
	if _, err := os.Stat(filepath.Join(tenantsDir, "acme-replica")); !os.IsNotExist(err) {
		t.Errorf("acme replica marker still present: %v", err)
	}
	if _, err := os.Stat(filepath.Join(tenantsDir, "globex-replica")); err != nil {
		t.Errorf("globex replica marker removed by a --tenant acme promote: %v", err)
	}
	wantAcme := replication.Epoch{Number: 1, StartLSN: lsn["acme"]}
	if got, err := replication.ReadEpoch(filepath.Join(tenantsDir, "acme")); err != nil || got != wantAcme {
		t.Errorf("acme epoch = %+v, %v, want %+v, nil", got, err, wantAcme)
	}
	if got, err := replication.ReadEpoch(filepath.Join(tenantsDir, "globex")); err != nil || got != (replication.Epoch{}) {
		t.Errorf("globex epoch = %+v, %v, want untouched zero value", got, err)
	}
}

// --tenant naming a tenant that has no marker is an operator mistake, not a
// no-op success.
func TestPromoteTenantFlagRejectsANonStandbyTenant(t *testing.T) {
	dataDir, _ := buildPromoteFixture(t)
	var stdout, stderr bytes.Buffer
	if rc := runPromote([]string{"--tenant", "nope", dataDir}, &stdout, &stderr, logging.Default()); rc != 2 {
		t.Fatalf("promote --tenant nope rc = %d, want 2: %s", rc, stderr.String())
	}
}

// TestPromoteFencesTheOldLeaderButNotAReplicaThatNeverPassedTheLSN is the
// runbook's second half made real (docs/troubleshooting.md "Promote a
// standby by hand"): after promotion, the demoted leader's own writes past
// the promotion LSN put it in a history the new leader's epoch does not
// extend, so pointing it at the new leader must resync rather than silently
// fork; a replica that never advanced past that LSN has no such foreign
// record and keeps following, live, under the new epoch.
func TestPromoteFencesTheOldLeaderButNotAReplicaThatNeverPassedTheLSN(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()

	oldLeaderDir := filepath.Join(dir, "old.gob.tenants")
	oldLeader, err := vcollection.OpenStoreSet(oldLeaderDir, vcollection.StoreLimits{MaxTenants: 8, MaxCollections: 8})
	if err != nil {
		t.Fatalf("open old leader: %v", err)
	}
	if _, err := oldLeader.CreateCollection(ctx, "acme", promoteFixtureSchema); err != nil {
		t.Fatalf("create collection: %v", err)
	}
	first := &vcollection.Document{Vectors: map[string]vcollection.Vector{"dense": {Dense: []float32{1, 0}}}}
	if err := oldLeader.AddDocument(ctx, "acme", "docs", first); err != nil {
		t.Fatalf("insert on old leader: %v", err)
	}

	oldNode, err := replication.NewTenantLeaderHandler(replication.LeaderConfig{Token: "old-token", SpoolDir: dir},
		oldLeader.Tenants, func(id string) (replication.Source, bool) { return oldLeader.Store(id) })
	if err != nil {
		t.Fatalf("build old node surface: %v", err)
	}
	oldSrv := httptest.NewServer(oldNode)
	defer oldSrv.Close()

	// Two standbys split off from the same point: one will be promoted, the
	// other stays behind and never sees what the old leader writes next.
	promotedDataDir := filepath.Join(dir, "promoted")
	promotedBase := filepath.Join(promotedDataDir, "index.gob.tenants", "acme")
	if err := os.MkdirAll(filepath.Dir(promotedBase), 0o750); err != nil {
		t.Fatalf("create promoted standby tenant store directory: %v", err)
	}
	promotedFollower := &replication.Follower{LeaderURL: oldSrv.URL, Token: "old-token", Tenant: "acme"}
	promotedReplica, err := promotedFollower.Open(ctx, promotedBase, promotedBase)
	if err != nil {
		t.Fatalf("sync soon-to-be-promoted standby: %v", err)
	}
	if err := promotedReplica.Close(); err != nil {
		t.Fatalf("close promoted standby: %v", err)
	}

	laggingBase := filepath.Join(dir, "lagging.gob.tenants", "acme")
	if err := os.MkdirAll(filepath.Dir(laggingBase), 0o750); err != nil {
		t.Fatalf("create lagging standby tenant store directory: %v", err)
	}
	laggingFollower := &replication.Follower{LeaderURL: oldSrv.URL, Token: "old-token", Tenant: "acme"}
	laggingReplica, err := laggingFollower.Open(ctx, laggingBase, laggingBase)
	if err != nil {
		t.Fatalf("sync lagging standby: %v", err)
	}
	if err := laggingReplica.Close(); err != nil {
		t.Fatalf("close lagging standby: %v", err)
	}

	// The old leader keeps writing past the point promotion will fence at.
	second := &vcollection.Document{Vectors: map[string]vcollection.Vector{"dense": {Dense: []float32{0, 1}}}}
	if err := oldLeader.AddDocument(ctx, "acme", "docs", second); err != nil {
		t.Fatalf("insert past the promotion point: %v", err)
	}
	if err := oldLeader.Close(); err != nil {
		t.Fatalf("close old leader: %v", err)
	}

	var stdout, stderr bytes.Buffer
	if rc := runPromote([]string{promotedDataDir}, &stdout, &stderr, logging.Default()); rc != 0 {
		t.Fatalf("promote rc = %d: %s", rc, stderr.String())
	}

	promotedSet, err := vcollection.OpenStoreSet(filepath.Join(promotedDataDir, "index.gob.tenants"), vcollection.StoreLimits{MaxTenants: 8, MaxCollections: 8})
	if err != nil {
		t.Fatalf("reopen promoted store: %v", err)
	}
	defer func() { _ = promotedSet.Close() }()
	newNode, err := replication.NewTenantLeaderHandler(replication.LeaderConfig{
		Token:    "new-token",
		SpoolDir: dir,
		Epoch:    func(id string) (replication.Epoch, error) { return replication.ReadEpoch(promotedSet.Base(id)) },
	}, promotedSet.Tenants, func(id string) (replication.Source, bool) { return promotedSet.Store(id) })
	if err != nil {
		t.Fatalf("build new node surface: %v", err)
	}
	newSrv := httptest.NewServer(newNode)
	defer newSrv.Close()

	// The old leader, restarted as a standby of the new one: its own record
	// past the promotion LSN was never part of the new leader's history.
	oldBase := filepath.Join(oldLeaderDir, "acme")
	oldStore, err := vcollection.OpenDurableStore(oldBase, oldBase)
	if err != nil {
		t.Fatalf("reopen old leader's acme store: %v", err)
	}
	defer func() { _ = oldStore.Abort() }()
	oldFollowing := &replication.Follower{LeaderURL: newSrv.URL, Token: "new-token", Tenant: "acme"}
	if err := oldFollowing.Bind(ctx, oldStore, oldBase); err != nil {
		t.Fatalf("bind demoted leader to the new one: %v", err)
	}
	bindCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	if err := oldFollowing.Follow(bindCtx, oldStore, oldBase); !errors.Is(err, replication.ErrResyncRequired) {
		t.Fatalf("demoted leader past the promotion point Follow = %v, want ErrResyncRequired", err)
	}

	// The lagging replica never wrote past the promotion LSN, so it adopts
	// the new epoch and keeps following live.
	laggingStore, err := vcollection.OpenDurableStore(laggingBase, laggingBase)
	if err != nil {
		t.Fatalf("reopen lagging standby: %v", err)
	}
	defer func() { _ = laggingStore.Close() }()
	laggingFollowing := &replication.Follower{LeaderURL: newSrv.URL, Token: "new-token", Tenant: "acme"}
	if err := laggingFollowing.Bind(ctx, laggingStore, laggingBase); err != nil {
		t.Fatalf("bind lagging standby to the new leader: %v", err)
	}

	third := &vcollection.Document{Vectors: map[string]vcollection.Vector{"dense": {Dense: []float32{1, 1}}}}
	if err := promotedSet.AddDocument(ctx, "acme", "docs", third); err != nil {
		t.Fatalf("insert on the new leader: %v", err)
	}

	followCtx, cancel2 := context.WithTimeout(ctx, 10*time.Second)
	defer cancel2()
	failed := make(chan error, 1)
	go func() { failed <- laggingFollowing.Follow(followCtx, laggingStore, laggingBase) }()
	deadline := time.Now().Add(10 * time.Second)
	for {
		if _, ok := laggingStore.Tenants().GetDocument("acme", "docs", third.ID); ok {
			break
		}
		select {
		case err := <-failed:
			t.Fatalf("lagging standby Follow ended before applying the post-promotion record: %v", err)
		default:
		}
		if time.Now().After(deadline) {
			t.Fatal("post-promotion record never reached the lagging standby")
		}
		time.Sleep(10 * time.Millisecond)
	}
	cancel2()
	<-failed
}
