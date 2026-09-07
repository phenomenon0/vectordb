package main

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	vcollection "github.com/phenomenon0/vectordb/internal/collection"
	"github.com/phenomenon0/vectordb/internal/logging"
	"github.com/phenomenon0/vectordb/internal/replication"
)

// captureStderr redirects the package-level os.Stderr for the duration of fn
// and returns what was written. runReplicate reports its own config errors
// with fmt.Fprintf(os.Stderr, ...) rather than through the logger.
func captureStderr(t *testing.T, fn func()) string {
	t.Helper()
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("os.Pipe: %v", err)
	}
	orig := os.Stderr
	os.Stderr = w
	fn()
	os.Stderr = orig
	w.Close()
	out, _ := io.ReadAll(r)
	return string(out)
}

// canonicalLeaderForTest builds the client handler over a real durable store.
func canonicalLeaderForTest(t *testing.T) (http.Handler, *CollectionHTTPServer, string) {
	t.Helper()
	t.Setenv("API_TOKEN", "client-api-token-for-tenants")
	t.Setenv("JWT_SECRET", "")
	t.Setenv("REQUIRE_AUTH", "1")
	indexPath := filepath.Join(t.TempDir(), "index.gob")
	handler, collections := newCanonicalHTTPHandler(testServerRuntime(t), NewHashEmbedder(4), indexPath)
	t.Cleanup(func() { _ = collections.Close() })
	if !collections.IsDurable() {
		t.Fatal("test leader has no durable store to replicate")
	}
	return handler, collections, indexPath
}

// The node surface must not exist unless it was configured. An operator who
// never set the node token has not consented to a route that exports every
// tenant, and it must not appear because some other feature was enabled.
func TestReplicationSurfaceIsAbsentWithoutTheNodeToken(t *testing.T) {
	handler, collections, indexPath := canonicalLeaderForTest(t)
	mounted, err := canonicalReplicationSurface(handler, collections, indexPath, "", logging.Default())
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodGet, replication.PathPrefix+"status", nil)
	req.Header.Set("Authorization", "Bearer client-api-token-for-tenants")
	resp := httptest.NewRecorder()
	mounted.ServeHTTP(resp, req)
	if resp.Code != http.StatusNotFound {
		t.Fatalf("node route answered %d with replication disabled, want 404: %s", resp.Code, resp.Body.String())
	}
}

// The node token and the client API token are different credentials, and the
// node surface accepts only its own. This is the whole reason these routes are
// not in the V3 contract: /replication/v1/snapshot exports every tenant in one
// request, so a tenant's own token -- and the anonymous server-admin context
// that credentialless dev mode grants -- must not reach it.
func TestClientCredentialsCannotReachTheNodeSurface(t *testing.T) {
	handler, collections, indexPath := canonicalLeaderForTest(t)
	mounted, err := canonicalReplicationSurface(handler, collections, indexPath, "node-token-distinct-from-the-client-one", logging.Default())
	if err != nil {
		t.Fatal(err)
	}
	call := func(path, token string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodGet, path, nil)
		if token != "" {
			req.Header.Set("Authorization", "Bearer "+token)
		}
		resp := httptest.NewRecorder()
		mounted.ServeHTTP(resp, req)
		return resp
	}
	for _, route := range []string{"status", "snapshot", "journal"} {
		path := replication.PathPrefix + route
		for _, token := range []string{"", "client-api-token-for-tenants"} {
			if resp := call(path, token); resp.Code != http.StatusUnauthorized {
				t.Errorf("%s with token %q = %d, want 401", path, token, resp.Code)
			}
		}
		if resp := call(path, "node-token-distinct-from-the-client-one"); resp.Code == http.StatusUnauthorized {
			t.Errorf("%s rejected the node token", path)
		}
	}
	// Mounting the node surface must not move the client surface underneath it.
	if resp := call("/v3/status", "client-api-token-for-tenants"); resp.Code != http.StatusOK {
		t.Errorf("/v3/status = %d after mounting replication, want 200: %s", resp.Code, resp.Body.String())
	}
	if resp := call("/no/such/route", "client-api-token-for-tenants"); resp.Code != http.StatusNotFound {
		t.Errorf("unknown route = %d after mounting replication, want 404", resp.Code)
	}
}

// A memory-only process has no journal to stream. Serving an empty node
// surface would look healthy to a follower that will never receive a record,
// so the configuration is refused at startup instead.
func TestReplicationRefusesToStartWithoutADurableStore(t *testing.T) {
	collections := NewCollectionHTTPServer(filepath.Join(t.TempDir(), "index.gob.tenants"))
	if _, err := canonicalReplicationSurface(http.NotFoundHandler(), collections, "index.gob", "node-token-distinct-from-the-client-one", logging.Default()); err == nil {
		t.Fatal("replication mounted on a process with no durable store")
	}
}

// runReplicate reads serverConfig for its token and data path, but it never
// reads the serve-only knobs (rate limits, embedder dimension, ...) that
// loadServerConfig also validates. It must reject the one env key it does
// care about (VECTORDB_MODE) and stay silent about every other key's
// garbage, whether or not that garbage would fail a `serve` startup.
func TestRunReplicateValidatesOnlyWhatItReads(t *testing.T) {
	t.Run("invalid VECTORDB_MODE blocks it", func(t *testing.T) {
		t.Setenv(replicationTokenEnv, "node-token")
		t.Setenv("VECTORDB_MODE", "bogus")
		var rc int
		stderr := captureStderr(t, func() {
			rc = runReplicate([]string{"--leader", "http://leader.example", "--tenant", "acme"}, logging.Default())
		})
		if rc != 2 {
			t.Fatalf("rc = %d, want 2", rc)
		}
		if !strings.Contains(stderr, "unknown mode: bogus (valid: local)") {
			t.Fatalf("stderr = %q, want the unknown-mode rejection", stderr)
		}
	})

	t.Run("garbage in an unread serve key does not block it", func(t *testing.T) {
		t.Setenv("TENANT_RPS", "0")
		t.Setenv("DEEPDATA_EMBED_DIM", "not-a-number")
		var rc int
		stderr := captureStderr(t, func() {
			rc = runReplicate([]string{"--leader", "http://leader.example", "--tenant", "acme"}, logging.Default())
		})
		if rc != 2 {
			t.Fatalf("rc = %d, want 2 (missing replication token, not a config rejection)", rc)
		}
		if strings.Contains(stderr, "TENANT_RPS") || strings.Contains(stderr, "DEEPDATA_EMBED_DIM") {
			t.Fatalf("stderr = %q, replicate must not fail on keys it never reads", stderr)
		}
		if !strings.Contains(stderr, replicationTokenEnv+" must be set") {
			t.Fatalf("stderr = %q, want the missing-token message", stderr)
		}
	})
}

// tenantDocsSchema is the one-collection schema every tenant in the
// replicateAll test gets.
var tenantDocsSchema = vcollection.CollectionSchema{
	Name: "docs",
	Fields: []vcollection.VectorField{{
		Name:  "dense",
		Type:  vcollection.VectorTypeDense,
		Dim:   2,
		Index: vcollection.IndexConfig{Type: vcollection.IndexTypeFLAT},
	}},
}

// seedTenantDoc creates tenant's "docs" collection on leader and inserts one
// document tagged with tenant's own name, so a later read can tell which
// tenant's data came back.
func seedTenantDoc(t *testing.T, leader *vcollection.StoreSet, tenant string) uint64 {
	t.Helper()
	ctx := context.Background()
	if _, err := leader.CreateCollection(ctx, tenant, tenantDocsSchema); err != nil {
		t.Fatalf("create collection for %s on leader: %v", tenant, err)
	}
	doc := vcollection.Document{
		Vectors:  map[string]vcollection.Vector{"dense": vcollection.Vector{Dense: []float32{1, 0}}},
		Metadata: map[string]interface{}{"origin": tenant},
	}
	if err := leader.AddDocument(ctx, tenant, "docs", &doc); err != nil {
		t.Fatalf("insert on leader tenant %s: %v", tenant, err)
	}
	return doc.ID
}

// multiTenantLeaderForTest builds a real leader -- the same StoreSet `serve`
// opens -- seeded with acme and globex, each holding one document, and a
// per-tenant node surface over it.
func multiTenantLeaderForTest(t *testing.T) (leader *vcollection.StoreSet, leaderURL string, docIDs map[string]uint64) {
	t.Helper()
	dir := t.TempDir()
	leader, err := vcollection.OpenStoreSet(filepath.Join(dir, "leader.gob.tenants"), vcollection.StoreLimits{MaxTenants: 8, MaxCollections: 8})
	if err != nil {
		t.Fatalf("open leader store set: %v", err)
	}
	t.Cleanup(func() {
		if err := leader.Close(); err != nil {
			t.Errorf("close leader store set: %v", err)
		}
	})

	docIDs = map[string]uint64{
		"acme":   seedTenantDoc(t, leader, "acme"),
		"globex": seedTenantDoc(t, leader, "globex"),
	}

	node, err := replication.NewTenantLeaderHandler(replication.LeaderConfig{Token: "node-token", SpoolDir: dir},
		leader.Tenants, func(id string) (replication.Source, bool) { return leader.Store(id) })
	if err != nil {
		t.Fatalf("build node surface: %v", err)
	}
	srv := httptest.NewServer(node)
	t.Cleanup(srv.Close)
	return leader, srv.URL, docIDs
}

// waitForReplicaMarker polls until base carries a replica marker -- meaning
// a followTenant loop finished bootstrapping there -- or fails the test.
func waitForReplicaMarker(t *testing.T, base string) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if _, isReplica, err := replication.ReplicaLeaderID(base); err == nil && isReplica {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("no replica marker appeared at %s within the deadline", base)
}

// TestReplicateAllFollowsEveryTenantTheLeaderLists is the B4 contract:
// `deepdata replicate` without --tenant follows every tenant the leader
// lists, into its own subdirectory, and picks up a tenant created after it
// started on the next re-list.
func TestReplicateAllFollowsEveryTenantTheLeaderLists(t *testing.T) {
	orig := relistInterval
	relistInterval = 30 * time.Millisecond
	t.Cleanup(func() { relistInterval = orig })

	leader, leaderURL, docIDs := multiTenantLeaderForTest(t)

	indexPath := filepath.Join(t.TempDir(), "replica.gob")
	tenantsDir := indexPath + ".tenants"
	if err := os.MkdirAll(tenantsDir, 0o750); err != nil {
		t.Fatalf("create replica tenant store directory: %v", err)
	}
	template := replication.Follower{LeaderURL: leaderURL, Token: "node-token", Client: &http.Client{}}

	ctx, cancel := context.WithCancel(context.Background())
	rcCh := make(chan int, 1)
	go func() { rcCh <- replicateAll(ctx, template, tenantsDir, 50*time.Millisecond, logging.Default()) }()

	waitForReplicaMarker(t, filepath.Join(tenantsDir, "acme"))
	waitForReplicaMarker(t, filepath.Join(tenantsDir, "globex"))

	docIDs["initech"] = seedTenantDoc(t, leader, "initech")
	waitForReplicaMarker(t, filepath.Join(tenantsDir, "initech"))

	cancel()
	select {
	case rc := <-rcCh:
		if rc != 0 {
			t.Fatalf("replicateAll rc = %d, want 0", rc)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("replicateAll did not stop after ctx was canceled")
	}

	handler := newCanonicalSurfaceTestHandlerAt(t, indexPath)
	for _, tenant := range []string{"acme", "globex", "initech"} {
		path := "/v3/tenants/" + tenant + "/collections/docs/docs/" + strconv.FormatUint(docIDs[tenant], 10)
		resp := canonicalCall(t, handler, http.MethodGet, path, "")
		if resp.Code != http.StatusOK {
			t.Errorf("tenant %s: get leader document from replica = %d: %s", tenant, resp.Code, resp.Body.String())
			continue
		}
		if !strings.Contains(resp.Body.String(), `"`+tenant+`"`) {
			t.Errorf("tenant %s: replica returned a document that is not the leader's: %s", tenant, resp.Body.String())
		}
	}
}
