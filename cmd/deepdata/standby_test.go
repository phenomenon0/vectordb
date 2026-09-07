package main

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	vcollection "github.com/phenomenon0/vectordb/internal/collection"
	"github.com/phenomenon0/vectordb/internal/logging"
)

// standbyHandlerForTest builds a fresh, empty durable store the same way
// `deepdata serve` does and returns its handler and CollectionHTTPServer, so
// a test can start a standby against the same StoreSet the HTTP surface
// answers requests from.
func standbyHandlerForTest(t *testing.T) (http.Handler, *CollectionHTTPServer) {
	t.Helper()
	t.Setenv("API_TOKEN", "")
	t.Setenv("JWT_SECRET", "")
	t.Setenv("REQUIRE_AUTH", "0")
	indexPath := filepath.Join(t.TempDir(), "standby.gob")
	handler, collections := newCanonicalHTTPHandler(testServerRuntime(t), NewHashEmbedder(4), indexPath)
	t.Cleanup(func() { _ = collections.Close() })
	if !collections.IsDurable() {
		t.Fatal("test standby has no durable store")
	}
	return handler, collections
}

// waitForStandbyState polls until id reports state, or fails the test.
func waitForStandbyState(t *testing.T, st *standby, id, state string) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if tenant := st.snapshot()[id]; tenant.state == state {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("tenant %q did not reach state %q within the deadline (last: %+v)", id, state, st.snapshot()[id])
}

func readyzBody(t *testing.T, handler http.Handler) map[string]any {
	t.Helper()
	resp := canonicalCall(t, handler, http.MethodGet, "/readyz", "")
	if resp.Code != http.StatusOK {
		t.Fatalf("GET /readyz = %d: %s", resp.Code, resp.Body.String())
	}
	var body map[string]any
	if err := json.Unmarshal(resp.Body.Bytes(), &body); err != nil {
		t.Fatalf("readyz body is not JSON: %v", err)
	}
	return body
}

// A standby is a `deepdata serve` following a leader in-process: it answers
// reads for what it has synced, refuses every write like any other replica
// node, refuses to mint tenants the leader may still ship, and picks up a
// tenant the leader created after the standby started -- all without a
// separate `replicate` process.
func TestServeFollowServesReadsWhileFollowing(t *testing.T) {
	orig := relistInterval
	relistInterval = 30 * time.Millisecond
	t.Cleanup(func() { relistInterval = orig })

	leader, leaderURL, docIDs := multiTenantLeaderForTest(t)
	handler, collections := standbyHandlerForTest(t)
	set := collections.Stores()
	set.SetReadOnly(true)

	st := startStandby(context.Background(), leaderURL, "node-token", 30*time.Millisecond, set, logging.Default())
	collections.SetStandby(st)

	waitForStandbyState(t, st, "acme", "streaming")
	waitForStandbyState(t, st, "globex", "streaming")

	// A tenant created on the leader after the standby started appears after
	// the next re-list.
	docIDs["initech"] = seedTenantDoc(t, leader, "initech")
	waitForStandbyState(t, st, "initech", "streaming")

	// An insert on the leader is readable on the standby within 5s.
	newDoc := vcollection.Document{
		Vectors:  map[string]vcollection.Vector{"dense": vcollection.Vector{Dense: []float32{0, 1}}},
		Metadata: map[string]interface{}{"origin": "acme-2"},
	}
	if err := leader.AddDocument(context.Background(), "acme", "docs", &newDoc); err != nil {
		t.Fatalf("insert on leader: %v", err)
	}
	deadline := time.Now().Add(5 * time.Second)
	path := "/v3/tenants/acme/collections/docs/docs/" + strconv.FormatUint(newDoc.ID, 10)
	var resp = canonicalCall(t, handler, http.MethodGet, path, "")
	for resp.Code != http.StatusOK && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
		resp = canonicalCall(t, handler, http.MethodGet, path, "")
	}
	if resp.Code != http.StatusOK {
		t.Fatalf("get post-start leader insert from standby = %d: %s", resp.Code, resp.Body.String())
	}

	// A standby write through the HTTP handler is refused, same as any
	// other replica.
	resp = canonicalCall(t, handler, http.MethodPost, "/v3/tenants/acme/collections/docs/docs",
		`{"vectors":{"dense":[0,1]},"metadata":{"origin":"intruder"}}`)
	if resp.Code != http.StatusForbidden {
		t.Errorf("write to a followed tenant = %d, want 403: %s", resp.Code, resp.Body.String())
	}

	// Creating a collection for a brand new tenant on a following (read-only)
	// standby is refused too: it must not mint a local tenant the leader may
	// still ship.
	resp = canonicalCall(t, handler, http.MethodPost, "/v3/tenants/newco/collections",
		`{"name":"docs","fields":[{"name":"dense","type":"dense","dim":2,"index":{"type":"flat"}}]}`)
	if resp.Code != http.StatusForbidden {
		t.Errorf("create collection for a new tenant on a standby = %d, want 403: %s", resp.Code, resp.Body.String())
	}

	body := readyzBody(t, handler)
	following, ok := body["following"].(map[string]any)
	if !ok {
		t.Fatalf("readyz body has no following: %v", body)
	}
	if following["leader"] != leaderURL {
		t.Errorf("following.leader = %v, want %q", following["leader"], leaderURL)
	}
	tenants, ok := following["tenants"].(map[string]any)
	if !ok {
		t.Fatalf("following.tenants missing: %v", following)
	}
	acme, ok := tenants["acme"].(map[string]any)
	if !ok {
		t.Fatalf("following.tenants.acme missing: %v", tenants)
	}
	if acme["state"] != "streaming" {
		t.Errorf("following.tenants.acme.state = %v, want streaming", acme["state"])
	}
	if lag, _ := acme["lag"].(float64); lag != 0 {
		t.Errorf("following.tenants.acme.lag = %v, want 0", acme["lag"])
	}

	st.stop()
	if err := set.Close(); err != nil {
		t.Errorf("close standby store set: %v", err)
	}
}

// A tenant directory that already holds unrelated local data under the same
// name a leader also uses must never be folded into that leader's history:
// the standby refuses to bind it, leaving it exactly as ordinary and
// writable as it was before the standby ever started, while every other
// tenant follows normally.
func TestServeFollowKeepsAForeignTenantLocal(t *testing.T) {
	_, leaderURL, _ := multiTenantLeaderForTest(t)

	handler, collections := standbyHandlerForTest(t)
	set := collections.Stores()
	ctx := context.Background()
	if _, err := set.CreateCollection(ctx, "acme", tenantDocsSchema); err != nil {
		t.Fatalf("create local acme collection on standby: %v", err)
	}
	localDoc := vcollection.Document{
		Vectors:  map[string]vcollection.Vector{"dense": vcollection.Vector{Dense: []float32{1, 0}}},
		Metadata: map[string]interface{}{"origin": "local"},
	}
	if err := set.AddDocument(ctx, "acme", "docs", &localDoc); err != nil {
		t.Fatalf("insert local acme document: %v", err)
	}

	st := startStandby(context.Background(), leaderURL, "node-token", 30*time.Millisecond, set, logging.Default())
	collections.SetStandby(st)

	waitForStandbyState(t, st, "acme", "stopped")
	waitForStandbyState(t, st, "globex", "streaming")

	if errText := st.snapshot()["acme"].err; !strings.Contains(errText, "acme") && !strings.Contains(strings.ToLower(errText), "store") {
		t.Errorf("acme stopped error = %q, want a store-mismatch explanation", errText)
	}

	// acme's own local data still answers reads...
	path := "/v3/tenants/acme/collections/docs/docs/" + strconv.FormatUint(localDoc.ID, 10)
	if resp := canonicalCall(t, handler, http.MethodGet, path, ""); resp.Code != http.StatusOK {
		t.Errorf("get local acme document = %d: %s", resp.Code, resp.Body.String())
	}
	// ...and its writes still succeed: it was never bound as a replica.
	resp := canonicalCall(t, handler, http.MethodPost, "/v3/tenants/acme/collections/docs/docs",
		`{"vectors":{"dense":[0,1]},"metadata":{"origin":"still-local"}}`)
	if resp.Code != http.StatusCreated && resp.Code != http.StatusOK {
		t.Errorf("write to the foreign local acme tenant = %d, want success: %s", resp.Code, resp.Body.String())
	}

	st.stop()
	if err := set.Close(); err != nil {
		t.Errorf("close standby store set: %v", err)
	}
}

// DEEPDATA_LEADER_URL without DEEPDATA_REPLICATION_TOKEN is a config error,
// not a silent no-op: without the node token a standby cannot authenticate
// to the leader it was just told to follow.
func TestServeFollowRequiresTheNodeToken(t *testing.T) {
	t.Setenv("DEEPDATA_LEADER_URL", "http://leader.internal:8080")
	t.Setenv("DEEPDATA_REPLICATION_TOKEN", "")
	_, errs := loadServerConfig(nil, os.Getenv)
	for _, e := range errs {
		if strings.Contains(e, "DEEPDATA_LEADER_URL") && strings.Contains(e, "DEEPDATA_REPLICATION_TOKEN") {
			return
		}
	}
	t.Fatalf("loadServerConfig errs = %v, want an error naming both DEEPDATA_LEADER_URL and DEEPDATA_REPLICATION_TOKEN", errs)
}
