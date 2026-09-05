package main

import (
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"github.com/phenomenon0/vectordb/internal/logging"
	"github.com/phenomenon0/vectordb/internal/replication"
)

// canonicalLeaderForTest builds the client handler over a real durable store.
func canonicalLeaderForTest(t *testing.T) (http.Handler, *CollectionHTTPServer, string) {
	t.Helper()
	t.Setenv("API_TOKEN", "client-api-token-for-tenants")
	t.Setenv("JWT_SECRET", "")
	t.Setenv("REQUIRE_AUTH", "1")
	indexPath := filepath.Join(t.TempDir(), "index.gob")
	handler, collections := newCanonicalHTTPHandler(newServerRuntime(), NewHashEmbedder(4), indexPath)
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
	t.Setenv(replicationTokenEnv, "")
	mounted, err := canonicalReplicationSurface(handler, collections, indexPath, logging.Default())
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
	t.Setenv(replicationTokenEnv, "node-token-distinct-from-the-client-one")
	mounted, err := canonicalReplicationSurface(handler, collections, indexPath, logging.Default())
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
	t.Setenv(replicationTokenEnv, "node-token-distinct-from-the-client-one")
	collections := NewCollectionHTTPServer(filepath.Join(t.TempDir(), "index.gob.collections"))
	if _, err := canonicalReplicationSurface(http.NotFoundHandler(), collections, "index.gob", logging.Default()); err == nil {
		t.Fatal("replication mounted on a process with no durable store")
	}
}
