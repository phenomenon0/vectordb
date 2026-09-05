package main

import (
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

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
	collections := NewCollectionHTTPServer(filepath.Join(t.TempDir(), "index.gob.collections"))
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
		t.Setenv("VECTORDB_MODE", "bogus")
		var rc int
		stderr := captureStderr(t, func() {
			rc = runReplicate([]string{"--leader", "http://leader.example"}, logging.Default())
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
			rc = runReplicate([]string{"--leader", "http://leader.example"}, logging.Default())
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
