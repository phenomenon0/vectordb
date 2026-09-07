package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	vcollection "github.com/phenomenon0/vectordb/internal/collection"
	"github.com/phenomenon0/vectordb/internal/replication"
)

// replicaDirectoryForTest produces a real replica directory the way an
// operator does -- `deepdata replicate --tenant acme` against a running
// leader with two tenants -- and returns the index path a later `deepdata
// serve` would be pointed at, plus the leader-minted document ID that must be
// readable through it. globex exists on the leader but is never replicated,
// so it must come out unknown on the replica.
//
// It goes through the real Follower and a real socket rather than calling
// MakeReplica by hand: the thing under test is whether a directory that was
// synced by one process is still read-only when a different process opens it,
// and a hand-marked in-memory store would answer a question nobody asks.
func replicaDirectoryForTest(t *testing.T) (indexPath string, docID uint64) {
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

	schema := vcollection.CollectionSchema{
		Name: "docs",
		Fields: []vcollection.VectorField{{
			Name:  "dense",
			Type:  vcollection.VectorTypeDense,
			Dim:   2,
			Index: vcollection.IndexConfig{Type: vcollection.IndexTypeFLAT},
		}},
	}
	if _, err := leader.CreateCollection(ctx, "acme", schema); err != nil {
		t.Fatalf("create collection for acme on leader: %v", err)
	}
	doc := vcollection.Document{
		Vectors:  map[string]vcollection.Vector{"dense": vcollection.Vector{Dense: []float32{1, 0}}},
		Metadata: map[string]interface{}{"origin": "leader"},
	}
	if err := leader.AddDocument(ctx, "acme", "docs", &doc); err != nil {
		t.Fatalf("insert on leader: %v", err)
	}
	if _, err := leader.CreateCollection(ctx, "globex", schema); err != nil {
		t.Fatalf("create collection for globex on leader: %v", err)
	}

	node, err := replication.NewTenantLeaderHandler(replication.LeaderConfig{Token: "node-token", SpoolDir: dir},
		leader.Tenants, func(id string) (replication.Source, bool) { return leader.Store(id) })
	if err != nil {
		t.Fatalf("build node surface: %v", err)
	}
	srv := httptest.NewServer(node)
	defer srv.Close()

	indexPath = filepath.Join(dir, "replica.gob")
	tenantDir := indexPath + ".tenants"
	if err := os.MkdirAll(tenantDir, 0o750); err != nil {
		t.Fatalf("create replica tenant store directory: %v", err)
	}
	base := filepath.Join(tenantDir, "acme")
	follower := &replication.Follower{LeaderURL: srv.URL, Token: "node-token", Tenant: "acme"}
	replica, err := follower.Open(ctx, base, base)
	if err != nil {
		t.Fatalf("sync replica directory: %v", err)
	}
	if !replica.IsReplica() {
		t.Fatal("freshly synced store is not a replica")
	}
	// The syncing process stops here. Everything after this point is a
	// separate process opening the directory it left behind.
	if err := replica.Close(); err != nil {
		t.Fatalf("close replica after sync: %v", err)
	}
	return indexPath, doc.ID
}

func canonicalCall(t *testing.T, handler http.Handler, method, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	handler.ServeHTTP(resp, req)
	return resp
}

// A replica directory is safe to serve for reads and must never take a write.
//
// 403 rather than 500 is the whole point: 500 says "something broke, retry",
// and a client SDK will. Nothing about this node will ever accept the write,
// so the answer has to be a classifiable refusal a caller can route on -- and
// the store must be byte-for-byte unchanged afterwards, because a write that
// landed would consume the LSN the leader's next record needs and wedge the
// replica permanently.
func TestServingAReplicaRefusesEveryWriteWithoutTouchingTheStore(t *testing.T) {
	indexPath, docID := replicaDirectoryForTest(t)
	handler := newCanonicalSurfaceTestHandlerAt(t, indexPath)

	writes := []struct {
		name   string
		method string
		path   string
		body   string
	}{
		{"create collection", http.MethodPost, "/v3/tenants/acme/collections", `{"name":"more","fields":[{"name":"dense","type":"dense","dim":2,"index":{"type":"flat"}}]}`},
		{"insert document", http.MethodPost, "/v3/tenants/acme/collections/docs/docs", `{"vectors":{"dense":[0,1]},"metadata":{"origin":"intruder"}}`},
		{"upsert document", http.MethodPut, "/v3/tenants/acme/collections/docs/docs/9", `{"vectors":{"dense":[0,1]}}`},
		{"batch insert", http.MethodPost, "/v3/tenants/acme/collections/docs/docs/batch", `{"documents":[{"vectors":{"dense":[0,1]}}]}`},
		{"delete document", http.MethodDelete, "/v3/tenants/acme/collections/docs/docs", `{"doc_id":` + strconv.FormatUint(docID, 10) + `}`},
		{"delete collection", http.MethodDelete, "/v3/tenants/acme/collections/docs", ""},
	}
	for _, w := range writes {
		resp := canonicalCall(t, handler, w.method, w.path, w.body)
		if resp.Code != http.StatusForbidden {
			t.Errorf("%s on a replica = %d, want 403: %s", w.name, resp.Code, resp.Body.String())
			continue
		}
		var envelope struct {
			Code      string `json:"code"`
			Retryable bool   `json:"retryable"`
		}
		if err := json.Unmarshal(resp.Body.Bytes(), &envelope); err != nil {
			t.Errorf("%s: refusal is not the error envelope: %s", w.name, resp.Body.String())
			continue
		}
		if envelope.Code != "permission_denied" {
			t.Errorf("%s: code = %q, want permission_denied", w.name, envelope.Code)
		}
		if envelope.Retryable {
			t.Errorf("%s: a replica will never accept this write; reporting it retryable makes an SDK loop forever", w.name)
		}
	}

	// Nothing the refused writes named may exist, and what the leader wrote
	// must still be exactly there.
	resp := canonicalCall(t, handler, http.MethodGet, "/v3/tenants/acme/collections", "")
	if resp.Code != http.StatusOK {
		t.Fatalf("list collections on a replica = %d: %s", resp.Code, resp.Body.String())
	}
	var listed struct {
		Collections []struct {
			Name          string `json:"name"`
			DocumentCount int    `json:"document_count"`
		} `json:"collections"`
	}
	if err := json.Unmarshal(resp.Body.Bytes(), &listed); err != nil {
		t.Fatal(err)
	}
	if len(listed.Collections) != 1 || listed.Collections[0].Name != "docs" {
		t.Fatalf("replica holds %+v, want exactly the leader's one collection", listed.Collections)
	}
	if resp := canonicalCall(t, handler, http.MethodGet, "/v3/tenants/acme/collections/docs/docs/"+strconv.FormatUint(docID, 10), ""); resp.Code != http.StatusOK {
		t.Errorf("the leader's document is gone after a refused delete: %d %s", resp.Code, resp.Body.String())
	}
}

// A read replica is only worth serving if it serves. The reads have to answer
// from the leader's own state, under the leader's own document IDs -- a
// replica that re-mints IDs returns different data under the same key while
// every count-based health check reports agreement.
func TestServingAReplicaStillAnswersReads(t *testing.T) {
	indexPath, docID := replicaDirectoryForTest(t)
	handler := newCanonicalSurfaceTestHandlerAt(t, indexPath)

	resp := canonicalCall(t, handler, http.MethodGet, "/v3/tenants/acme/collections/docs/docs/"+strconv.FormatUint(docID, 10), "")
	if resp.Code != http.StatusOK {
		t.Fatalf("get leader document %d from a replica = %d: %s", docID, resp.Code, resp.Body.String())
	}
	if !strings.Contains(resp.Body.String(), `"leader"`) {
		t.Errorf("replica returned a document that is not the leader's: %s", resp.Body.String())
	}

	resp = canonicalCall(t, handler, http.MethodPost, "/v3/tenants/acme/collections/docs/search", `{"queries":{"dense":[1,0]},"top_k":1}`)
	if resp.Code != http.StatusOK {
		t.Fatalf("search a replica = %d: %s", resp.Code, resp.Body.String())
	}
	var search struct {
		Documents []struct {
			ID uint64 `json:"id"`
		} `json:"documents"`
	}
	if err := json.Unmarshal(resp.Body.Bytes(), &search); err != nil {
		t.Fatal(err)
	}
	if len(search.Documents) != 1 || search.Documents[0].ID != docID {
		t.Errorf("search returned %+v, want the leader-minted id %d", search.Documents, docID)
	}

	if resp := canonicalCall(t, handler, http.MethodGet, "/v3/tenants/acme/collections/docs", ""); resp.Code != http.StatusOK {
		t.Errorf("collection info on a replica = %d: %s", resp.Code, resp.Body.String())
	}

	// globex exists on the leader but "replicate --tenant acme" only synced
	// acme's tenant directory, so this replica never opened a store for it.
	if resp := canonicalCall(t, handler, http.MethodGet, "/v3/tenants/globex/collections/docs", ""); resp.Code != http.StatusNotFound {
		t.Errorf("unreplicated tenant globex on this replica = %d, want 404: %s", resp.Code, resp.Body.String())
	}
}

// The guard is scoped to replicas, not to the write paths. A normal directory
// must answer exactly as it did before this feature existed.
func TestServingANonReplicaDirectoryStillAcceptsWrites(t *testing.T) {
	handler := newCanonicalSurfaceTestHandlerAt(t, filepath.Join(t.TempDir(), "index.gob"))
	create := `{"name":"docs","fields":[{"name":"dense","type":"dense","dim":2,"index":{"type":"flat"}}]}`
	if resp := canonicalCall(t, handler, http.MethodPost, "/v3/tenants/acme/collections", create); resp.Code != http.StatusCreated {
		t.Fatalf("create on a normal store = %d, want 201: %s", resp.Code, resp.Body.String())
	}
	if resp := canonicalCall(t, handler, http.MethodPut, "/v3/tenants/acme/collections/docs/docs/1", `{"vectors":{"dense":[1,0]}}`); resp.Code != http.StatusOK {
		t.Fatalf("upsert on a normal store = %d, want 200: %s", resp.Code, resp.Body.String())
	}
}

// Readiness is what a load balancer reads. A replica is ready -- it serves
// reads correctly -- so it must stay 200 and must not be pulled from the
// pool; what it must not do is let the pool believe it can take writes.
func TestReadinessReportsReadOnlyOnlyForAReplica(t *testing.T) {
	readiness := func(handler http.Handler) (int, map[string]any) {
		resp := canonicalCall(t, handler, http.MethodGet, "/readyz", "")
		var body map[string]any
		if err := json.Unmarshal(resp.Body.Bytes(), &body); err != nil {
			t.Fatalf("/readyz body is not JSON: %s", resp.Body.String())
		}
		return resp.Code, body
	}

	indexPath, _ := replicaDirectoryForTest(t)
	code, body := readiness(newCanonicalSurfaceTestHandlerAt(t, indexPath))
	if code != http.StatusOK {
		t.Fatalf("/readyz on a replica = %d, want 200; a replica serves reads: %v", code, body)
	}
	if body["read_only"] != true {
		t.Errorf("/readyz read_only = %v on a replica, want true: %v", body["read_only"], body)
	}

	code, body = readiness(newCanonicalSurfaceTestHandlerAt(t, filepath.Join(t.TempDir(), "index.gob")))
	if code != http.StatusOK {
		t.Fatalf("/readyz on a normal store = %d, want 200: %v", code, body)
	}
	if body["read_only"] != false {
		t.Errorf("/readyz read_only = %v on a writable store, want false: %v", body["read_only"], body)
	}
}
