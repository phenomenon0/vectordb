package main

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"testing"

	vcollection "github.com/phenomenon0/vectordb/internal/collection"
	"github.com/phenomenon0/vectordb/internal/sparse"
)

// legacyMigrationExport is the artifact the offline half of
// docs/upgrade-to-0.2-rc.md produces: one entry per legacy (tenant, collection)
// pair carrying the schema and every document. It is deliberately plain JSON —
// the migration hands data between two binaries that share no memory, so
// anything that only round-trips in-process would not be a rehearsal.
type legacyMigrationExport struct {
	Tenant     string                       `json:"tenant"`
	Collection string                       `json:"collection"`
	Schema     vcollection.CollectionSchema `json:"schema"`
	Documents  []vcollection.Document       `json:"documents"`
}

func legacyMigrationFixture() []legacyMigrationExport {
	sparseVector := func(index uint32) *sparse.SparseVector {
		return &sparse.SparseVector{Indices: []uint32{index}, Values: []float32{1}, Dim: 16}
	}
	return []legacyMigrationExport{
		{
			Tenant:     "acme",
			Collection: "docs",
			Schema: vcollection.CollectionSchema{
				Name: "docs",
				Fields: []vcollection.VectorField{
					{Name: "embedding", Type: vcollection.VectorTypeDense, Dim: 4,
						Index: vcollection.IndexConfig{Type: vcollection.IndexTypeHNSW}},
					{Name: "keywords", Type: vcollection.VectorTypeSparse, Dim: 16,
						Index: vcollection.IndexConfig{Type: vcollection.IndexTypeInverted}},
				},
			},
			Documents: []vcollection.Document{
				{ID: 101, Vectors: map[string]vcollection.Vector{
					"embedding": vcollection.Vector{Dense: []float32{1, 0, 0, 0}}, "keywords": vcollection.Vector{Sparse: sparseVector(2)}},
					Metadata: map[string]interface{}{"source": "legacy", "lang": "en", "rank": 1}},
				{ID: 102, Vectors: map[string]vcollection.Vector{
					"embedding": vcollection.Vector{Dense: []float32{0, 1, 0, 0}}, "keywords": vcollection.Vector{Sparse: sparseVector(5)}},
					Metadata: map[string]interface{}{"source": "legacy", "lang": "fr", "rank": 2}},
				{ID: 103, Vectors: map[string]vcollection.Vector{
					"embedding": vcollection.Vector{Dense: []float32{0, 0, 1, 0}}, "keywords": vcollection.Vector{Sparse: sparseVector(9)}},
					Metadata: map[string]interface{}{"source": "legacy", "lang": "de", "rank": 3}},
				{ID: 104, Vectors: map[string]vcollection.Vector{
					"embedding": vcollection.Vector{Dense: []float32{0, 0, 0, 1}}, "keywords": vcollection.Vector{Sparse: sparseVector(13)}},
					Metadata: map[string]interface{}{"source": "legacy", "lang": "es", "rank": 4}},
			},
		},
		{
			Tenant:     "globex",
			Collection: "notes",
			Schema: vcollection.CollectionSchema{
				Name: "notes",
				Fields: []vcollection.VectorField{
					{Name: "vector", Type: vcollection.VectorTypeDense, Dim: 4,
						Index: vcollection.IndexConfig{Type: vcollection.IndexTypeFLAT}},
				},
			},
			Documents: []vcollection.Document{
				{ID: 501, Vectors: map[string]vcollection.Vector{"vector": vcollection.Vector{Dense: []float32{1, 1, 0, 0}}},
					Metadata: map[string]interface{}{"owner": "ops"}},
				{ID: 502, Vectors: map[string]vcollection.Vector{"vector": vcollection.Vector{Dense: []float32{0, 1, 1, 0}}},
					Metadata: map[string]interface{}{"owner": "sre"}},
				{ID: 503, Vectors: map[string]vcollection.Vector{"vector": vcollection.Vector{Dense: []float32{0, 0, 1, 1}}},
					Metadata: map[string]interface{}{"owner": "eng"}},
			},
		},
	}
}

// writeLegacyV2Root builds a pre-0.2 data directory the way a 0.1 process left
// one: a .manager half for the default namespace and a .tenants half for every
// tenant. Both are written because a migration that reads only one silently
// drops data.
func writeLegacyV2Root(t *testing.T, dataDir string) {
	t.Helper()
	ctx := context.Background()
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		t.Fatal(err)
	}
	base := filepath.Join(dataDir, "index.gob.collections")

	manager := vcollection.NewCollectionManager(base)
	if err := manager.Save(base + ".manager"); err != nil {
		t.Fatal(err)
	}

	tenants := vcollection.NewTenantManager(base)
	for _, fixture := range legacyMigrationFixture() {
		if _, err := tenants.CreateCollection(ctx, fixture.Tenant, fixture.Schema); err != nil {
			t.Fatal(err)
		}
		for i := range fixture.Documents {
			doc := fixture.Documents[i]
			if err := tenants.AddDocument(ctx, fixture.Tenant, fixture.Collection, &doc); err != nil {
				t.Fatal(err)
			}
		}
	}
	if err := tenants.Save(base + ".tenants"); err != nil {
		t.Fatal(err)
	}
}

// exportLegacyV2Root is the offline export step. It reads the legacy root with
// the legacy managers only — the release candidate never opens it, which is the
// whole point of the procedure.
func exportLegacyV2Root(t *testing.T, dataDir string) []legacyMigrationExport {
	t.Helper()
	base := filepath.Join(dataDir, "index.gob.collections")
	tenants := vcollection.NewTenantManager(base)
	if err := tenants.Load(base + ".tenants"); err != nil {
		t.Fatal(err)
	}

	var exports []legacyMigrationExport
	for _, tenantID := range tenants.ListTenants() {
		for _, name := range tenants.ListCollections(tenantID) {
			coll, err := tenants.GetCollection(tenantID, name)
			if err != nil {
				t.Fatal(err)
			}
			stored := coll.ExportDocuments()
			ids := make([]uint64, 0, len(stored))
			for id := range stored {
				ids = append(ids, id)
			}
			sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
			documents := make([]vcollection.Document, 0, len(ids))
			for _, id := range ids {
				documents = append(documents, *stored[id])
			}
			exports = append(exports, legacyMigrationExport{
				Tenant:     tenantID,
				Collection: name,
				Schema:     coll.Schema(),
				Documents:  documents,
			})
		}
	}
	sort.Slice(exports, func(i, j int) bool {
		if exports[i].Tenant != exports[j].Tenant {
			return exports[i].Tenant < exports[j].Tenant
		}
		return exports[i].Collection < exports[j].Collection
	})
	return exports
}

// asWireJSON pushes a value through JSON so expectations are compared in the
// same representation the server answers in (integers arrive as float64).
func asWireJSON(t *testing.T, value interface{}) map[string]interface{} {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	var decoded map[string]interface{}
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	return decoded
}

func legacyMigrationDocCount(t *testing.T, collectionURL string) int {
	t.Helper()
	response, body := canonicalJSONRequest(t, http.MethodGet, collectionURL, nil)
	if response.StatusCode != http.StatusOK {
		t.Fatalf("get migrated collection returned %d: %s", response.StatusCode, body)
	}
	var result struct {
		Collection struct {
			DocCount int `json:"doc_count"`
		} `json:"collection"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		t.Fatal(err)
	}
	return result.Collection.DocCount
}

func legacyMigrationSearchTop(t *testing.T, collectionURL string, query []byte) vcollection.Document {
	t.Helper()
	response, body := canonicalJSONRequest(t, http.MethodPost, collectionURL+"/search", query)
	if response.StatusCode != http.StatusOK {
		t.Fatalf("search migrated collection returned %d: %s", response.StatusCode, body)
	}
	var result tenantSearchJSONResponse
	if err := json.Unmarshal(body, &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Documents) == 0 {
		t.Fatalf("search returned no documents for %s: %s", query, body)
	}
	return result.Documents[0]
}

// TestLegacyV2ExportImportRoundTrip rehearses the offline migration in
// docs/upgrade-to-0.2-rc.md end to end: a pre-0.2 data directory is exported
// without the release candidate ever opening it, replayed into a fresh
// directory through canonical V3 mutations, and read back after a restart.
//
// The assertions are the promises an operator actually depends on. Document IDs
// survive, because external systems reference them. Metadata survives verbatim,
// because it carries the caller's own keys. Every vector still retrieves its own
// document, dense and sparse alike, because a migration that preserved counts
// while corrupting vectors would pass a count check and fail in production. And
// the legacy root is byte-identical afterwards, because the procedure's
// fallback is that untouched directory.
func TestLegacyV2ExportImportRoundTrip(t *testing.T) {
	legacyDir := filepath.Join(t.TempDir(), "legacy")
	writeLegacyV2Root(t, legacyDir)

	legacyBase := filepath.Join(legacyDir, "index.gob.collections")
	legacyArtifacts := []string{legacyBase + ".manager", legacyBase + ".tenants"}
	beforeExport := make(map[string][32]byte, len(legacyArtifacts))
	for _, path := range legacyArtifacts {
		beforeExport[path] = testFileSHA256(t, path)
	}

	exports := exportLegacyV2Root(t, legacyDir)
	expected := legacyMigrationFixture()
	if len(exports) != len(expected) {
		t.Fatalf("exported %d collections, want %d", len(exports), len(expected))
	}
	for _, path := range legacyArtifacts {
		if got := testFileSHA256(t, path); got != beforeExport[path] {
			t.Fatalf("offline export modified the legacy root artifact %s", path)
		}
	}

	migratedDir := filepath.Join(t.TempDir(), "state")
	httpAddress := unusedLoopbackAddress(t)
	grpcAddress := unusedLoopbackAddress(t)
	process := startCanonicalTestProcess(t, migratedDir, httpAddress, grpcAddress)
	defer func() { process.stopIfRunning() }()
	process.waitReady(t, httpAddress)

	for i, export := range exports {
		want := expected[i]
		if export.Tenant != want.Tenant || export.Collection != want.Collection {
			t.Fatalf("export %d = %s/%s, want %s/%s",
				i, export.Tenant, export.Collection, want.Tenant, want.Collection)
		}
		if len(export.Documents) != len(want.Documents) {
			t.Fatalf("%s/%s exported %d documents, want %d",
				export.Tenant, export.Collection, len(export.Documents), len(want.Documents))
		}

		tenantURL := "http://" + httpAddress + "/v3/tenants/" + export.Tenant + "/collections"
		schemaBody, err := json.Marshal(export.Schema)
		if err != nil {
			t.Fatal(err)
		}
		response, body := canonicalJSONRequest(t, http.MethodPost, tenantURL, schemaBody)
		if response.StatusCode != http.StatusCreated {
			t.Fatalf("create migrated collection returned %d: %s", response.StatusCode, body)
		}

		batchBody, err := json.Marshal(map[string]interface{}{"documents": export.Documents})
		if err != nil {
			t.Fatal(err)
		}
		documentsURL := tenantURL + "/" + export.Collection + "/docs"
		response, body = canonicalJSONRequest(t, http.MethodPost, documentsURL+"/batch", batchBody)
		if response.StatusCode != http.StatusOK {
			t.Fatalf("import migrated documents returned %d: %s", response.StatusCode, body)
		}
		var batchResult struct {
			IDs      []uint64 `json:"ids"`
			Inserted int      `json:"inserted"`
		}
		if err := json.Unmarshal(body, &batchResult); err != nil {
			t.Fatal(err)
		}
		wantIDs := make([]uint64, len(export.Documents))
		for j := range export.Documents {
			wantIDs[j] = export.Documents[j].ID
		}
		if batchResult.Inserted != len(wantIDs) || !reflect.DeepEqual(batchResult.IDs, wantIDs) {
			t.Fatalf("%s/%s import acknowledged %+v, want ids %v",
				export.Tenant, export.Collection, batchResult, wantIDs)
		}
	}

	// The RC has to be able to serve the migrated state from disk alone, so
	// every check below runs against a process that never saw the import.
	process.terminate(t)
	process = startCanonicalTestProcess(t, migratedDir, httpAddress, grpcAddress)
	process.waitReady(t, httpAddress)

	for _, export := range exports {
		collectionURL := "http://" + httpAddress + "/v3/tenants/" + export.Tenant +
			"/collections/" + export.Collection
		if got := legacyMigrationDocCount(t, collectionURL); got != len(export.Documents) {
			t.Fatalf("%s/%s migrated document count = %d, want %d",
				export.Tenant, export.Collection, got, len(export.Documents))
		}

		for _, document := range export.Documents {
			for _, field := range export.Schema.Fields {
				vector, ok := document.Vectors[field.Name]
				if !ok {
					continue
				}
				query, err := json.Marshal(map[string]interface{}{
					"queries": map[string]interface{}{field.Name: vector},
					"top_k":   1,
				})
				if err != nil {
					t.Fatal(err)
				}
				top := legacyMigrationSearchTop(t, collectionURL, query)
				if top.ID != document.ID {
					t.Fatalf("%s/%s field %s: nearest document to %d's own vector is %d",
						export.Tenant, export.Collection, field.Name, document.ID, top.ID)
				}
				gotMetadata := asWireJSON(t, top.Metadata)
				wantMetadata := asWireJSON(t, document.Metadata)
				if !reflect.DeepEqual(gotMetadata, wantMetadata) {
					t.Fatalf("%s/%s document %d metadata = %v, want %v",
						export.Tenant, export.Collection, document.ID, gotMetadata, wantMetadata)
				}
			}
		}
	}

	for _, path := range legacyArtifacts {
		if got := testFileSHA256(t, path); got != beforeExport[path] {
			t.Fatalf("migration modified the legacy root artifact %s", path)
		}
	}

	process.terminate(t)
}
