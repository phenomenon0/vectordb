package collection

import (
	"context"
	"encoding/json"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/phenomenon0/vectordb/internal/sparse"
)

// legacyEncodeDurableMutationReference reproduces the pre-single-pass encoder:
// it marshaled the typed payload to an intermediate buffer and wrapped those
// bytes in durableMutationEnvelope for a second pass. Journals written by that
// encoder are the compatibility baseline the current encoder must match.
func legacyEncodeDurableMutationReference(t *testing.T, m canonicalMutation, version uint16) []byte {
	t.Helper()
	var payload any
	switch m.typeName {
	case mutationCreateCollection:
		payload = durableCreateCollection{TenantID: m.tenantID, Schema: m.schema}
	case mutationDeleteCollection:
		payload = durableCollectionTarget{TenantID: m.tenantID, CollectionName: m.collectionName}
	case mutationInsertDocument:
		if len(m.documents) != 1 {
			t.Fatalf("reference encoder: insert mutation must contain exactly one document, got %d", len(m.documents))
		}
		payload = durableInsertDocument{TenantID: m.tenantID, CollectionName: m.collectionName, Document: m.documents[0]}
	case mutationBatchInsert:
		payload = durableBatchInsert{TenantID: m.tenantID, CollectionName: m.collectionName, Documents: m.documents}
	case mutationUpsertDocument:
		if len(m.documents) != 1 {
			t.Fatalf("reference encoder: upsert mutation must contain exactly one document, got %d", len(m.documents))
		}
		payload = durableUpsertDocument{TenantID: m.tenantID, CollectionName: m.collectionName, Document: m.documents[0]}
	case mutationDeleteDocument:
		payload = durableDeleteDocument{TenantID: m.tenantID, CollectionName: m.collectionName, DocumentID: m.documentID}
	case mutationCreateTenant, mutationUpdateTenant:
		payload = m.record
	case mutationDeleteTenant:
		payload = durableTenantTarget{TenantID: m.tenantID}
	default:
		t.Fatalf("reference encoder: unknown mutation type %q", m.typeName)
	}
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal reference payload: %v", err)
	}
	data, err := json.Marshal(durableMutationEnvelope{Version: version, Type: m.typeName, Payload: payloadBytes})
	if err != nil {
		t.Fatalf("marshal reference envelope: %v", err)
	}
	return data
}

func goldenTestSchema() CollectionSchema {
	return CollectionSchema{
		Name: "golden-docs",
		Fields: []VectorField{
			{
				Name: "embedding",
				Type: VectorTypeDense,
				Dim:  4,
				Index: IndexConfig{
					Type:   IndexTypeHNSW,
					Params: map[string]interface{}{"m": float64(16), "ef_construction": float64(200)},
				},
			},
			{
				Name: "keywords",
				Type: VectorTypeSparse,
				Dim:  1024,
				Index: IndexConfig{
					Type:   IndexTypeInverted,
					Params: map[string]interface{}{"k1": float64(1.2), "b": float64(0.75)},
				},
			},
		},
	}
}

// goldenTestDocument deliberately mixes representation shapes that stress the
// envelope bytes: Go-typed vectors, a *sparse.SparseVector whose MarshalJSON
// key order differs from its normalized map form, HTML-escapable strings, and
// nested metadata containers.
func goldenTestDocument(id uint64) Document {
	return Document{
		ID: id,
		Vectors: map[string]Vector{
			"embedding": Vector{Dense: []float32{0.25, 0.5, 0.75, 1}},
			"keywords": Vector{Sparse: &sparse.SparseVector{
				Indices: []uint32{1, 5},
				Values:  []float32{0.5, 2},
				Dim:     1024,
			}},
		},
		Metadata: map[string]interface{}{
			"title":  "alpha & <beta>",
			"weight": float64(0.125),
			"flags":  []interface{}{true, nil, "x"},
			"nested": map[string]interface{}{"depth": float64(2), "labels": []interface{}{"l1", "l2"}},
		},
	}
}

// expectedDecodedGoldenDocument mirrors what decodeCollectionJSON must produce
// for goldenTestDocument: all numbers become float64 and the sparse vector
// becomes its plain map form. Building the expectation by hand (rather than by
// re-decoding the input) keeps the round-trip assertion independent of the
// code under test.
func expectedDecodedGoldenDocument(id uint64) Document {
	return Document{
		ID: id,
		Vectors: map[string]Vector{
			"embedding": {Dense: []float32{0.25, 0.5, 0.75, 1}},
			"keywords": {Sparse: &sparse.SparseVector{
				Indices: []uint32{1, 5},
				Values:  []float32{0.5, 2},
				Dim:     1024,
			}},
		},
		Metadata: map[string]interface{}{
			"title":  "alpha & <beta>",
			"weight": float64(0.125),
			"flags":  []interface{}{true, nil, "x"},
			"nested": map[string]interface{}{"depth": float64(2), "labels": []interface{}{"l1", "l2"}},
		},
	}
}

func TestDurableMutationEncodingSinglePassMatchesLegacyBytes(t *testing.T) {
	schema := goldenTestSchema()
	docA := goldenTestDocument(7)
	docB := goldenTestDocument(9)
	docB.Vectors = map[string]Vector{"embedding": {Dense: []float32{-0.5, -0.25, 0.125, 0}}}

	mutations := []struct {
		name    string
		version uint16
		before  canonicalMutation
		verify  func(t *testing.T, decoded canonicalMutation)
	}{
		{
			name:    "create collection",
			version: durableCollectionMutationVersion,
			before: canonicalMutation{
				typeName: mutationCreateCollection, tenantID: "tenant-a", collectionName: "golden-docs", schema: schema,
			},
			verify: func(t *testing.T, decoded canonicalMutation) {
				if decoded.schema.Name != schema.Name || !reflect.DeepEqual(decoded.schema.Fields, schema.Fields) {
					t.Fatalf("decoded schema = %+v, want %+v", decoded.schema, schema)
				}
			},
		},
		{
			name:    "batch insert",
			version: durableCollectionMutationVersion,
			before: canonicalMutation{
				typeName: mutationBatchInsert, tenantID: "tenant-a", collectionName: "golden-docs",
				documents: []Document{docA, docB}, nextID: 10,
			},
			verify: func(t *testing.T, decoded canonicalMutation) {
				want := []Document{expectedDecodedGoldenDocument(7), {
					ID:       9,
					Vectors:  map[string]Vector{"embedding": {Dense: []float32{-0.5, -0.25, 0.125, 0}}},
					Metadata: expectedDecodedGoldenDocument(9).Metadata,
				}}
				if !reflect.DeepEqual(decoded.documents, want) {
					t.Fatalf("decoded documents mismatch:\n got %#v\nwant %#v", decoded.documents, want)
				}
			},
		},
		{
			name:    "insert document",
			version: durableCollectionMutationVersion,
			before: canonicalMutation{
				typeName: mutationInsertDocument, tenantID: "tenant-a", collectionName: "golden-docs",
				documents: []Document{docA}, nextID: 11,
			},
			verify: func(t *testing.T, decoded canonicalMutation) {
				want := []Document{expectedDecodedGoldenDocument(7)}
				if !reflect.DeepEqual(decoded.documents, want) {
					t.Fatalf("decoded documents mismatch:\n got %#v\nwant %#v", decoded.documents, want)
				}
			},
		},
		{
			name:    "upsert document",
			version: durableCollectionMutationVersion,
			before: canonicalMutation{
				typeName: mutationUpsertDocument, tenantID: "tenant-a", collectionName: "golden-docs",
				documents: []Document{docA}, nextID: 12,
			},
			verify: func(t *testing.T, decoded canonicalMutation) {
				want := []Document{expectedDecodedGoldenDocument(7)}
				if !reflect.DeepEqual(decoded.documents, want) {
					t.Fatalf("decoded documents mismatch:\n got %#v\nwant %#v", decoded.documents, want)
				}
			},
		},
		{
			name:    "delete document",
			version: durableCollectionMutationVersion,
			before: canonicalMutation{
				typeName: mutationDeleteDocument, tenantID: "tenant-a", collectionName: "golden-docs", documentID: 7,
			},
			verify: func(t *testing.T, decoded canonicalMutation) {
				if decoded.documentID != 7 {
					t.Fatalf("decoded document ID = %d, want 7", decoded.documentID)
				}
			},
		},
		{
			name:    "delete collection legacy v1",
			version: durableCollectionMutationVersionV1,
			before: canonicalMutation{
				typeName: mutationDeleteCollection, tenantID: "tenant-a", collectionName: "golden-docs",
			},
			verify: func(t *testing.T, decoded canonicalMutation) {
				if decoded.version != durableCollectionMutationVersionV1 {
					t.Fatalf("decoded version = %d, want v1 passthrough", decoded.version)
				}
			},
		},
		{
			name:    "create tenant",
			version: durableCollectionMutationVersion,
			before: canonicalMutation{
				typeName: mutationCreateTenant, tenantID: "tenant-a",
				record: TenantRecord{TenantID: "tenant-a", Status: TenantStatusActive, Quota: TenantQuota{MaxDocuments: 100}},
			},
			verify: func(t *testing.T, decoded canonicalMutation) {
				want := TenantRecord{TenantID: "tenant-a", Status: TenantStatusActive, Quota: TenantQuota{MaxDocuments: 100}}
				if decoded.record != want {
					t.Fatalf("decoded record = %+v, want %+v", decoded.record, want)
				}
			},
		},
		{
			name:    "update tenant",
			version: durableCollectionMutationVersion,
			before: canonicalMutation{
				typeName: mutationUpdateTenant, tenantID: "tenant-a",
				record: TenantRecord{TenantID: "tenant-a", Status: TenantStatusSuspended},
			},
			verify: func(t *testing.T, decoded canonicalMutation) {
				want := TenantRecord{TenantID: "tenant-a", Status: TenantStatusSuspended}
				if decoded.record != want {
					t.Fatalf("decoded record = %+v, want %+v", decoded.record, want)
				}
			},
		},
		{
			name:    "delete tenant",
			version: durableCollectionMutationVersion,
			before: canonicalMutation{
				typeName: mutationDeleteTenant, tenantID: "tenant-a",
			},
			verify: func(t *testing.T, decoded canonicalMutation) {
				if decoded.tenantID != "tenant-a" {
					t.Fatalf("decoded tenant id = %q, want tenant-a", decoded.tenantID)
				}
			},
		},
	}

	for _, tc := range mutations {
		t.Run(tc.name, func(t *testing.T) {
			current, err := encodeDurableMutationVersion(tc.before, tc.version)
			if err != nil {
				t.Fatalf("encode durable mutation: %v", err)
			}
			legacy := legacyEncodeDurableMutationReference(t, tc.before, tc.version)
			if string(current) != string(legacy) {
				t.Fatalf("single-pass bytes diverged from legacy encoder:\n new: %s\nlegacy: %s", current, legacy)
			}
			decoded, err := decodeDurableMutation(current)
			if err != nil {
				t.Fatalf("decode single-pass bytes: %v", err)
			}
			if decoded.version != tc.version ||
				decoded.typeName != tc.before.typeName ||
				decoded.tenantID != tc.before.tenantID ||
				decoded.collectionName != tc.before.collectionName {
				t.Fatalf("decoded envelope = %+v, want %+v", decoded, tc.before)
			}
			tc.verify(t, decoded)
		})
	}
}

// TestDocumentClonerIsolatesCallerCompositeTypes pins the deep-copy contract
// of cloneDocumentPreservingTypes across the whole value vocabulary callers
// can reach, including the reflection fallback for exotic composite types.
func TestDocumentClonerIsolatesCallerCompositeTypes(t *testing.T) {
	inputVector := []float32{1, 2, 3, 4}
	inputTags := []string{"keep"}
	inputItems := []interface{}{float64(1), map[string]interface{}{"inner": "value"}}
	inputDeep := map[string]interface{}{"list": []interface{}{float64(9)}}
	inputSparse := &sparse.SparseVector{Indices: []uint32{3, 7}, Values: []float32{0.25, 0.75}, Dim: 16}
	inputGrid := [][]float32{{1, 2}, {3, 4}}
	doc := Document{
		ID: 1,
		Vectors: map[string]Vector{
			"embedding": Vector{Dense: inputVector},
			"keywords":  Vector{Sparse: inputSparse},
		},
		Metadata: map[string]interface{}{
			"tags":  inputTags,
			"items": inputItems,
			"deep":  inputDeep,
			"grid":  inputGrid,
		},
	}

	clone := cloneDocumentPreservingTypes(doc)

	// Mutate every reachable caller-owned container after the clone.
	inputVector[0] = 999
	inputTags[0] = "mutated"
	inputItems[0] = float64(999)
	inputItems[1].(map[string]interface{})["inner"] = "mutated"
	inputDeep["list"].([]interface{})[0] = float64(999)
	inputDeep["added"] = "later"
	inputSparse.Values[0] = 999
	inputGrid[0][0] = 999

	if got := clone.Vectors["embedding"].Dense; got[0] != 1 {
		t.Fatalf("caller dense-vector mutation leaked into clone: %v", got)
	}
	if got, ok := clone.Metadata["tags"].([]string); ok && got[0] != "keep" {
		t.Fatalf("caller []string mutation leaked into clone: %v", got)
	} else if !ok {
		t.Fatalf("clone dropped []string metadata type: %T", clone.Metadata["tags"])
	}
	items := clone.Metadata["items"].([]interface{})
	if items[0] != float64(1) {
		t.Fatalf("caller []interface{} element mutation leaked into clone: %v", items[0])
	}
	if got := items[1].(map[string]interface{})["inner"]; got != "value" {
		t.Fatalf("caller nested-map mutation leaked into clone: %v", got)
	}
	deep := clone.Metadata["deep"].(map[string]interface{})
	if got := deep["list"].([]interface{})[0]; got != float64(9) {
		t.Fatalf("caller nested-list mutation leaked into clone: %v", got)
	}
	if _, leaked := deep["added"]; leaked {
		t.Fatal("caller map insertion leaked into clone")
	}
	if got := clone.Vectors["keywords"].Sparse; got.Values[0] != 0.25 {
		t.Fatalf("caller sparse-vector mutation leaked into clone: %v", got.Values)
	}
	grid := clone.Metadata["grid"].([][]float32)
	if grid[0][0] != 1 {
		t.Fatalf("caller nested-slice mutation leaked into clone: %v", grid)
	}

	// Reverse direction: mutating the CLONE must not reach the caller either.
	clone.Vectors["embedding"].Dense[1] = -999
	clone.Metadata["tags"].([]string)[0] = "clone-write"
	if inputVector[1] != 2 || inputTags[0] != "mutated" {
		t.Fatalf("clone mutation reached caller state: vector=%v tags=%v", inputVector, inputTags)
	}
}

// TestDurableStoreBatchInsertIsolatesCallerState proves the canonical batch
// insert path stores a fully isolated snapshot of caller-owned vectors and
// metadata, preserves Go-typed vectors in memory, and leaves a replay-
// equivalent record behind after checkpoint and reload.
func TestDurableStoreBatchInsertIsolatesCallerState(t *testing.T) {
	ctx := context.Background()
	base := filepath.Join(t.TempDir(), "collections")
	store, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	tenants := store.Tenants()
	if _, err := tenants.CreateCollection(ctx, "t", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}

	callerVector := []float32{1, 2, 3, 4}
	callerMetadata := map[string]interface{}{
		"count": float64(42),
		"tags":  []string{"keep"},
		"deep":  map[string]interface{}{"list": []interface{}{"original"}},
	}
	batch := []Document{
		{Vectors: map[string]Vector{"embedding": {Dense: callerVector}}, Metadata: callerMetadata},
		{ID: 50, Vectors: map[string]Vector{"embedding": Vector{Dense: []float32{5, 6, 7, 8}}}},
	}
	if err := tenants.BatchAddDocuments(ctx, "t", "docs", batch); err != nil {
		t.Fatal(err)
	}
	if batch[0].ID == 0 {
		t.Fatal("batch insert did not surface assigned ID")
	}

	// Mutate every caller-owned structure after the committed insert.
	callerVector[0] = 999
	callerMetadata["count"] = float64(-1)
	callerMetadata["tags"].([]string)[0] = "mutated"
	callerMetadata["deep"].(map[string]interface{})["list"].([]interface{})[0] = "mutated"
	delete(callerMetadata, "tags")
	batch[0].Vectors = nil
	batch[0].Metadata = map[string]interface{}{"count": "replaced"}

	stored, ok := durableTestStoredDocument(t, store, "t", "docs", batch[0].ID)
	if !ok {
		t.Fatal("stored document missing after batch insert")
	}
	if got := stored.Vectors["embedding"].Dense; got[0] != 1 {
		t.Fatalf("caller vector mutation leaked into store: %v", got)
	}
	if stored.Metadata["count"] != float64(42) {
		t.Fatalf("caller metadata overwrite leaked into store: %v", stored.Metadata["count"])
	}
	if got := stored.Metadata["tags"].([]string)[0]; got != "keep" {
		t.Fatalf("caller []string mutation leaked into store: %v", got)
	}
	if got := stored.Metadata["deep"].(map[string]interface{})["list"].([]interface{})[0]; got != "original" {
		t.Fatalf("caller nested metadata mutation leaked into store: %v", got)
	}
	// Preserved Go typing is the point of the swap: Vector.Dense is always
	// []float32, so index insertion uses the caller's slice with no unboxing.
	if stored.Vectors["embedding"].Dense == nil {
		t.Fatalf("stored dense vector missing: %+v", stored.Vectors["embedding"])
	}

	upsertVector := []float32{5, 6, 7, 8}
	upsertMetadata := map[string]interface{}{"count": float64(7)}
	upserted := Document{
		ID:       50,
		Vectors:  map[string]Vector{"embedding": {Dense: upsertVector}},
		Metadata: upsertMetadata,
	}
	if err := tenants.UpsertDocument(ctx, "t", "docs", &upserted); err != nil {
		t.Fatal(err)
	}
	upsertVector[1] = -999
	storedUpserted, ok := durableTestStoredDocument(t, store, "t", "docs", 50)
	if !ok {
		t.Fatal("upserted document missing")
	}
	if got := storedUpserted.Vectors["embedding"].Dense; got[1] != 6 {
		t.Fatalf("caller vector mutation leaked into upsert: %v", got)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	reopened, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	reloaded, ok := durableTestStoredDocument(t, reopened, "t", "docs", batch[0].ID)
	if !ok {
		t.Fatal("document lost across checkpoint reload")
	}
	vector := reloaded.Vectors["embedding"].Dense
	if vector == nil {
		t.Fatalf("reloaded dense vector missing: %+v", reloaded.Vectors["embedding"])
	}
	if vector[0] != 1 {
		t.Fatalf("reloaded vector value drifted: %v", vector)
	}
	if got := reloaded.Metadata["tags"].([]interface{})[0]; got != "keep" {
		t.Fatalf("reloaded metadata value drifted: %v", got)
	}
}
