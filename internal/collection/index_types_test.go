package collection

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
)

// indexTypeFixture returns a one-field schema and two documents for it, shaped
// for whichever vector kind the index type serves.
func indexTypeFixture(t *testing.T, indexType IndexType) (CollectionSchema, []Document, interface{}) {
	t.Helper()
	kind, ok := indexType.Vectors()
	if !ok {
		t.Fatalf("IndexTypes contains %s, but Vectors does not classify it", indexType)
	}
	schema := CollectionSchema{
		Name:   "docs",
		Fields: []VectorField{{Name: "field", Type: kind, Dim: 8, Index: IndexConfig{Type: indexType}}},
	}
	switch kind {
	case VectorTypeDense:
		return schema, []Document{
			{Vectors: map[string]interface{}{"field": []float32{1, 0, 0, 0, 0, 0, 0, 0}}, Metadata: map[string]interface{}{"n": 1}},
			{Vectors: map[string]interface{}{"field": []float32{0, 1, 0, 0, 0, 0, 0, 0}}, Metadata: map[string]interface{}{"n": 2}},
		}, []float32{1, 0, 0, 0, 0, 0, 0, 0}
	case VectorTypeSparse:
		return schema, []Document{
			{Vectors: map[string]interface{}{"field": map[string]interface{}{"indices": []uint32{1}, "values": []float32{2}, "dim": 8}}, Metadata: map[string]interface{}{"n": 1}},
			{Vectors: map[string]interface{}{"field": map[string]interface{}{"indices": []uint32{5}, "values": []float32{2}, "dim": 8}}, Metadata: map[string]interface{}{"n": 2}},
		}, map[string]interface{}{"indices": []uint32{1}, "values": []float32{2}, "dim": 8}
	default:
		t.Fatalf("no fixture for vector kind %s", kind)
		return CollectionSchema{}, nil, nil
	}
}

// Every member of IndexTypes must survive a full durability cycle: created and
// journaled, checkpointed into the snapshot, then replayed from the journal
// tail on reopen and searched. An index type that parses but cannot be built
// or replayed corrupts a collection on restart — the schema is accepted and
// acknowledged, and the collection comes back missing the index that answers
// its searches. This test is what makes IndexTypes a real list rather than a
// wish: add a member without a constructor case and it fails here.
func TestIndexTypesRoundTripThroughTheJournalAndSnapshot(t *testing.T) {
	for _, indexType := range IndexTypes {
		t.Run(indexType.String(), func(t *testing.T) {
			ctx := context.Background()
			schema, docs, query := indexTypeFixture(t, indexType)
			base := filepath.Join(t.TempDir(), "collections")
			store, err := OpenDurableStore(base, base)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := store.Tenants().CreateCollection(ctx, "t", schema); err != nil {
				t.Fatalf("create collection with index %s: %v", indexType, err)
			}
			// The first document is checkpointed into the snapshot, the
			// second is left in the journal tail, so the reopen below has to
			// rebuild the index from both sources.
			if err := store.Tenants().AddDocument(ctx, "t", schema.Name, &docs[0]); err != nil {
				t.Fatalf("insert into %s collection: %v", indexType, err)
			}
			if err := store.Checkpoint(); err != nil {
				t.Fatalf("checkpoint %s collection: %v", indexType, err)
			}
			if err := store.Tenants().AddDocument(ctx, "t", schema.Name, &docs[1]); err != nil {
				t.Fatalf("post-checkpoint insert into %s collection: %v", indexType, err)
			}
			abandonDurableStoreForTest(t, store)

			reopened, err := OpenDurableStore(base, base)
			if err != nil {
				t.Fatalf("reopen store holding a %s index: %v", indexType, err)
			}
			defer reopened.Close()
			if info := durableTestCollectionInfo(t, reopened, "t", schema.Name); info.DocCount != 2 {
				t.Fatalf("%s collection after reopen: doc_count=%d, want 2", indexType, info.DocCount)
			}
			response, err := reopened.Tenants().SearchCollection(ctx, "t", SearchRequest{
				CollectionName: schema.Name,
				Queries:        map[string]interface{}{"field": query},
				TopK:           1,
			})
			if err != nil {
				t.Fatalf("search rebuilt %s index: %v", indexType, err)
			}
			if len(response.Documents) != 1 || response.Documents[0].ID != docs[0].ID {
				t.Fatalf("search rebuilt %s index returned %+v, want document %d", indexType, response.Documents, docs[0].ID)
			}
		})
	}
}

// Names outside IndexTypes must be refused with the sentinel the transports
// already map to invalid_argument / InvalidArgument, not with an untyped
// error that reaches the caller as a 500. The retired types are the sharp
// case: they still have journaled ordinals, so they must fail by name rather
// than be reinterpreted as whichever type now holds that ordinal.
func TestIndexTypesRejectNamesOutsideTheVocabulary(t *testing.T) {
	for _, name := range []string{"ivf", "diskann", "pq", "", "HNSW"} {
		t.Run("parse/"+name, func(t *testing.T) {
			got, err := ParseIndexType(name)
			if err == nil {
				t.Fatalf("ParseIndexType(%q) = %s, want an error", name, got)
			}
			if !errors.Is(err, ErrInvalidArgument) {
				t.Fatalf("ParseIndexType(%q) error = %v, want ErrInvalidArgument", name, err)
			}
		})
	}

	// The ordinals stay reserved, so a schema can still carry a retired type
	// even though no wire name parses to one; the validator is the backstop.
	ctx := context.Background()
	base := filepath.Join(t.TempDir(), "collections")
	store, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	for _, tc := range []struct {
		name      string
		vector    VectorType
		indexType IndexType
	}{
		{"retired-ivf", VectorTypeDense, IndexTypeIVF},
		{"retired-diskann", VectorTypeDense, IndexTypeDiskANN},
		{"unallocated-ordinal", VectorTypeDense, IndexType(99)},
		{"sparse-index-on-dense-field", VectorTypeDense, IndexTypeInverted},
		{"dense-index-on-sparse-field", VectorTypeSparse, IndexTypeHNSW},
	} {
		t.Run("schema/"+tc.name, func(t *testing.T) {
			schema := CollectionSchema{
				Name:   tc.name,
				Fields: []VectorField{{Name: "field", Type: tc.vector, Dim: 8, Index: IndexConfig{Type: tc.indexType}}},
			}
			_, err := store.Tenants().CreateCollection(ctx, "t", schema)
			if err == nil {
				t.Fatalf("create with index %s on a %s field succeeded", tc.indexType, tc.vector)
			}
			if !errors.Is(err, ErrInvalidArgument) {
				t.Fatalf("create error = %v, want ErrInvalidArgument", err)
			}
		})
	}
	// A refused schema is a rejection, not a fault: the store stays writable.
	if err := store.Err(); err != nil {
		t.Fatalf("schema rejection faulted the store: %v", err)
	}
}
