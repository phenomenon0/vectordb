package index

import (
	"context"
	"testing"

	"github.com/phenomenon0/vectordb/internal/filter"
)

// TestFLATFilteredSearch verifies that metadata filters are honored by the
// FLAT index. Before the fix, FLAT ignored the Search params filter entirely
// and returned unfiltered results, silently violating the caller's filter.
func TestFLATFilteredSearch(t *testing.T) {
	ctx := context.Background()
	idx, err := NewFLATIndex(3, map[string]interface{}{"metric": "cosine"})
	if err != nil {
		t.Fatalf("create flat: %v", err)
	}
	flatIdx := idx.(*FLATIndex)

	vectors := []struct {
		id  uint64
		vec []float32
		cat string
	}{
		{1, []float32{0.0, 1.0, 0.0}, "electronics"},
		{2, []float32{0.1, 0.99, 0.0}, "electronics"},
		{3, []float32{1.0, 0.0, 0.0}, "clothing"},
	}
	for _, v := range vectors {
		if err := flatIdx.Add(ctx, v.id, v.vec); err != nil {
			t.Fatalf("add %d: %v", v.id, err)
		}
		if err := flatIdx.SetMetadata(v.id, map[string]interface{}{"category": v.cat}); err != nil {
			t.Fatalf("metadata %d: %v", v.id, err)
		}
	}

	t.Run("FilterMatches", func(t *testing.T) {
		results, err := flatIdx.Search(ctx, []float32{0.0, 1.0, 0.0}, 10, HNSWSearchParams{
			Filter: filter.Eq("category", "clothing"),
		})
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		var ids []uint64
		for _, r := range results {
			ids = append(ids, r.ID)
			if r.Metadata["category"] != "clothing" {
				t.Errorf("result %d category = %v, want clothing", r.ID, r.Metadata["category"])
			}
		}
		if len(ids) != 1 || ids[0] != 3 {
			t.Errorf("filter=clothing returned ids %v, want [3]", ids)
		}
	})

	t.Run("FilterDoesNotMatch", func(t *testing.T) {
		results, err := flatIdx.Search(ctx, []float32{0.0, 1.0, 0.0}, 10, HNSWSearchParams{
			Filter: filter.Eq("category", "nonexistent"),
		})
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		if len(results) != 0 {
			t.Errorf("impossible filter returned %d results, want 0", len(results))
		}
	})

	t.Run("UnfilteredStillWorks", func(t *testing.T) {
		results, err := flatIdx.Search(ctx, []float32{0.0, 1.0, 0.0}, 10, DefaultSearchParams{})
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		if len(results) != 3 {
			t.Errorf("unfiltered search returned %d results, want 3", len(results))
		}
	})
}

// TestFLATFilteredSearchPersistence verifies metadata survives the snapshot
// export/import round-trip so filters still apply after restart.
func TestFLATFilteredSearchPersistence(t *testing.T) {
	ctx := context.Background()
	idx, err := NewFLATIndex(2, map[string]interface{}{"metric": "cosine"})
	if err != nil {
		t.Fatalf("create flat: %v", err)
	}
	src := idx.(*FLATIndex)
	if err := src.Add(ctx, 1, []float32{0, 1}); err != nil {
		t.Fatalf("add: %v", err)
	}
	if err := src.SetMetadata(1, map[string]interface{}{"keep": "yes"}); err != nil {
		t.Fatalf("metadata: %v", err)
	}

	data, err := src.Export()
	if err != nil {
		t.Fatalf("export: %v", err)
	}

	idx2, err := NewFLATIndex(2, map[string]interface{}{"metric": "cosine"})
	if err != nil {
		t.Fatalf("create flat2: %v", err)
	}
	dst := idx2.(*FLATIndex)
	if err := dst.Import(data); err != nil {
		t.Fatalf("import: %v", err)
	}

	results, err := dst.Search(ctx, []float32{0, 1}, 10, HNSWSearchParams{
		Filter: filter.Eq("keep", "no"),
	})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("after import, filter=keep:no returned %d results, want 0 (metadata lost in round-trip?)", len(results))
	}

	results, err = dst.Search(ctx, []float32{0, 1}, 10, HNSWSearchParams{
		Filter: filter.Eq("keep", "yes"),
	})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(results) != 1 || results[0].ID != 1 {
		t.Errorf("after import, filter=keep:yes returned ids %v, want [1]", idsOf(results))
	}
}

// SetMetadata on a tombstoned ID and on a missing ID behaves like HNSW.
func TestFLATSetMetadataLifecycle(t *testing.T) {
	ctx := context.Background()
	idx, err := NewFLATIndex(2, map[string]interface{}{"metric": "cosine"})
	if err != nil {
		t.Fatalf("create flat: %v", err)
	}
	flatIdx := idx.(*FLATIndex)

	if err := flatIdx.Add(ctx, 1, []float32{0, 1}); err != nil {
		t.Fatalf("add: %v", err)
	}
	if err := flatIdx.SetMetadata(1, map[string]interface{}{"a": "b"}); err != nil {
		t.Fatalf("set metadata: %v", err)
	}
	// Clearing via nil must not panic nor error.
	if err := flatIdx.SetMetadata(1, nil); err != nil {
		t.Fatalf("clear metadata: %v", err)
	}
	// Missing id errors, mirroring HNSW.
	if err := flatIdx.SetMetadata(404, map[string]interface{}{"a": "b"}); err == nil {
		t.Errorf("SetMetadata on missing id should error")
	}
}

func flatResultIDs(rs []Result) []uint64 {
	out := make([]uint64, 0, len(rs))
	for _, r := range rs {
		out = append(out, r.ID)
	}
	return out
}
