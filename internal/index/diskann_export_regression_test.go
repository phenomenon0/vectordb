package index

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/phenomenon0/vectordb/internal/filter"
)

func TestDiskANNExportImportSelfContainedSearchAndMetadata(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	sourcePath := filepath.Join(dir, "source.idx")
	restoredPath := filepath.Join(dir, "restored.idx")

	idx, err := NewDiskANNIndex(4, map[string]interface{}{
		"memory_limit":    1,
		"max_degree":      8,
		"ef_construction": 16,
		"ef_search":       16,
		"index_path":      sourcePath,
		"metric":          "euclidean",
	})
	if err != nil {
		t.Fatalf("create source index: %v", err)
	}
	source := idx.(*DiskANNIndex)

	vectors := map[uint64][]float32{
		1: {0, 0, 0, 0},
		2: {1, 0, 0, 0},
		3: {10, 10, 10, 10},
		4: {20, 20, 20, 20},
	}
	categories := map[uint64]string{
		1: "alpha",
		2: "alpha",
		3: "beta",
		4: "beta",
	}

	for id, vec := range vectors {
		if err := idx.Add(context.Background(), id, vec); err != nil {
			t.Fatalf("add vector %d: %v", id, err)
		}
		if err := source.SetMetadata(id, map[string]interface{}{"category": categories[id]}); err != nil {
			t.Fatalf("set metadata for %d: %v", id, err)
		}
	}

	exported, err := idx.Export()
	if err != nil {
		t.Fatalf("export: %v", err)
	}

	if err := source.Close(); err != nil {
		t.Fatalf("close source index: %v", err)
	}
	if err := os.Remove(sourcePath); err != nil && !os.IsNotExist(err) {
		t.Fatalf("remove source index file: %v", err)
	}

	restoredIdx, err := NewDiskANNIndex(4, map[string]interface{}{
		"memory_limit":    1,
		"max_degree":      8,
		"ef_construction": 16,
		"ef_search":       16,
		"index_path":      restoredPath,
		"metric":          "euclidean",
	})
	if err != nil {
		t.Fatalf("create restored index: %v", err)
	}
	restored := restoredIdx.(*DiskANNIndex)
	defer restored.Close()

	if err := restoredIdx.Import(exported); err != nil {
		t.Fatalf("import: %v", err)
	}

	if restored.indexPath != restoredPath {
		t.Fatalf("import rewrote index path: got %q want %q", restored.indexPath, restoredPath)
	}
	if len(restored.unquantizedOffsetIndex) == 0 {
		t.Fatalf("expected imported disk vectors to rebuild offset index")
	}

	query := []float32{10, 10, 10, 10}
	results, err := restoredIdx.Search(context.Background(), query, 4, HNSWSearchParams{})
	if err != nil {
		t.Fatalf("search after import: %v", err)
	}

	foundID3 := false
	for _, result := range results {
		if result.ID == 3 {
			foundID3 = true
			break
		}
	}
	if !foundID3 {
		t.Fatalf("expected imported index to find disk-backed vector 3, got %+v", results)
	}

	filtered, err := restoredIdx.Search(context.Background(), query, 4, HNSWSearchParams{
		Filter: &filter.ComparisonFilter{
			Field:    "category",
			Operator: filter.OpEqual,
			Value:    "beta",
		},
	})
	if err != nil {
		t.Fatalf("filtered search after import: %v", err)
	}
	if len(filtered) == 0 {
		t.Fatalf("expected filtered results after import")
	}
	for _, result := range filtered {
		if result.Metadata["category"] != "beta" {
			t.Fatalf("filtered result lost metadata: %+v", result.Metadata)
		}
	}
}

func TestDiskANNExportImportFloat16PreservesQuantizedVectors(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	sourcePath := filepath.Join(dir, "float16-source.idx")
	restoredPath := filepath.Join(dir, "float16-restored.idx")

	idx, err := NewDiskANNIndex(8, map[string]interface{}{
		"memory_limit":    1,
		"max_degree":      8,
		"ef_construction": 16,
		"ef_search":       16,
		"index_path":      sourcePath,
		"metric":          "euclidean",
		"quantization": map[string]interface{}{
			"type": "float16",
		},
	})
	if err != nil {
		t.Fatalf("create source quantized index: %v", err)
	}
	source := idx.(*DiskANNIndex)

	targetID := uint64(5)
	targetVec := []float32{5, 5, 5, 5, 5, 5, 5, 5}
	for id := uint64(1); id <= targetID; id++ {
		vec := []float32{float32(id), float32(id), float32(id), float32(id), float32(id), float32(id), float32(id), float32(id)}
		if id == targetID {
			vec = append([]float32(nil), targetVec...)
		}
		if err := idx.Add(context.Background(), id, vec); err != nil {
			t.Fatalf("add vector %d: %v", id, err)
		}
	}

	exported, err := idx.Export()
	if err != nil {
		t.Fatalf("export quantized index: %v", err)
	}
	if err := source.Close(); err != nil {
		t.Fatalf("close source quantized index: %v", err)
	}
	if err := os.Remove(sourcePath); err != nil && !os.IsNotExist(err) {
		t.Fatalf("remove source quantized file: %v", err)
	}

	restoredIdx, err := NewDiskANNIndex(8, map[string]interface{}{
		"memory_limit":    1,
		"max_degree":      8,
		"ef_construction": 16,
		"ef_search":       16,
		"index_path":      restoredPath,
		"metric":          "euclidean",
	})
	if err != nil {
		t.Fatalf("create restored quantized index: %v", err)
	}
	restored := restoredIdx.(*DiskANNIndex)
	defer restored.Close()

	if err := restoredIdx.Import(exported); err != nil {
		t.Fatalf("import quantized index: %v", err)
	}
	if restored.quantizer == nil {
		t.Fatalf("expected quantizer to be restored")
	}

	results, err := restoredIdx.Search(context.Background(), targetVec, 3, HNSWSearchParams{})
	if err != nil {
		t.Fatalf("search restored quantized index: %v", err)
	}
	if len(results) == 0 {
		t.Fatalf("expected results from restored quantized index")
	}

	foundTarget := false
	for _, result := range results {
		if result.ID == targetID {
			foundTarget = true
			break
		}
	}
	if !foundTarget {
		t.Fatalf("expected restored quantized index to find target id %d, got %+v", targetID, results)
	}
}
