package index

import (
	"context"
	"testing"
)

func TestBinaryWrapperDeleteDoesNotDropRescoreCache(t *testing.T) {
	idx, err := NewBinaryWrapper(4, map[string]interface{}{"keep_full_vectors": true})
	if err != nil {
		t.Fatalf("create wrapper: %v", err)
	}
	wrapper := idx.(*BinaryWrapper)

	if err := wrapper.Add(context.Background(), 1, []float32{1, 0, 0, 0}); err != nil {
		t.Fatalf("seed vector: %v", err)
	}
	if _, ok := wrapper.vectors[1]; !ok {
		t.Fatal("expected rescoring cache to contain seeded vector")
	}

	if err := wrapper.Delete(context.Background(), 1); err == nil {
		t.Fatal("expected delete to remain unsupported")
	}
	if _, ok := wrapper.vectors[1]; !ok {
		t.Fatal("unsupported delete should not mutate rescoring cache")
	}
}

func TestIVFBinaryWrapperDeleteDoesNotDropRescoreCache(t *testing.T) {
	idx, err := NewIVFBinaryWrapper(4, map[string]interface{}{
		"nlist":             1,
		"nprobe":            1,
		"keep_full_vectors": true,
	})
	if err != nil {
		t.Fatalf("create wrapper: %v", err)
	}
	wrapper := idx.(*IVFBinaryWrapper)

	samples := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
	if err := wrapper.Train(samples); err != nil {
		t.Fatalf("train wrapper: %v", err)
	}
	if err := wrapper.Add(context.Background(), 1, []float32{1, 0, 0, 0}); err != nil {
		t.Fatalf("seed vector: %v", err)
	}
	if _, ok := wrapper.vectors[1]; !ok {
		t.Fatal("expected rescoring cache to contain seeded vector")
	}

	if err := wrapper.Delete(context.Background(), 1); err == nil {
		t.Fatal("expected delete to remain unsupported")
	}
	if _, ok := wrapper.vectors[1]; !ok {
		t.Fatal("unsupported delete should not mutate rescoring cache")
	}
}
