package collection

import (
	"context"
	"path/filepath"
	"testing"
)

// BenchmarkDurableInsertSingle measures the acknowledged single-document
// mutation path: one journal append (write + fsync) and one in-memory apply
// per document. This is the ingest floor for callers that cannot batch.
func BenchmarkDurableInsertSingle(b *testing.B) {
	ctx := context.Background()
	base := filepath.Join(b.TempDir(), "collections")
	store, err := OpenDurableStore(base, base)
	if err != nil {
		b.Fatal(err)
	}
	abandonDurableStoreForTestB(b, store)

	tenants := store.Tenants()
	if _, err := tenants.CreateCollection(ctx, "bench", durableTestSchema("docs")); err != nil {
		b.Fatal(err)
	}

	doc := durableTestDocument(0.5)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		d := doc
		if err := tenants.AddDocument(ctx, "bench", "docs", &d); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkDurableInsertBatch measures the acknowledged batch path: one journal
// append per batch of 100 documents.
func BenchmarkDurableInsertBatch100(b *testing.B) {
	ctx := context.Background()
	base := filepath.Join(b.TempDir(), "collections")
	store, err := OpenDurableStore(base, base)
	if err != nil {
		b.Fatal(err)
	}
	abandonDurableStoreForTestB(b, store)

	tenants := store.Tenants()
	if _, err := tenants.CreateCollection(ctx, "bench", durableTestSchema("docs")); err != nil {
		b.Fatal(err)
	}
	makeBatch := func() []Document {
		batch := make([]Document, 100)
		for i := range batch {
			batch[i] = durableTestDocument(float32(i%97) / 97)
		}
		return batch
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		batch := makeBatch()
		b.StartTimer()
		if err := tenants.BatchAddDocuments(ctx, "bench", "docs", batch); err != nil {
			b.Fatal(err)
		}
	}
}

func abandonDurableStoreForTestB(b *testing.B, store *DurableStore) {
	b.Cleanup(func() {
		if err := store.Abort(); err != nil {
			b.Fatalf("abort durable store: %v", err)
		}
	})
}
