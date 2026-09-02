package collection

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func ephemeralTestSchema(name string) CollectionSchema {
	schema := durableTestSchema(name)
	schema.Durability = DurabilityEphemeral
	return schema
}

func ephemeralJournalBytes(t *testing.T, store *DurableStore) int64 {
	t.Helper()
	var total int64
	for _, path := range []string{store.journal.currentPath, store.journal.frozenPath} {
		info, err := os.Stat(path)
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			t.Fatal(err)
		}
		total += info.Size()
	}
	return total
}

// ephemeralStoreWithBothClasses creates one durable and one ephemeral
// collection in the same store and returns their common tenant.
func ephemeralStoreWithBothClasses(t *testing.T, store *DurableStore) string {
	t.Helper()
	ctx := context.Background()
	tenants := store.Tenants()
	if _, err := tenants.CreateCollection(ctx, "tenant-a", durableTestSchema("kept")); err != nil {
		t.Fatal(err)
	}
	if _, err := tenants.CreateCollection(ctx, "tenant-a", ephemeralTestSchema("frames")); err != nil {
		t.Fatal(err)
	}
	return "tenant-a"
}

// The class exists so a stream that inserts and evicts documents all day costs
// no journal: an ephemeral document mutation must leave both the applied LSN
// and the journal bytes exactly where the last durable mutation left them.
func TestEphemeralDocumentMutationsWriteNoJournalRecord(t *testing.T) {
	ctx := context.Background()
	base := filepath.Join(t.TempDir(), "collections")
	store, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	tenant := ephemeralStoreWithBothClasses(t, store)
	tenants := store.Tenants()

	durableDoc := durableTestDocument(1)
	if err := tenants.AddDocument(ctx, tenant, "kept", &durableDoc); err != nil {
		t.Fatal(err)
	}
	lsnAfterDurable := store.Metadata().AppliedLSN
	bytesAfterDurable := ephemeralJournalBytes(t, store)

	// All four document mutations, so none of them can be the one that leaks.
	frame := durableTestDocument(2)
	if err := tenants.AddDocument(ctx, tenant, "frames", &frame); err != nil {
		t.Fatal(err)
	}
	if err := tenants.BatchAddDocuments(ctx, tenant, "frames", []Document{durableTestDocument(3), durableTestDocument(4)}); err != nil {
		t.Fatal(err)
	}
	upsert := durableTestDocument(5)
	upsert.ID = 99
	if err := tenants.UpsertDocument(ctx, tenant, "frames", &upsert); err != nil {
		t.Fatal(err)
	}
	if err := tenants.DeleteDocument(ctx, tenant, "frames", frame.ID); err != nil {
		t.Fatal(err)
	}

	if got := store.Metadata().AppliedLSN; got != lsnAfterDurable {
		t.Fatalf("applied LSN = %d after ephemeral mutations, want %d", got, lsnAfterDurable)
	}
	if got := ephemeralJournalBytes(t, store); got != bytesAfterDurable {
		t.Fatalf("journal grew to %d bytes on ephemeral mutations, want %d", got, bytesAfterDurable)
	}
	if got := durableTestCollectionInfo(t, store, tenant, "frames").DocCount; got != 3 {
		t.Fatalf("ephemeral collection holds %d documents in memory, want 3", got)
	}

	// A durable insert still journals, so the bypass is scoped to the class.
	second := durableTestDocument(6)
	if err := tenants.AddDocument(ctx, tenant, "kept", &second); err != nil {
		t.Fatal(err)
	}
	if got := store.Metadata().AppliedLSN; got != lsnAfterDurable+1 {
		t.Fatalf("applied LSN = %d after a durable insert, want %d", got, lsnAfterDurable+1)
	}
	if ephemeralJournalBytes(t, store) <= bytesAfterDurable {
		t.Fatal("durable insert did not grow the journal")
	}
}

// The collection's existence is still class A: create and delete-collection are
// journaled, so the schema comes back and its upstream refills it.
func TestEphemeralCollectionIsEmptyAfterCloseAndReopen(t *testing.T) {
	ctx := context.Background()
	base := filepath.Join(t.TempDir(), "collections")
	store, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	tenant := ephemeralStoreWithBothClasses(t, store)
	tenants := store.Tenants()
	kept := durableTestDocument(1)
	if err := tenants.AddDocument(ctx, tenant, "kept", &kept); err != nil {
		t.Fatal(err)
	}
	if err := tenants.BatchAddDocuments(ctx, tenant, "frames", []Document{durableTestDocument(2), durableTestDocument(3)}); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	reopened, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatalf("reopen after close: %v", err)
	}
	defer reopened.Close()
	frames := durableTestCollectionInfo(t, reopened, tenant, "frames")
	if frames.DocCount != 0 {
		t.Fatalf("ephemeral collection reloaded %d documents, want 0", frames.DocCount)
	}
	if frames.Durability != DurabilityEphemeral {
		t.Fatalf("reloaded durability = %q, want %q", frames.Durability, DurabilityEphemeral)
	}
	keptInfo := durableTestCollectionInfo(t, reopened, tenant, "kept")
	if keptInfo.DocCount != 1 {
		t.Fatalf("durable collection reloaded %d documents, want 1", keptInfo.DocCount)
	}
	if keptInfo.Durability != DurabilityDurable {
		t.Fatalf("durable collection reports %q, want %q", keptInfo.Durability, DurabilityDurable)
	}
}

// A checkpoint is the other way documents reach the snapshot: it must write the
// ephemeral collection's schema and none of its documents.
func TestEphemeralCollectionIsEmptyAfterCheckpointAndReopen(t *testing.T) {
	ctx := context.Background()
	base := filepath.Join(t.TempDir(), "collections")
	store, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	tenant := ephemeralStoreWithBothClasses(t, store)
	tenants := store.Tenants()
	kept := durableTestDocument(1)
	if err := tenants.AddDocument(ctx, tenant, "kept", &kept); err != nil {
		t.Fatal(err)
	}
	if err := tenants.BatchAddDocuments(ctx, tenant, "frames", []Document{durableTestDocument(2), durableTestDocument(3)}); err != nil {
		t.Fatal(err)
	}
	if err := store.Checkpoint(); err != nil {
		t.Fatal(err)
	}
	// Abort, not Close: the reopen must read the checkpoint that Checkpoint
	// wrote, not a second one written on the way out.
	abandonDurableStoreForTest(t, store)

	reopened, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatalf("reopen after checkpoint: %v", err)
	}
	defer reopened.Close()
	if got := durableTestCollectionInfo(t, reopened, tenant, "frames").DocCount; got != 0 {
		t.Fatalf("ephemeral collection reloaded %d documents after checkpoint, want 0", got)
	}
	if got := durableTestCollectionInfo(t, reopened, tenant, "kept").DocCount; got != 1 {
		t.Fatalf("durable collection reloaded %d documents after checkpoint, want 1", got)
	}
}

// Durability is a closed vocabulary because a typo must not silently promise
// persistence the store will not deliver.
func TestEphemeralUnknownDurabilityValueIsRejected(t *testing.T) {
	ctx := context.Background()
	base := filepath.Join(t.TempDir(), "collections")
	store, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	schema := durableTestSchema("frames")
	schema.Durability = "temporary"
	_, err = store.Tenants().CreateCollection(ctx, "tenant-a", schema)
	if !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("create with durability %q returned %v, want ErrInvalidArgument", schema.Durability, err)
	}
}
