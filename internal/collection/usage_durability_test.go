package collection

import (
	"bytes"
	"context"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// The usage sidecar is durability class B: it carries a ranking hint, so
// losing it must cost ranking quality and nothing else. These tests pin
// both halves of that bargain — the hint survives a restart, and every way
// the sidecar can be unusable leaves the collection answering searches with
// no fault latched.

const usageDurabilityTenant = "tenant-a"

// usageProbeQuery matches the exact document. usageNearVector is ten
// degrees off it: close enough that a usage nudge can reorder the two,
// far enough that similarity alone never does.
var (
	usageProbeQuery = []float32{1, 0, 0, 0}
	usageNearVector = []float32{
		float32(math.Cos(10 * math.Pi / 180)),
		float32(math.Sin(10 * math.Pi / 180)),
		0, 0,
	}
)

func openUsageDurabilityStore(t *testing.T, base string) *DurableStore {
	t.Helper()
	store, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatalf("open durable store: %v", err)
	}
	return store
}

// seedUsageDurabilityCollection creates the two-document collection and
// returns the exact-match and near-match document IDs.
func seedUsageDurabilityCollection(t *testing.T, store *DurableStore) (exactID, nearID uint64) {
	t.Helper()
	ctx := context.Background()
	if _, err := store.Tenants().CreateCollection(ctx, usageDurabilityTenant, CollectionSchema{
		Name: "docs",
		Fields: []VectorField{{
			Name: "dense", Type: VectorTypeDense, Dim: 4,
			Index: IndexConfig{Type: IndexTypeFLAT},
		}},
	}); err != nil {
		t.Fatalf("create collection: %v", err)
	}
	exact := Document{Vectors: map[string]Vector{"dense": {Dense: usageProbeQuery}}}
	near := Document{Vectors: map[string]Vector{"dense": {Dense: usageNearVector}}}
	for _, doc := range []*Document{&exact, &near} {
		if err := store.Tenants().AddDocument(ctx, usageDurabilityTenant, "docs", doc); err != nil {
			t.Fatalf("add document: %v", err)
		}
	}
	return exact.ID, near.ID
}

// usageDurabilityOrder runs the probe query and returns the ranked IDs.
func usageDurabilityOrder(t *testing.T, store *DurableStore, boost float64) []uint64 {
	t.Helper()
	resp, err := store.Tenants().SearchCollection(context.Background(), usageDurabilityTenant, SearchRequest{
		CollectionName: "docs",
		Queries:        map[string]interface{}{"dense": usageProbeQuery},
		TopK:           2,
		UsageBoost:     boost,
	})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	ids := make([]uint64, 0, len(resp.Documents))
	for _, doc := range resp.Documents {
		ids = append(ids, doc.ID)
	}
	return ids
}

// touchNearDocument builds usage on the near document only, the way an
// agent does: repeated searches that return just that document.
func touchNearDocument(t *testing.T, store *DurableStore, nearID uint64, times int) {
	t.Helper()
	for i := 0; i < times; i++ {
		resp, err := store.Tenants().SearchCollection(context.Background(), usageDurabilityTenant, SearchRequest{
			CollectionName: "docs",
			Queries:        map[string]interface{}{"dense": usageNearVector},
			TopK:           1,
			UsageBoost:     0,
		})
		if err != nil {
			t.Fatalf("touch search: %v", err)
		}
		if len(resp.Documents) != 1 || resp.Documents[0].ID != nearID {
			t.Fatalf("touch search returned %v, want only document %d", resp.Documents, nearID)
		}
	}
}

func TestUsageSidecarSurvivesRestart(t *testing.T) {
	base := filepath.Join(t.TempDir(), "collections")
	store := openUsageDurabilityStore(t, base)
	exactID, nearID := seedUsageDurabilityCollection(t, store)

	if got := usageDurabilityOrder(t, store, 0.9); got[0] != exactID {
		t.Fatalf("cold order = %v, want the exact match %d first (no usage recorded yet)", got, exactID)
	}
	touchNearDocument(t, store, nearID, 10)
	before := usageDurabilityOrder(t, store, 0.9)
	if before[0] != nearID {
		t.Fatalf("order after touches = %v, want the touched document %d first", before, nearID)
	}
	if err := store.Close(); err != nil {
		t.Fatalf("close store: %v", err)
	}

	reopened := openUsageDurabilityStore(t, base)
	defer func() {
		if err := reopened.Close(); err != nil {
			t.Errorf("close reopened store: %v", err)
		}
	}()
	if !reopened.UsageLoaded() {
		t.Fatal("UsageLoaded false after a clean close wrote the sidecar")
	}
	// The point of the sidecar: the same query reorders the same way after
	// a restart. Without it the tracker would be empty and the exact match
	// would lead again, as the cold order above shows.
	after := usageDurabilityOrder(t, reopened, 0.9)
	if len(after) != len(before) || after[0] != before[0] || after[1] != before[1] {
		t.Fatalf("order after restart = %v, want the pre-restart order %v", after, before)
	}
	// A hint, never a takeover: raw similarity ranking is unchanged.
	if raw := usageDurabilityOrder(t, reopened, 0); raw[0] != exactID {
		t.Fatalf("unboosted order = %v, want the exact match %d first", raw, exactID)
	}
}

func TestUsageSidecarCorruptionKeepsCollectionUp(t *testing.T) {
	cases := []struct {
		name     string
		contents string
	}{
		{"garbage bytes", "\x00\x01not json at all"},
		{"unknown version", `{"version":99,"collections":[]}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			base := filepath.Join(t.TempDir(), "collections")
			store := openUsageDurabilityStore(t, base)
			exactID, nearID := seedUsageDurabilityCollection(t, store)
			touchNearDocument(t, store, nearID, 10)
			if err := store.Close(); err != nil {
				t.Fatalf("close store: %v", err)
			}
			if err := os.WriteFile(usageSidecarPath(base), []byte(tc.contents), 0o600); err != nil {
				t.Fatal(err)
			}

			reopened := openUsageDurabilityStore(t, base)
			defer func() {
				if err := reopened.Close(); err != nil {
					t.Errorf("close reopened store: %v", err)
				}
			}()
			if reopened.UsageLoaded() {
				t.Fatal("UsageLoaded true after the sidecar was discarded")
			}
			// Class B, not class A: an unusable ranking hint must not
			// latch the fault that takes /readyz to 503.
			if err := reopened.Err(); err != nil {
				t.Fatalf("store health = %v, want nil (a bad sidecar must not fault the store)", err)
			}
			order := usageDurabilityOrder(t, reopened, 0.9)
			if len(order) != 2 {
				t.Fatalf("search returned %v, want both documents", order)
			}
			// The hint is gone, so similarity alone decides.
			if order[0] != exactID {
				t.Fatalf("order = %v, want the exact match %d first with no usage restored", order, exactID)
			}
		})
	}
}

func TestUsageSidecarAbsentOpensWithEmptyTracker(t *testing.T) {
	base := filepath.Join(t.TempDir(), "collections")
	store := openUsageDurabilityStore(t, base)
	exactID, nearID := seedUsageDurabilityCollection(t, store)
	touchNearDocument(t, store, nearID, 10)
	if err := store.Close(); err != nil {
		t.Fatalf("close store: %v", err)
	}
	// A data directory written before the sidecar existed has no
	// usage.json. That is a normal open, not a corruption.
	if err := os.Remove(usageSidecarPath(base)); err != nil {
		t.Fatal(err)
	}

	reopened := openUsageDurabilityStore(t, base)
	defer func() {
		if err := reopened.Close(); err != nil {
			t.Errorf("close reopened store: %v", err)
		}
	}()
	// UsageLoaded answers "did you lose hints?", not "did you read a file".
	// A store that never had a sidecar lost nothing, so reporting false here
	// would make the status signal accuse every new deployment.
	if !reopened.UsageLoaded() {
		t.Fatal("UsageLoaded false with no sidecar on disk; only a discarded sidecar is a loss")
	}
	if err := reopened.Err(); err != nil {
		t.Fatalf("store health = %v, want nil", err)
	}
	if order := usageDurabilityOrder(t, reopened, 0.9); order[0] != exactID {
		t.Fatalf("order = %v, want the exact match %d first with an empty tracker", order, exactID)
	}
}

func TestUsageSidecarSnapshotCommitIgnoresCrashedTempFile(t *testing.T) {
	dir := t.TempDir()
	base := filepath.Join(dir, "collections")
	store := openUsageDurabilityStore(t, base)
	_, nearID := seedUsageDurabilityCollection(t, store)
	touchNearDocument(t, store, nearID, 10)

	// A snapshot commits the sidecar; Close is not the only writer.
	if err := store.Checkpoint(); err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	committed, err := os.ReadFile(usageSidecarPath(base))
	if err != nil {
		t.Fatalf("read sidecar after checkpoint: %v", err)
	}
	var doc usageSidecarDocument
	if err := decodeCollectionJSON(committed, &doc); err != nil {
		t.Fatalf("decode sidecar: %v", err)
	}
	if doc.Version != usageDocumentVersion || len(doc.Collections) != 1 {
		t.Fatalf("sidecar = %+v, want version %d and one collection", doc, usageDocumentVersion)
	}
	entry := doc.Collections[0]
	if entry.TenantID != usageDurabilityTenant || entry.Collection != "docs" {
		t.Fatalf("sidecar names %s/%s, want %s/docs", entry.TenantID, entry.Collection, usageDurabilityTenant)
	}
	var nearCount uint32
	for _, record := range entry.Entries {
		if record.DocID == nearID {
			nearCount = record.Count
		}
	}
	if nearCount < 10 {
		t.Fatalf("near document count = %d, want at least the 10 touches", nearCount)
	}

	// A crash between the temp write and the rename leaves a stray
	// half-written generation beside the committed one. The next open must
	// read the committed file and never the stray.
	stray := filepath.Join(dir, "."+filepath.Base(usageSidecarPath(base))+".tmp-crashed")
	if err := os.WriteFile(stray, []byte(`{"version":1,"collect`), 0o600); err != nil {
		t.Fatal(err)
	}
	abandonDurableStoreForTest(t, store)

	reopened := openUsageDurabilityStore(t, base)
	defer func() {
		if err := reopened.Close(); err != nil {
			t.Errorf("close reopened store: %v", err)
		}
	}()
	if !reopened.UsageLoaded() {
		t.Fatal("UsageLoaded false, want the committed generation loaded past the stray temp file")
	}
	if order := usageDurabilityOrder(t, reopened, 0.9); order[0] != nearID {
		t.Fatalf("order = %v, want the touched document %d first", order, nearID)
	}
	current, err := os.ReadFile(usageSidecarPath(base))
	if err != nil {
		t.Fatalf("read sidecar after reopen: %v", err)
	}
	if !bytes.Equal(current, committed) {
		t.Fatal("the committed sidecar generation changed; the stray temp file must not be promoted")
	}
	if _, err := os.Stat(stray); err != nil {
		t.Fatalf("stray temp file: %v, want it left untouched for forensics", err)
	}
}

func TestUsageTrackerExportImportRoundTrip(t *testing.T) {
	tracker := NewUsageTracker()
	tracker.Record(7)
	tracker.Record(7)
	tracker.Record(3)

	doc := tracker.Export()
	if doc.Version != usageDocumentVersion {
		t.Fatalf("export version = %d, want %d", doc.Version, usageDocumentVersion)
	}
	// Deterministic output: sorted by document ID, so identical state
	// always produces identical sidecar bytes.
	if len(doc.Entries) != 2 || doc.Entries[0].DocID != 3 || doc.Entries[1].DocID != 7 {
		t.Fatalf("export entries = %+v, want documents 3 then 7", doc.Entries)
	}

	restored := NewUsageTracker()
	if err := restored.Import(doc); err != nil {
		t.Fatalf("import: %v", err)
	}
	// Score is the only thing usage_boost reads, so the round trip is
	// correct exactly when every score matches. Timestamps are exported at
	// millisecond granularity, so scores match to within one millisecond of
	// decay — far below the difference between any two ranked documents.
	for _, id := range []uint64{3, 7} {
		want, got := tracker.Score(id), restored.Score(id)
		if math.Abs(want-got) > 1e-5 {
			t.Fatalf("document %d score = %v, want %v", id, got, want)
		}
	}

	// Documents this build could not have written are refused, and a
	// refused import leaves the tracker as it was.
	bad := []UsageDocument{
		{Version: usageDocumentVersion + 1, Entries: doc.Entries},
		{Version: usageDocumentVersion, Entries: []UsageRecord{{DocID: 0, Count: 1}}},
		{Version: usageDocumentVersion, Entries: []UsageRecord{{DocID: 5, Count: 0}}},
		{Version: usageDocumentVersion, Entries: []UsageRecord{{DocID: 5, Count: 1}, {DocID: 5, Count: 2}}},
	}
	for _, candidate := range bad {
		if err := restored.Import(candidate); err == nil {
			t.Fatalf("import accepted %+v, want an error", candidate)
		}
	}
	if got := restored.Len(); got != 2 {
		t.Fatalf("entries after refused imports = %d, want the 2 already there", got)
	}
}
