package collection

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"testing"
)

// TestDurableStoreTenantLifecycleSurvivesRestart pins the tenant record
// lifecycle end to end: provisioning, suspension blocking every data-plane
// verb, reactivation, and deletion — each transition surviving a restart and
// a checkpoint+restart, and never latching the store fault.
func TestDurableStoreTenantLifecycleSurvivesRestart(t *testing.T) {
	ctx := context.Background()
	base := filepath.Join(t.TempDir(), "tenant-lifecycle")

	store, err := OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}

	rec := TenantRecord{TenantID: "acme", Status: TenantStatusActive}
	if err := store.Tenants().CreateTenant(ctx, rec); err != nil {
		t.Fatalf("create tenant: %v", err)
	}
	if err := store.Tenants().CreateTenant(ctx, rec); !errors.Is(err, ErrTenantExists) {
		t.Fatalf("duplicate create tenant error = %v, want ErrTenantExists", err)
	}

	if err := store.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	store, err = OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := store.Tenants().ListTenantsChecked()
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 1 || ids[0] != "acme" {
		t.Fatalf("tenants after restart = %v, want [acme]", ids)
	}
	got, ok := store.Tenants().getTenantRecord("acme")
	if !ok || got != rec {
		t.Fatalf("tenant record after restart = %+v, %v, want %+v, true", got, ok, rec)
	}

	if _, err := store.Tenants().CreateCollection(ctx, "acme", durableTestSchema("docs")); err != nil {
		t.Fatalf("create collection before suspension: %v", err)
	}
	doc := durableTestDocument(1)
	if err := store.Tenants().AddDocument(ctx, "acme", "docs", &doc); err != nil {
		t.Fatalf("add document before suspension: %v", err)
	}

	suspended := rec
	suspended.Status = TenantStatusSuspended
	if err := store.Tenants().UpdateTenant(ctx, suspended); err != nil {
		t.Fatalf("suspend tenant: %v", err)
	}

	if _, err := store.Tenants().CreateCollection(ctx, "acme", durableTestSchema("other")); !errors.Is(err, ErrTenantSuspended) {
		t.Fatalf("create collection while suspended error = %v, want ErrTenantSuspended", err)
	}
	blocked := durableTestDocument(2)
	if err := store.Tenants().AddDocument(ctx, "acme", "docs", &blocked); !errors.Is(err, ErrTenantSuspended) {
		t.Fatalf("add document while suspended error = %v, want ErrTenantSuspended", err)
	}
	includeVectors := false
	if _, err := store.Tenants().SearchCollection(ctx, "acme", SearchRequest{
		CollectionName: "docs",
		Queries:        map[string]interface{}{"embedding": []float32{1, 0, 0, 0}},
		TopK:           1,
		IncludeVectors: &includeVectors,
	}); !errors.Is(err, ErrTenantSuspended) {
		t.Fatalf("search while suspended error = %v, want ErrTenantSuspended", err)
	}
	if _, err := store.Tenants().GetDocumentChecked("acme", "docs", doc.ID); !errors.Is(err, ErrTenantSuspended) {
		t.Fatalf("get document while suspended error = %v, want ErrTenantSuspended", err)
	}
	if err := store.Err(); err != nil {
		t.Fatalf("suspension rejections must not fault the store: %v", err)
	}

	reactivated := rec
	if err := store.Tenants().UpdateTenant(ctx, reactivated); err != nil {
		t.Fatalf("reactivate tenant: %v", err)
	}
	if _, err := store.Tenants().CreateCollection(ctx, "acme", durableTestSchema("other")); err != nil {
		t.Fatalf("create collection after reactivation: %v", err)
	}
	doc2 := durableTestDocument(3)
	if err := store.Tenants().AddDocument(ctx, "acme", "docs", &doc2); err != nil {
		t.Fatalf("add document after reactivation: %v", err)
	}

	if err := store.Checkpoint(); err != nil {
		t.Fatalf("checkpoint: %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatalf("close after checkpoint: %v", err)
	}
	store, err = OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	got, ok = store.Tenants().getTenantRecord("acme")
	if !ok || got != rec {
		t.Fatalf("tenant record after checkpoint restart = %+v, %v, want %+v, true", got, ok, rec)
	}
	if _, err := store.Tenants().GetCollectionInfo("acme", "docs"); err != nil {
		t.Fatalf("collection missing after checkpoint restart: %v", err)
	}
	if _, err := store.Tenants().GetCollectionInfo("acme", "other"); err != nil {
		t.Fatalf("second collection missing after checkpoint restart: %v", err)
	}

	if err := store.Tenants().DeleteTenant(ctx, "acme"); err != nil {
		t.Fatalf("delete tenant: %v", err)
	}
	ids, err = store.Tenants().ListTenantsChecked()
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 0 {
		t.Fatalf("tenants after delete = %v, want none", ids)
	}
	if _, err := store.Tenants().GetCollectionInfo("acme", "docs"); err == nil {
		t.Fatal("collection survived tenant delete")
	}
	if err := store.Tenants().DeleteTenant(ctx, "acme"); !errors.Is(err, ErrTenantNotFound) {
		t.Fatalf("second delete tenant error = %v, want ErrTenantNotFound", err)
	}

	if err := store.Abort(); err != nil {
		t.Fatalf("abort: %v", err)
	}

	// A record-only tenant (no collections) still consumes a MaxTenants
	// admission slot, so a second tenant's first collection create is
	// rejected exactly as if the record-only tenant owned collections.
	t.Run("record-only tenant holds a MaxTenants slot", func(t *testing.T) {
		quotaBase := filepath.Join(t.TempDir(), "tenant-quota")
		quotaStore := openLimitsTestStore(t, quotaBase, StoreLimits{MaxTenants: 1, MaxCollections: 10})
		if err := quotaStore.Tenants().CreateTenant(ctx, TenantRecord{TenantID: "solo", Status: TenantStatusActive}); err != nil {
			t.Fatalf("create record-only tenant: %v", err)
		}
		requireLimitsTestCounts(t, quotaStore, 1, 0)
		if _, err := quotaStore.Tenants().CreateCollection(ctx, "other", durableTestSchema("docs")); !errors.Is(err, ErrTenantLimitExceeded) {
			t.Fatalf("second tenant create collection error = %v, want ErrTenantLimitExceeded", err)
		}
	})
}

// TestDocumentBytesIsPure pins documentBytes as a deterministic, allocation-
// free estimate. Quota admission calls it on the hot write path: if it
// allocated or drifted between calls on the same document, the byte cap
// would flake under GC pressure and disagree with itself on a retry.
func TestDocumentBytesIsPure(t *testing.T) {
	doc := durableTestDocument(1)
	doc.ID = 7

	first := documentBytes(&doc)
	if second := documentBytes(&doc); second != first {
		t.Fatalf("documentBytes(doc) = %d then %d, want identical", first, second)
	}

	longer := doc
	longer.Vectors = map[string]Vector{"embedding": {Dense: []float32{1, 2, 3, 4, 5, 6, 7, 8}}}
	if got := documentBytes(&longer); got <= first {
		t.Fatalf("longer dense vector documentBytes = %d, want > %d", got, first)
	}

	nested := doc
	nested.Metadata = map[string]interface{}{
		"tags":   []interface{}{"a", "b", "c"},
		"nested": map[string]interface{}{"key": "value"},
	}
	if got := documentBytes(&nested); got <= first {
		t.Fatalf("nested metadata documentBytes = %d, want > %d", got, first)
	}

	if allocs := testing.AllocsPerRun(100, func() { documentBytes(&doc) }); allocs != 0 {
		t.Fatalf("documentBytes allocs/op = %v, want 0", allocs)
	}
}

// TestDurableStoreTenantQuotaAdmission pins per-tenant quota enforcement end
// to end: document-count and byte admission at the local write entry points,
// atomic batch rejection, a shrink never rejecting, a per-tenant record
// override beating the server default, collection-count admission, usage
// falling back to zero once its documents/collection are gone, and that none
// of this reaches replay -- a restart with limits already below current
// usage must still open cleanly and keep every document.
func TestDurableStoreTenantQuotaAdmission(t *testing.T) {
	ctx := context.Background()
	base := filepath.Join(t.TempDir(), "tenant-quota-admission")
	limits := StoreLimits{
		MaxTenants:           10,
		MaxCollections:       10,
		MaxTenantDocuments:   3,
		MaxTenantBytes:       200,
		MaxTenantCollections: 1,
	}
	store, err := OpenDurableStoreWithLimits(base, base, limits)
	if err != nil {
		t.Fatal(err)
	}

	if err := store.Tenants().CreateTenant(ctx, TenantRecord{TenantID: "acme", Status: TenantStatusActive}); err != nil {
		t.Fatalf("create tenant: %v", err)
	}
	if _, err := store.Tenants().CreateCollection(ctx, "acme", durableTestSchema("docs")); err != nil {
		t.Fatalf("create collection: %v", err)
	}

	// Three inserts fill the MaxTenantDocuments=3 cap exactly; the fourth is
	// rejected without faulting the store.
	docs := make([]Document, 3)
	for i := range docs {
		docs[i] = durableTestDocument(float32(i))
		if err := store.Tenants().AddDocument(ctx, "acme", "docs", &docs[i]); err != nil {
			t.Fatalf("add document %d: %v", i, err)
		}
	}
	fourth := durableTestDocument(3)
	if err := store.Tenants().AddDocument(ctx, "acme", "docs", &fourth); !errors.Is(err, ErrTenantQuotaExceeded) {
		t.Fatalf("fourth add error = %v, want ErrTenantQuotaExceeded", err)
	}
	if err := store.Err(); err != nil {
		t.Fatalf("quota rejection must not fault the store: %v", err)
	}
	info, err := store.Tenants().GetTenantInfo("acme")
	if err != nil {
		t.Fatal(err)
	}
	if info.Usage.Documents != 3 {
		t.Fatalf("usage after rejected fourth add = %d, want 3", info.Usage.Documents)
	}

	// Delete one document, freeing exactly one slot (usage now 2), then a
	// batch of two must be rejected as a whole: 2+2 > 3.
	if err := store.Tenants().DeleteDocument(ctx, "acme", "docs", docs[0].ID); err != nil {
		t.Fatalf("delete document: %v", err)
	}
	if info, err = store.Tenants().GetTenantInfo("acme"); err != nil {
		t.Fatal(err)
	} else if info.Usage.Documents != 2 {
		t.Fatalf("usage after delete = %d, want 2", info.Usage.Documents)
	}
	batch := []Document{durableTestDocument(10), durableTestDocument(11)}
	if err := store.Tenants().BatchAddDocuments(ctx, "acme", "docs", batch); !errors.Is(err, ErrTenantQuotaExceeded) {
		t.Fatalf("batch add error = %v, want ErrTenantQuotaExceeded", err)
	}
	if info, err = store.Tenants().GetTenantInfo("acme"); err != nil {
		t.Fatal(err)
	} else if info.Usage.Documents != 2 {
		t.Fatalf("usage after rejected batch = %d, want 2 (atomic reject)", info.Usage.Documents)
	}

	// An upsert that grows an existing document past MaxTenantBytes is
	// rejected; the document count is unchanged so bytes is the only
	// dimension that can be the cause.
	grown := docs[1]
	grown.Metadata = map[string]interface{}{"note": strings.Repeat("x", 500)}
	if err := store.Tenants().UpsertDocument(ctx, "acme", "docs", &grown); !errors.Is(err, ErrTenantQuotaExceeded) {
		t.Fatalf("growing upsert error = %v, want ErrTenantQuotaExceeded", err)
	}
	beforeShrink, err := store.Tenants().GetTenantInfo("acme")
	if err != nil {
		t.Fatal(err)
	}

	// The same document shrinking is accepted: a negative byte delta never
	// trips the cap.
	shrunk := docs[1]
	shrunk.Metadata = nil
	if err := store.Tenants().UpsertDocument(ctx, "acme", "docs", &shrunk); err != nil {
		t.Fatalf("shrinking upsert: %v", err)
	}
	afterShrink, err := store.Tenants().GetTenantInfo("acme")
	if err != nil {
		t.Fatal(err)
	}
	wantShrinkDelta := documentBytes(&shrunk) - documentBytes(&docs[1])
	if afterShrink.Usage.Bytes != beforeShrink.Usage.Bytes+wantShrinkDelta {
		t.Fatalf("bytes after shrink = %d, want %d", afterShrink.Usage.Bytes, beforeShrink.Usage.Bytes+wantShrinkDelta)
	}
	if afterShrink.Usage.Documents != beforeShrink.Usage.Documents {
		t.Fatalf("documents changed by an in-place upsert: got %d, want %d", afterShrink.Usage.Documents, beforeShrink.Usage.Documents)
	}

	// Deleting a document lowers both Documents and Bytes by exactly its own
	// footprint.
	before, err := store.Tenants().GetTenantInfo("acme")
	if err != nil {
		t.Fatal(err)
	}
	shrunkBytes := documentBytes(&shrunk)
	if err := store.Tenants().DeleteDocument(ctx, "acme", "docs", shrunk.ID); err != nil {
		t.Fatalf("delete document: %v", err)
	}
	after, err := store.Tenants().GetTenantInfo("acme")
	if err != nil {
		t.Fatal(err)
	}
	if after.Usage.Documents != before.Usage.Documents-1 || after.Usage.Bytes != before.Usage.Bytes-shrunkBytes {
		t.Fatalf("usage after document delete = %+v, want documents %d bytes %d",
			after.Usage, before.Usage.Documents-1, before.Usage.Bytes-shrunkBytes)
	}

	// Deleting the whole collection zeroes both counters.
	if err := store.Tenants().DeleteCollection(ctx, "acme", "docs"); err != nil {
		t.Fatalf("delete collection: %v", err)
	}
	if after, err = store.Tenants().GetTenantInfo("acme"); err != nil {
		t.Fatal(err)
	} else if after.Usage.Documents != 0 || after.Usage.Bytes != 0 {
		t.Fatalf("usage after collection delete = %+v, want zero", after.Usage)
	}

	// MaxTenantCollections=1: acme's first collection since the delete is
	// admitted, a second is rejected.
	if _, err := store.Tenants().CreateCollection(ctx, "acme", durableTestSchema("docs2")); err != nil {
		t.Fatalf("create replacement collection: %v", err)
	}
	if _, err := store.Tenants().CreateCollection(ctx, "acme", durableTestSchema("docs3")); !errors.Is(err, ErrTenantQuotaExceeded) {
		t.Fatalf("second collection error = %v, want ErrTenantQuotaExceeded", err)
	}

	// A tenant record's own Quota override beats the server default: acme's
	// server-wide MaxTenantDocuments is 3, but this record raises it to 10.
	override := TenantRecord{TenantID: "override", Status: TenantStatusActive, Quota: TenantQuota{MaxDocuments: 10, MaxBytes: 10_000}}
	if err := store.Tenants().CreateTenant(ctx, override); err != nil {
		t.Fatalf("create override tenant: %v", err)
	}
	if _, err := store.Tenants().CreateCollection(ctx, "override", durableTestSchema("docs")); err != nil {
		t.Fatalf("create override collection: %v", err)
	}
	for i := 0; i < 5; i++ {
		d := durableTestDocument(float32(i))
		if err := store.Tenants().AddDocument(ctx, "override", "docs", &d); err != nil {
			t.Fatalf("override tenant add document %d: %v", i, err)
		}
	}
	overrideInfo, err := store.Tenants().GetTenantInfo("override")
	if err != nil {
		t.Fatal(err)
	}
	if overrideInfo.Usage.Documents != 5 {
		t.Fatalf("override tenant documents = %d, want 5 (beyond server default 3)", overrideInfo.Usage.Documents)
	}
	if overrideInfo.Quota.MaxDocuments != 10 {
		t.Fatalf("override tenant effective quota = %+v, want MaxDocuments 10", overrideInfo.Quota)
	}

	// Close + reopen: usage is derived fresh from the replayed collections
	// rather than persisted directly, so it must land on exactly the same
	// numbers.
	preClose, err := store.Tenants().ListTenantInfos()
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	reopened, err := OpenDurableStoreWithLimits(base, base, limits)
	if err != nil {
		t.Fatal(err)
	}
	postOpen, err := reopened.Tenants().ListTenantInfos()
	if err != nil {
		t.Fatal(err)
	}
	if len(preClose) != len(postOpen) {
		t.Fatalf("tenant info count after reopen = %d, want %d", len(postOpen), len(preClose))
	}
	for i := range preClose {
		if preClose[i] != postOpen[i] {
			t.Fatalf("tenant info[%d] after reopen = %+v, want %+v", i, postOpen[i], preClose[i])
		}
	}
	if err := reopened.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	// Replay never rejects: reopening with limits already below current
	// usage must still open cleanly and keep every document.
	tinyLimits := StoreLimits{MaxTenants: 10, MaxCollections: 10, MaxTenantDocuments: 1, MaxTenantBytes: 1, MaxTenantCollections: 1}
	tinyStore, err := OpenDurableStoreWithLimits(base, base, tinyLimits)
	if err != nil {
		t.Fatalf("reopen with tighter limits: %v", err)
	}
	tinyInfo, err := tinyStore.Tenants().GetTenantInfo("override")
	if err != nil {
		t.Fatal(err)
	}
	if tinyInfo.Usage.Documents != 5 {
		t.Fatalf("documents after reopen with tighter limits = %d, want 5", tinyInfo.Usage.Documents)
	}
	if _, err := tinyStore.Tenants().GetCollectionInfo("override", "docs"); err != nil {
		t.Fatalf("collection missing after reopen with tighter limits: %v", err)
	}
	if err := tinyStore.Abort(); err != nil {
		t.Fatalf("abort: %v", err)
	}
}
