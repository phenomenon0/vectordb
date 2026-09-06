package collection

import (
	"context"
	"errors"
	"path/filepath"
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
