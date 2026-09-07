package collection

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func storeSetSearchRequest(collectionName string) SearchRequest {
	return SearchRequest{
		CollectionName: collectionName,
		Queries:        map[string]interface{}{"embedding": []float32{1, 0, 0, 0}},
		TopK:           1,
	}
}

func TestStoreSetOneStorePerTenant(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	limits := StoreLimits{MaxTenants: 8, MaxCollections: 8}

	set, err := OpenStoreSet(dir, limits)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := set.CreateCollection(ctx, "acme", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}
	if _, err := set.CreateCollection(ctx, "globex", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"acme.initialized", "acme.journal", "globex.initialized", "globex.journal"} {
		if _, err := os.Stat(filepath.Join(dir, name)); err != nil {
			t.Fatalf("expected %s on disk: %v", name, err)
		}
	}
	acmeInfosBefore, err := set.ListCollectionInfosChecked("acme")
	if err != nil {
		t.Fatal(err)
	}
	globexInfosBefore, err := set.ListCollectionInfosChecked("globex")
	if err != nil {
		t.Fatal(err)
	}
	if err := set.Close(); err != nil {
		t.Fatal(err)
	}

	reopened, err := OpenStoreSet(dir, limits)
	if err != nil {
		t.Fatal(err)
	}

	if got := reopened.Tenants(); !reflect.DeepEqual(got, []string{"acme", "globex"}) {
		t.Fatalf("Tenants() = %v, want [acme globex]", got)
	}
	acmeInfosAfter, err := reopened.ListCollectionInfosChecked("acme")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(acmeInfosBefore, acmeInfosAfter) {
		t.Fatalf("acme collection infos changed across reopen: %+v vs %+v", acmeInfosBefore, acmeInfosAfter)
	}
	globexInfosAfter, err := reopened.ListCollectionInfosChecked("globex")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(globexInfosBefore, globexInfosAfter) {
		t.Fatalf("globex collection infos changed across reopen: %+v vs %+v", globexInfosBefore, globexInfosAfter)
	}

	if err := reopened.Close(); err != nil {
		t.Fatal(err)
	}

	// A dir that already holds two tenants refuses to admit a third once
	// MaxTenants is lowered to exactly that count.
	capped, err := OpenStoreSet(dir, StoreLimits{MaxTenants: 2, MaxCollections: 8})
	if err != nil {
		t.Fatal(err)
	}
	defer capped.Close()
	if _, err := capped.CreateCollection(ctx, "umbrella", durableTestSchema("docs")); !errors.Is(err, ErrTenantLimitExceeded) {
		t.Fatalf("third tenant create err = %v, want ErrTenantLimitExceeded", err)
	}

	// A dir-wide MaxCollections cap is enforced across every tenant's store,
	// not just one tenant's own.
	collDir := t.TempDir()
	collSet, err := OpenStoreSet(collDir, StoreLimits{MaxTenants: 8, MaxCollections: 2})
	if err != nil {
		t.Fatal(err)
	}
	defer collSet.Close()
	if _, err := collSet.CreateCollection(ctx, "acme", durableTestSchema("one")); err != nil {
		t.Fatal(err)
	}
	if _, err := collSet.CreateCollection(ctx, "globex", durableTestSchema("one")); err != nil {
		t.Fatal(err)
	}
	if _, err := collSet.CreateCollection(ctx, "acme", durableTestSchema("two")); !errors.Is(err, ErrCollectionLimitExceeded) {
		t.Fatalf("collection over dir-wide cap err = %v, want ErrCollectionLimitExceeded", err)
	}
}

func TestStoreSetRejectsInvalidTenantIDs(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	set, err := OpenStoreSet(dir, StoreLimits{MaxTenants: 8, MaxCollections: 8})
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()

	for _, id := range []string{"../etc", "", "a/b", strings.Repeat("x", 65)} {
		if _, err := set.CreateCollection(ctx, id, durableTestSchema("docs")); !errors.Is(err, ErrInvalidTenantID) {
			t.Fatalf("tenant %q: err = %v, want ErrInvalidTenantID", id, err)
		}
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("expected nothing written to disk, found %v", entries)
	}
}

func TestStoreSetIsolatesBrokenTenant(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	limits := StoreLimits{MaxTenants: 8, MaxCollections: 8}

	set, err := OpenStoreSet(dir, limits)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := set.CreateCollection(ctx, "acme", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}
	if _, err := set.CreateCollection(ctx, "globex", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}
	// Abort, not Close: Close would checkpoint and rewrite the journal,
	// leaving nothing meaningful to corrupt (mirrors
	// TestDurableStoreRejectsCorruptJournalFrame).
	if err := set.Abort(); err != nil {
		t.Fatal(err)
	}

	journalPath := filepath.Join(dir, "acme.journal")
	original, err := os.ReadFile(journalPath)
	if err != nil {
		t.Fatal(err)
	}
	corrupt := append([]byte(nil), original...)
	corrupt[len(corrupt)-1] ^= 0xff
	if err := os.WriteFile(journalPath, corrupt, 0o600); err != nil {
		t.Fatal(err)
	}

	reopened, err := OpenStoreSet(dir, limits)
	if err != nil {
		t.Fatalf("OpenStoreSet with one corrupt tenant: %v", err)
	}
	defer reopened.Close()

	if _, ok := reopened.FaultedTenants()["acme"]; !ok {
		t.Fatalf("FaultedTenants() = %v, want acme present", reopened.FaultedTenants())
	}
	if _, err := reopened.CreateCollection(ctx, "acme", durableTestSchema("more")); !errors.Is(err, ErrDurableStoreFaulted) {
		t.Fatalf("acme create err = %v, want ErrDurableStoreFaulted", err)
	}
	if _, err := reopened.ListCollectionInfosChecked("acme"); !errors.Is(err, ErrDurableStoreFaulted) {
		t.Fatalf("acme list err = %v, want ErrDurableStoreFaulted", err)
	}

	if _, err := reopened.CreateCollection(ctx, "globex", durableTestSchema("second")); err != nil {
		t.Fatalf("globex create: %v", err)
	}
	if _, err := reopened.SearchCollection(ctx, "globex", storeSetSearchRequest("second")); err != nil {
		t.Fatalf("globex search: %v", err)
	}

	after, err := os.ReadFile(journalPath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(after, corrupt) {
		t.Fatal("acme journal bytes changed after a failed open")
	}
}

func TestStoreSetDeleteTenantRemovesOnlyThatPrefix(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	limits := StoreLimits{MaxTenants: 8, MaxCollections: 8}

	set, err := OpenStoreSet(dir, limits)
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()

	if _, err := set.CreateCollection(ctx, "acme", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}
	if _, err := set.CreateCollection(ctx, "globex", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}
	if err := set.Checkpoint(); err != nil {
		t.Fatal(err)
	}

	globexBefore := map[string][]byte{}
	for _, suffix := range tenantArtifactSuffixes {
		data, err := os.ReadFile(filepath.Join(dir, "globex"+suffix))
		if err == nil {
			globexBefore[suffix] = data
		}
	}
	if len(globexBefore) == 0 {
		t.Fatal("expected at least one globex artifact on disk before delete")
	}

	if err := set.DeleteTenant(ctx, "acme"); err != nil {
		t.Fatal(err)
	}
	for _, suffix := range tenantArtifactSuffixes {
		path := filepath.Join(dir, "acme"+suffix)
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Fatalf("acme%s still present after DeleteTenant: %v", suffix, err)
		}
	}
	for suffix, want := range globexBefore {
		got, err := os.ReadFile(filepath.Join(dir, "globex"+suffix))
		if err != nil {
			t.Fatalf("globex%s missing after deleting acme: %v", suffix, err)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("globex%s changed after deleting acme", suffix)
		}
	}

	if _, ok := set.Store("acme"); ok {
		t.Fatal("acme still registered as an open store")
	}
	if err := set.DeleteTenant(ctx, "acme"); !errors.Is(err, ErrTenantNotFound) {
		t.Fatalf("second delete err = %v, want ErrTenantNotFound", err)
	}
}

func TestExportTenantSnapshotIsOpenable(t *testing.T) {
	ctx := context.Background()
	limits := StoreLimits{MaxTenants: 8, MaxCollections: 8}
	srcBase := filepath.Join(t.TempDir(), "unified")

	src, err := OpenDurableStoreWithLimits(srcBase, srcBase, limits)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := src.Tenants().CreateCollection(ctx, "acme", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}
	if _, err := src.Tenants().CreateCollection(ctx, "globex", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}

	var lastID uint64
	for i := 0; i < 3; i++ {
		doc := durableTestDocument(float32(i))
		if err := src.Tenants().AddDocument(ctx, "acme", "docs", &doc); err != nil {
			t.Fatal(err)
		}
		lastID = doc.ID
	}
	// Delete the highest-ID document so the tenant's next_id cursor sits
	// above its live document count.
	if err := src.Tenants().DeleteDocument(ctx, "acme", "docs", lastID); err != nil {
		t.Fatal(err)
	}

	quota := TenantRecord{TenantID: "acme", Status: TenantStatusActive, Quota: TenantQuota{MaxDocuments: 100}}
	if err := src.Tenants().UpdateTenant(ctx, quota); err != nil {
		t.Fatal(err)
	}

	dstDir := t.TempDir()
	if err := ExportTenantSnapshot(src, "acme", filepath.Join(dstDir, "acme")); err != nil {
		t.Fatal(err)
	}
	srcStoreID := src.Metadata().StoreID
	if err := src.Close(); err != nil {
		t.Fatal(err)
	}

	set, err := OpenStoreSet(dstDir, limits)
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()

	if got := set.Tenants(); !reflect.DeepEqual(got, []string{"acme"}) {
		t.Fatalf("Tenants() = %v, want [acme] (globex must not have exported)", got)
	}
	for id := uint64(1); id < lastID; id++ {
		if _, err := set.GetDocumentChecked("acme", "docs", id); err != nil {
			t.Fatalf("document %d missing after export: %v", id, err)
		}
	}

	// Check the freshly opened metadata before any new mutation bumps
	// AppliedLSN off of its exported starting point.
	acmeStore, ok := set.Store("acme")
	if !ok {
		t.Fatal("acme store not open")
	}
	if got := acmeStore.Metadata().AppliedLSN; got != 0 {
		t.Fatalf("AppliedLSN = %d, want 0", got)
	}
	if acmeStore.Metadata().StoreID == srcStoreID {
		t.Fatal("exported StoreID equals the source store's StoreID")
	}

	newDoc := durableTestDocument(9)
	if err := set.AddDocument(ctx, "acme", "docs", &newDoc); err != nil {
		t.Fatal(err)
	}
	if newDoc.ID <= lastID {
		t.Fatalf("new document ID %d did not advance past the exported next_id (%d)", newDoc.ID, lastID)
	}

	rec, ok := set.GetTenantRecord("acme")
	if !ok || rec.Quota.MaxDocuments != 100 {
		t.Fatalf("GetTenantRecord(acme) = %+v, %v, want the exported quota", rec, ok)
	}
}

// TestStoreSetRefusesTenantHeldByAnotherOpen pins the documented rule that
// whichever process opens a tenant second is refused with "collection store
// is already open": a held lock is a deployment conflict, so the whole
// directory refuses to open (and releases what it had opened) rather than
// booting with that tenant silently faulted.
func TestStoreSetRefusesTenantHeldByAnotherOpen(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	limits := StoreLimits{MaxTenants: 8, MaxCollections: 8}

	set, err := OpenStoreSet(dir, limits)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := set.CreateCollection(ctx, "acme", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}
	if _, err := set.CreateCollection(ctx, "globex", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}
	if err := set.Abort(); err != nil {
		t.Fatal(err)
	}

	// Hold only globex, so the set opens acme first and must release it again.
	globexBase := filepath.Join(dir, "globex")
	holder, err := OpenDurableStore(globexBase, globexBase)
	if err != nil {
		t.Fatal(err)
	}
	defer abandonDurableStoreForTest(t, holder)

	second, err := OpenStoreSet(dir, limits)
	if err == nil {
		_ = second.Abort()
		t.Fatal("OpenStoreSet succeeded while another open holds globex")
	}
	if !errors.Is(err, ErrCollectionStoreLocked) || !strings.Contains(err.Error(), `"globex"`) {
		t.Fatalf("OpenStoreSet err = %v, want ErrCollectionStoreLocked naming globex", err)
	}
	acmeBase := filepath.Join(dir, "acme")
	acme, err := OpenDurableStore(acmeBase, acmeBase)
	if err != nil {
		t.Fatalf("acme still locked after the refused open: %v", err)
	}
	abandonDurableStoreForTest(t, acme)
}

// TestStoreSetAdoptTenantRunsBindBeforeServing pins the reason AdoptTenant
// exists: a tenant's files can appear on disk after the set has already
// booted (a follower seeding it mid-run), and bind must run while s.mu is
// still held so no request can reach the store before a replica marker is
// honored.
func TestStoreSetAdoptTenantRunsBindBeforeServing(t *testing.T) {
	ctx := context.Background()
	limits := StoreLimits{MaxTenants: 8, MaxCollections: 8}

	srcBase := filepath.Join(t.TempDir(), "unified")
	src, err := OpenDurableStoreWithLimits(srcBase, srcBase, limits)
	if err != nil {
		t.Fatal(err)
	}
	for _, tenantID := range []string{"acme", "globex"} {
		if _, err := src.Tenants().CreateCollection(ctx, tenantID, durableTestSchema("docs")); err != nil {
			t.Fatal(err)
		}
	}

	dir := t.TempDir()
	set, err := OpenStoreSet(dir, limits)
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()
	if got := set.Tenants(); len(got) != 0 {
		t.Fatalf("Tenants() before any export = %v, want none", got)
	}

	// The tenant's artifacts appear on disk after boot, exactly as a
	// follower's Seed would leave them.
	if err := ExportTenantSnapshot(src, "acme", filepath.Join(dir, "acme")); err != nil {
		t.Fatal(err)
	}
	if err := ExportTenantSnapshot(src, "globex", filepath.Join(dir, "globex")); err != nil {
		t.Fatal(err)
	}
	if err := src.Close(); err != nil {
		t.Fatal(err)
	}

	leaderID := [16]byte{1, 2, 3}
	store, err := set.AdoptTenant("acme", func(s *DurableStore) error {
		return s.MakeReplica(leaderID)
	})
	if err != nil {
		t.Fatalf("AdoptTenant(acme): %v", err)
	}
	if !store.IsReplica() {
		t.Fatal("AdoptTenant did not run bind before returning the store")
	}
	doc := durableTestDocument(1)
	if err := set.AddDocument(ctx, "acme", "docs", &doc); !errors.Is(err, ErrReplicaReadOnly) {
		t.Fatalf("AddDocument on an adopted replica = %v, want ErrReplicaReadOnly", err)
	}
	// Re-adopting an already-open tenant returns it unchanged; bind must not
	// run again.
	again, err := set.AdoptTenant("acme", func(*DurableStore) error {
		t.Fatal("bind called again for an already-open tenant")
		return nil
	})
	if err != nil || again != store {
		t.Fatalf("AdoptTenant(acme) again = %v, %v, want the same store back", again, err)
	}

	// A bind that fails must leave the tenant absent and its lock released.
	_, err = set.AdoptTenant("globex", func(*DurableStore) error {
		return errors.New("bind refused")
	})
	if err == nil || !strings.Contains(err.Error(), "bind refused") {
		t.Fatalf("AdoptTenant(globex) = %v, want the bind error", err)
	}
	for _, id := range set.Tenants() {
		if id == "globex" {
			t.Fatal("globex is registered despite its bind failing")
		}
	}
	globexBase := filepath.Join(dir, "globex")
	reopened, err := OpenDurableStore(globexBase, globexBase)
	if err != nil {
		t.Fatalf("globex still locked after a refused bind: %v", err)
	}
	if err := reopened.Close(); err != nil {
		t.Fatal(err)
	}
}

// TestStoreSetReadOnlyRefusesNewTenants pins the reason SetReadOnly exists: a
// following node must not mint a local tenant the leader may still ship, but
// it must not stop serving tenants it already has.
func TestStoreSetReadOnlyRefusesNewTenants(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	limits := StoreLimits{MaxTenants: 8, MaxCollections: 8}
	set, err := OpenStoreSet(dir, limits)
	if err != nil {
		t.Fatal(err)
	}
	defer set.Close()
	if _, err := set.CreateCollection(ctx, "acme", durableTestSchema("docs")); err != nil {
		t.Fatal(err)
	}

	set.SetReadOnly(true)
	if _, err := set.CreateCollection(ctx, "globex", durableTestSchema("docs")); !errors.Is(err, ErrReplicaReadOnly) {
		t.Fatalf("CreateCollection(globex) while read-only = %v, want ErrReplicaReadOnly", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "globex.initialized")); !os.IsNotExist(err) {
		t.Fatalf("globex must not have been minted on disk: %v", err)
	}
	// An existing tenant is unaffected: it keeps taking writes.
	if _, err := set.CreateCollection(ctx, "acme", durableTestSchema("more")); err != nil {
		t.Fatalf("existing tenant refused a write while the set is read-only: %v", err)
	}

	set.SetReadOnly(false)
	if _, err := set.CreateCollection(ctx, "globex", durableTestSchema("docs")); err != nil {
		t.Fatalf("CreateCollection(globex) after SetReadOnly(false): %v", err)
	}
}
