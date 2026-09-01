package collection

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func limitsTestSchema(name string) CollectionSchema {
	return CollectionSchema{
		Name: name,
		Fields: []VectorField{{
			Name:  "embedding",
			Type:  VectorTypeDense,
			Dim:   4,
			Index: IndexConfig{Type: IndexTypeFLAT},
		}},
	}
}

func limitsTestSchemaWithFields(name string, count, dimension int) CollectionSchema {
	fields := make([]VectorField, count)
	for i := range fields {
		fields[i] = VectorField{
			Name:  fmt.Sprintf("field_%d", i),
			Type:  VectorTypeDense,
			Dim:   dimension,
			Index: IndexConfig{Type: IndexTypeFLAT},
		}
	}
	return CollectionSchema{Name: name, Fields: fields}
}

func openLimitsTestStore(t *testing.T, base string, limits StoreLimits) *DurableStore {
	t.Helper()
	store, err := OpenDurableStoreWithLimits(base, base, limits)
	if err != nil {
		t.Fatalf("open durable store with limits %+v: %v", limits, err)
	}
	t.Cleanup(func() {
		if err := store.Abort(); err != nil {
			t.Errorf("abort durable store: %v", err)
		}
	})
	return store
}

func requireLimitsTestCounts(t *testing.T, store *DurableStore, wantTenants, wantCollections int) {
	t.Helper()
	gotTenants, gotCollections := store.Tenants().resourceCounts()
	if gotTenants != wantTenants || gotCollections != wantCollections {
		t.Fatalf(
			"resource counts = (%d tenants, %d collections), want (%d, %d)",
			gotTenants,
			gotCollections,
			wantTenants,
			wantCollections,
		)
	}
	if store.activeTenants != wantTenants || store.collectionCount != wantCollections {
		t.Fatalf(
			"cached resource counts = (%d tenants, %d collections), want (%d, %d)",
			store.activeTenants,
			store.collectionCount,
			wantTenants,
			wantCollections,
		)
	}
}

func TestStoreLimitsValidation(t *testing.T) {
	tests := []struct {
		name    string
		limits  StoreLimits
		wantErr string
	}{
		{name: "zero value", limits: StoreLimits{}, wantErr: "max tenants must be positive"},
		{name: "zero tenants", limits: StoreLimits{MaxCollections: 1}, wantErr: "max tenants must be positive"},
		{name: "negative tenants", limits: StoreLimits{MaxTenants: -1, MaxCollections: 1}, wantErr: "max tenants must be positive"},
		{name: "zero collections", limits: StoreLimits{MaxTenants: 1}, wantErr: "max collections must be positive"},
		{name: "negative collections", limits: StoreLimits{MaxTenants: 1, MaxCollections: -1}, wantErr: "max collections must be positive"},
		{name: "valid", limits: StoreLimits{MaxTenants: 1, MaxCollections: 1}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := test.limits.validateRequired()
			if test.wantErr == "" {
				if err != nil {
					t.Fatalf("validate valid limits: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("validation error = %v, want containing %q", err, test.wantErr)
			}
		})
	}

	base := filepath.Join(t.TempDir(), "invalid-constructor")
	if store, err := OpenDurableStoreWithLimits(base, base, StoreLimits{}); err == nil {
		_ = store.Abort()
		t.Fatal("OpenDurableStoreWithLimits accepted zero limits")
	}
}

func TestDurableStoreLimitsNPlusOneIsAtomicAcrossRestart(t *testing.T) {
	ctx := context.Background()

	t.Run("tenant limit", func(t *testing.T) {
		base := filepath.Join(t.TempDir(), "collections")
		limits := StoreLimits{MaxTenants: 2, MaxCollections: 10}
		store := openLimitsTestStore(t, base, limits)

		if _, err := store.Tenants().CreateCollection(ctx, "tenant-a", limitsTestSchema("alpha")); err != nil {
			t.Fatal(err)
		}
		if _, err := store.Tenants().CreateCollection(ctx, "tenant-b", limitsTestSchema("beta")); err != nil {
			t.Fatal(err)
		}
		before := store.Metadata().AppliedLSN
		if _, err := store.Tenants().CreateCollection(ctx, "tenant-c", limitsTestSchema("rejected")); !errors.Is(err, ErrTenantLimitExceeded) {
			t.Fatalf("N+1 tenant create error = %v, want ErrTenantLimitExceeded", err)
		}
		if got := store.Metadata().AppliedLSN; got != before {
			t.Fatalf("rejected tenant create advanced LSN from %d to %d", before, got)
		}
		if err := store.Err(); err != nil {
			t.Fatalf("rejected tenant create faulted store: %v", err)
		}
		requireLimitsTestCounts(t, store, 2, 2)
		ids, err := store.Tenants().ListTenantsChecked()
		if err != nil {
			t.Fatal(err)
		}
		if len(ids) != 2 || ids[0] != "tenant-a" || ids[1] != "tenant-b" {
			t.Fatalf("tenant map changed after rejected create: %v", ids)
		}

		if err := store.Abort(); err != nil {
			t.Fatalf("abort before replay: %v", err)
		}
		reopened := openLimitsTestStore(t, base, limits)
		if got := reopened.Metadata().AppliedLSN; got != before {
			t.Fatalf("replayed LSN = %d, want %d", got, before)
		}
		requireLimitsTestCounts(t, reopened, 2, 2)
		if _, err := reopened.Tenants().GetCollectionInfo("tenant-c", "rejected"); err == nil {
			t.Fatal("rejected tenant collection appeared after restart")
		}
	})

	t.Run("collection limit", func(t *testing.T) {
		base := filepath.Join(t.TempDir(), "collections")
		limits := StoreLimits{MaxTenants: 10, MaxCollections: 2}
		store := openLimitsTestStore(t, base, limits)

		if _, err := store.Tenants().CreateCollection(ctx, "tenant-a", limitsTestSchema("alpha")); err != nil {
			t.Fatal(err)
		}
		if _, err := store.Tenants().CreateCollection(ctx, "tenant-a", limitsTestSchema("beta")); err != nil {
			t.Fatal(err)
		}
		before := store.Metadata().AppliedLSN
		if _, err := store.Tenants().CreateCollection(ctx, "tenant-a", limitsTestSchema("rejected")); !errors.Is(err, ErrCollectionLimitExceeded) {
			t.Fatalf("N+1 collection create error = %v, want ErrCollectionLimitExceeded", err)
		}
		if got := store.Metadata().AppliedLSN; got != before {
			t.Fatalf("rejected collection create advanced LSN from %d to %d", before, got)
		}
		if err := store.Err(); err != nil {
			t.Fatalf("rejected collection create faulted store: %v", err)
		}
		requireLimitsTestCounts(t, store, 1, 2)

		if err := store.Abort(); err != nil {
			t.Fatalf("abort before replay: %v", err)
		}
		reopened := openLimitsTestStore(t, base, limits)
		if got := reopened.Metadata().AppliedLSN; got != before {
			t.Fatalf("replayed LSN = %d, want %d", got, before)
		}
		requireLimitsTestCounts(t, reopened, 1, 2)
		if _, err := reopened.Tenants().GetCollectionInfo("tenant-a", "rejected"); err == nil {
			t.Fatal("rejected collection appeared after restart")
		}
	})
}

func TestDurableStoreLowerLimitsDoNotBlockReplay(t *testing.T) {
	ctx := context.Background()
	base := filepath.Join(t.TempDir(), "collections")
	high := StoreLimits{MaxTenants: 4, MaxCollections: 4}
	store := openLimitsTestStore(t, base, high)

	if _, err := store.Tenants().CreateCollection(ctx, "tenant-a", limitsTestSchema("alpha")); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Tenants().CreateCollection(ctx, "tenant-b", limitsTestSchema("beta")); err != nil {
		t.Fatal(err)
	}
	wantLSN := store.Metadata().AppliedLSN
	if err := store.Abort(); err != nil {
		t.Fatalf("abort with uncheckpointed creates: %v", err)
	}

	reopened := openLimitsTestStore(t, base, StoreLimits{MaxTenants: 1, MaxCollections: 1})
	if got := reopened.Metadata().AppliedLSN; got != wantLSN {
		t.Fatalf("replayed LSN under lower limits = %d, want %d", got, wantLSN)
	}
	requireLimitsTestCounts(t, reopened, 2, 2)
	for tenant, collectionName := range map[string]string{"tenant-a": "alpha", "tenant-b": "beta"} {
		if _, err := reopened.Tenants().GetCollectionInfo(tenant, collectionName); err != nil {
			t.Fatalf("acknowledged collection %s/%s missing after replay: %v", tenant, collectionName, err)
		}
	}

	before := reopened.Metadata().AppliedLSN
	_, err := reopened.Tenants().CreateCollection(ctx, "tenant-a", limitsTestSchema("blocked"))
	if !errors.Is(err, ErrCollectionLimitExceeded) {
		t.Fatalf("new create above lowered limit error = %v, want ErrCollectionLimitExceeded", err)
	}
	if got := reopened.Metadata().AppliedLSN; got != before {
		t.Fatalf("rejected post-replay create advanced LSN from %d to %d", before, got)
	}
}

func TestDurableStoreConcurrentCreatesRespectCollectionCap(t *testing.T) {
	ctx := context.Background()
	base := filepath.Join(t.TempDir(), "collections")
	store := openLimitsTestStore(t, base, StoreLimits{MaxTenants: 32, MaxCollections: 1})

	const workers = 32
	start := make(chan struct{})
	errs := make(chan error, workers)
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			_, err := store.Tenants().CreateCollection(
				ctx,
				fmt.Sprintf("tenant-%d", i),
				limitsTestSchema(fmt.Sprintf("collection-%d", i)),
			)
			errs <- err
		}(i)
	}
	close(start)
	wg.Wait()
	close(errs)

	successes := 0
	limitFailures := 0
	for err := range errs {
		switch {
		case err == nil:
			successes++
		case errors.Is(err, ErrCollectionLimitExceeded):
			limitFailures++
		default:
			t.Fatalf("concurrent create returned unexpected error: %v", err)
		}
	}
	if successes != 1 || limitFailures != workers-1 {
		t.Fatalf("concurrent creates = %d successes and %d limit failures, want 1 and %d", successes, limitFailures, workers-1)
	}
	if got := store.Metadata().AppliedLSN; got != 1 {
		t.Fatalf("concurrent capped creates applied LSN = %d, want 1", got)
	}
	if err := store.Err(); err != nil {
		t.Fatalf("concurrent limit rejections faulted store: %v", err)
	}
	requireLimitsTestCounts(t, store, 1, 1)
}

func TestDurableStoreTenantChurnPrunesManagersAcrossRestart(t *testing.T) {
	ctx := context.Background()
	base := filepath.Join(t.TempDir(), "collections")
	limits := StoreLimits{MaxTenants: 1, MaxCollections: 1}
	store := openLimitsTestStore(t, base, limits)

	for i := 0; i < 16; i++ {
		tenantID := fmt.Sprintf("tenant-%d", i)
		if _, err := store.Tenants().CreateCollection(ctx, tenantID, limitsTestSchema("docs")); err != nil {
			t.Fatalf("churn create %s: %v", tenantID, err)
		}
		if err := store.Tenants().DeleteCollection(ctx, tenantID, "docs"); err != nil {
			t.Fatalf("churn delete %s: %v", tenantID, err)
		}
		requireLimitsTestCounts(t, store, 0, 0)
		ids, err := store.Tenants().ListTenantsChecked()
		if err != nil {
			t.Fatal(err)
		}
		if len(ids) != 0 {
			t.Fatalf("empty tenant manager retained after churn iteration %d: %v", i, ids)
		}
	}

	wantLSN := store.Metadata().AppliedLSN
	if err := store.Abort(); err != nil {
		t.Fatalf("abort churn store: %v", err)
	}
	reopened := openLimitsTestStore(t, base, limits)
	if got := reopened.Metadata().AppliedLSN; got != wantLSN {
		t.Fatalf("replayed churn LSN = %d, want %d", got, wantLSN)
	}
	requireLimitsTestCounts(t, reopened, 0, 0)
	ids, err := reopened.Tenants().ListTenantsChecked()
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 0 {
		t.Fatalf("empty tenant managers survived restart: %v", ids)
	}
	if _, err := reopened.Tenants().CreateCollection(ctx, "replacement", limitsTestSchema("docs")); err != nil {
		t.Fatalf("replacement tenant after churn/restart: %v", err)
	}
}

func TestCanonicalSchemaResourceBounds(t *testing.T) {
	ctx := context.Background()
	base := filepath.Join(t.TempDir(), "collections")
	store := openLimitsTestStore(t, base, StoreLimits{MaxTenants: 10, MaxCollections: 10})

	maxDimension := limitsTestSchema("dimension-max")
	maxDimension.Fields[0].Dim = MaxVectorDimension
	if _, err := store.Tenants().CreateCollection(ctx, "tenant", maxDimension); err != nil {
		t.Fatalf("create schema at maximum dimension: %v", err)
	}

	overDimension := limitsTestSchema("dimension-over")
	overDimension.Fields[0].Dim = MaxVectorDimension + 1
	before := store.Metadata().AppliedLSN
	if _, err := store.Tenants().CreateCollection(ctx, "tenant", overDimension); err == nil || !strings.Contains(err.Error(), "exceeds maximum") {
		t.Fatalf("dimension max+1 error = %v, want exceeds maximum", err)
	}
	if got := store.Metadata().AppliedLSN; got != before {
		t.Fatalf("rejected dimension max+1 advanced LSN from %d to %d", before, got)
	}

	if _, err := store.Tenants().CreateCollection(
		ctx,
		"tenant",
		limitsTestSchemaWithFields("fields-max", MaxSchemaFields, 1),
	); err != nil {
		t.Fatalf("create schema at maximum field count: %v", err)
	}

	before = store.Metadata().AppliedLSN
	if _, err := store.Tenants().CreateCollection(
		ctx,
		"tenant",
		limitsTestSchemaWithFields("fields-over", MaxSchemaFields+1, 1),
	); err == nil || !strings.Contains(err.Error(), "maximum") {
		t.Fatalf("field count max+1 error = %v, want maximum", err)
	}
	if got := store.Metadata().AppliedLSN; got != before {
		t.Fatalf("rejected field count max+1 advanced LSN from %d to %d", before, got)
	}

	overMetadata := limitsTestSchema("metadata-over")
	overMetadata.Metadata = map[string]interface{}{
		"payload": strings.Repeat("x", MaxSchemaMetadataBytes),
	}
	before = store.Metadata().AppliedLSN
	if _, err := store.Tenants().CreateCollection(ctx, "tenant", overMetadata); err == nil || !strings.Contains(err.Error(), "metadata") {
		t.Fatalf("oversized schema metadata error = %v, want metadata limit error", err)
	}
	if got := store.Metadata().AppliedLSN; got != before {
		t.Fatalf("rejected oversized metadata advanced LSN from %d to %d", before, got)
	}
}

func TestSearchIncludeVectorsHonorsResponseBudget(t *testing.T) {
	schema := limitsTestSchema("budget")
	schema.Fields[0].Dim = MaxVectorDimension
	coll, err := NewCollection(schema)
	if err != nil {
		t.Fatalf("create search-budget collection: %v", err)
	}
	t.Cleanup(coll.Close)

	query := make([]float32, MaxVectorDimension)
	query[0] = 1
	if err := coll.Add(context.Background(), &Document{
		ID:      1,
		Vectors: map[string]interface{}{"embedding": append([]float32(nil), query...)},
		Metadata: map[string]interface{}{
			"kind": "budget-test",
		},
	}); err != nil {
		t.Fatalf("add maximum-dimension document: %v", err)
	}
	includeVectors := true
	request := SearchRequest{
		CollectionName: schema.Name,
		Queries:        map[string]interface{}{"embedding": query},
		TopK:           MaxSearchTopK,
		IncludeVectors: &includeVectors,
	}
	if _, err := coll.Search(context.Background(), request); !errors.Is(err, ErrSearchResponseBudgetExceeded) {
		t.Fatalf("include_vectors search error = %v, want ErrSearchResponseBudgetExceeded", err)
	}

	includeVectors = false
	response, err := coll.Search(context.Background(), request)
	if err != nil {
		t.Fatalf("same search without vectors failed: %v", err)
	}
	if response == nil || len(response.Documents) != 1 {
		t.Fatalf("no-vectors search response = %+v, want one result", response)
	}
	if response.Documents[0].Vectors != nil {
		t.Fatalf("no-vectors search materialized vectors: %T", response.Documents[0].Vectors)
	}
	if got := response.Documents[0].Metadata["kind"]; got != "budget-test" {
		t.Fatalf("no-vectors search metadata = %v, want budget-test", got)
	}
}

func TestSearchMetadataHonorsResponseBudgetBeforeCloning(t *testing.T) {
	schema := limitsTestSchema("metadata-budget")
	coll, err := NewCollection(schema)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(coll.Close)

	largeValue := strings.Repeat("x", 6<<20)
	for id := uint64(1); id <= 3; id++ {
		if err := coll.Add(context.Background(), &Document{
			ID:       id,
			Vectors:  map[string]interface{}{"embedding": []float32{1, 0, 0, 0}},
			Metadata: map[string]interface{}{"payload": largeValue},
		}); err != nil {
			t.Fatalf("add metadata-heavy document %d: %v", id, err)
		}
	}

	includeVectors := false
	_, err = coll.Search(context.Background(), SearchRequest{
		CollectionName: schema.Name,
		Queries:        map[string]interface{}{"embedding": []float32{1, 0, 0, 0}},
		TopK:           3,
		IncludeVectors: &includeVectors,
	})
	if !errors.Is(err, ErrSearchResponseBudgetExceeded) {
		t.Fatalf("metadata-heavy search error = %v, want ErrSearchResponseBudgetExceeded", err)
	}
}
