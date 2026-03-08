package collection

import (
	"context"
	"fmt"
	"sync"
	"testing"
)

// helper to create a minimal valid schema with a dense FLAT field.
func testSchema(name string, dim int) CollectionSchema {
	return CollectionSchema{
		Name: name,
		Fields: []VectorField{
			{
				Name: "embedding",
				Type: VectorTypeDense,
				Dim:  dim,
				Index: IndexConfig{
					Type:   IndexTypeFLAT,
					Params: map[string]interface{}{},
				},
			},
		},
	}
}

// helper to create a document with a zero vector of the given dimension.
func testDoc(dim int) Document {
	vec := make([]float32, dim)
	return Document{
		Vectors:  map[string]interface{}{"embedding": vec},
		Metadata: map[string]interface{}{"source": "test"},
	}
}

// ---------------------------------------------------------------------------
// TEST: Two tenants can have same-named collections without conflict
// ---------------------------------------------------------------------------

func TestTenantNamespaceIsolation(t *testing.T) {
	tm := NewTenantManager("")
	ctx := context.Background()
	dim := 4

	// Both tenants create a collection named "docs"
	_, err := tm.CreateCollection(ctx, "tenant-a", testSchema("docs", dim))
	if err != nil {
		t.Fatalf("tenant-a create failed: %v", err)
	}

	_, err = tm.CreateCollection(ctx, "tenant-b", testSchema("docs", dim))
	if err != nil {
		t.Fatalf("tenant-b create failed: %v", err)
	}

	// Both should be independently accessible
	collA, err := tm.GetCollection("tenant-a", "docs")
	if err != nil {
		t.Fatalf("tenant-a get failed: %v", err)
	}

	collB, err := tm.GetCollection("tenant-b", "docs")
	if err != nil {
		t.Fatalf("tenant-b get failed: %v", err)
	}

	// They must be distinct objects
	if collA == collB {
		t.Fatal("tenant-a and tenant-b got the same collection pointer — no isolation")
	}

	// Add a doc to tenant-a only
	if err := tm.AddDocument(ctx, "tenant-a", "docs", testDoc(dim)); err != nil {
		t.Fatalf("add doc to tenant-a failed: %v", err)
	}

	if collA.Count() != 1 {
		t.Errorf("tenant-a docs count: want 1, got %d", collA.Count())
	}
	if collB.Count() != 0 {
		t.Errorf("tenant-b docs count: want 0, got %d", collB.Count())
	}
}

// ---------------------------------------------------------------------------
// TEST: Tenant A cannot access Tenant B's collections
// ---------------------------------------------------------------------------

func TestTenantCrossAccessDenied(t *testing.T) {
	tm := NewTenantManager("")
	ctx := context.Background()
	dim := 4

	// Only tenant-a has a collection
	_, err := tm.CreateCollection(ctx, "tenant-a", testSchema("private", dim))
	if err != nil {
		t.Fatalf("create failed: %v", err)
	}

	// tenant-b should NOT be able to access it
	_, err = tm.GetCollection("tenant-b", "private")
	if err == nil {
		t.Fatal("tenant-b accessed tenant-a's collection — isolation breach")
	}

	// tenant-b should NOT be able to add docs to it
	err = tm.AddDocument(ctx, "tenant-b", "private", testDoc(dim))
	if err == nil {
		t.Fatal("tenant-b added doc to tenant-a's collection — isolation breach")
	}

	// tenant-b should NOT be able to delete it
	err = tm.DeleteCollection(ctx, "tenant-b", "private")
	if err == nil {
		t.Fatal("tenant-b deleted tenant-a's collection — isolation breach")
	}

	// tenant-b listing should be empty
	list := tm.ListCollections("tenant-b")
	if len(list) != 0 {
		t.Errorf("tenant-b listing should be empty, got %v", list)
	}
}

// ---------------------------------------------------------------------------
// TEST: CRUD operations work per-tenant
// ---------------------------------------------------------------------------

func TestTenantCRUD(t *testing.T) {
	tm := NewTenantManager("")
	ctx := context.Background()
	dim := 4
	tenantID := "crud-tenant"

	// Create
	_, err := tm.CreateCollection(ctx, tenantID, testSchema("alpha", dim))
	if err != nil {
		t.Fatalf("create alpha failed: %v", err)
	}
	_, err = tm.CreateCollection(ctx, tenantID, testSchema("beta", dim))
	if err != nil {
		t.Fatalf("create beta failed: %v", err)
	}

	// List
	names := tm.ListCollections(tenantID)
	if len(names) != 2 {
		t.Fatalf("expected 2 collections, got %d: %v", len(names), names)
	}

	// Duplicate create should fail
	_, err = tm.CreateCollection(ctx, tenantID, testSchema("alpha", dim))
	if err == nil {
		t.Fatal("duplicate collection creation should fail")
	}

	// Get info
	info, err := tm.GetCollectionInfo(tenantID, "alpha")
	if err != nil {
		t.Fatalf("get info failed: %v", err)
	}
	if info.Name != "alpha" {
		t.Errorf("expected name 'alpha', got %q", info.Name)
	}

	// Add document
	if err := tm.AddDocument(ctx, tenantID, "alpha", testDoc(dim)); err != nil {
		t.Fatalf("add doc failed: %v", err)
	}

	// Stats
	stats, err := tm.GetTenantStats(tenantID)
	if err != nil {
		t.Fatalf("get stats failed: %v", err)
	}
	if stats.TotalDocuments != 1 {
		t.Errorf("expected 1 total doc, got %d", stats.TotalDocuments)
	}
	if stats.CollectionCount != 2 {
		t.Errorf("expected 2 collections, got %d", stats.CollectionCount)
	}

	// Delete collection
	if err := tm.DeleteCollection(ctx, tenantID, "beta"); err != nil {
		t.Fatalf("delete collection failed: %v", err)
	}

	names = tm.ListCollections(tenantID)
	if len(names) != 1 {
		t.Fatalf("expected 1 collection after delete, got %d", len(names))
	}

	// Drop tenant
	if err := tm.DropTenant(ctx, tenantID); err != nil {
		t.Fatalf("drop tenant failed: %v", err)
	}

	if tm.TenantCount() != 0 {
		t.Errorf("expected 0 tenants after drop, got %d", tm.TenantCount())
	}
}

// ---------------------------------------------------------------------------
// TEST: Concurrent tenant operations are safe
// ---------------------------------------------------------------------------

func TestTenantConcurrency(t *testing.T) {
	tm := NewTenantManager("")
	ctx := context.Background()
	dim := 4

	const numTenants = 20
	const numCollections = 5
	const numDocs = 10

	var wg sync.WaitGroup

	// Concurrently create tenants, collections, and add documents
	for i := 0; i < numTenants; i++ {
		wg.Add(1)
		go func(tenantIdx int) {
			defer wg.Done()
			tenantID := fmt.Sprintf("tenant-%d", tenantIdx)

			for j := 0; j < numCollections; j++ {
				collName := fmt.Sprintf("coll-%d", j)
				_, err := tm.CreateCollection(ctx, tenantID, testSchema(collName, dim))
				if err != nil {
					t.Errorf("tenant %s create %s failed: %v", tenantID, collName, err)
					return
				}
			}

			for j := 0; j < numCollections; j++ {
				collName := fmt.Sprintf("coll-%d", j)
				for k := 0; k < numDocs; k++ {
					if err := tm.AddDocument(ctx, tenantID, collName, testDoc(dim)); err != nil {
						t.Errorf("tenant %s add to %s failed: %v", tenantID, collName, err)
						return
					}
				}
			}
		}(i)
	}

	wg.Wait()

	// Verify isolation: each tenant should have exactly numCollections collections
	// and numDocs docs per collection
	tenants := tm.ListTenants()
	if len(tenants) != numTenants {
		t.Fatalf("expected %d tenants, got %d", numTenants, len(tenants))
	}

	for _, tid := range tenants {
		collections := tm.ListCollections(tid)
		if len(collections) != numCollections {
			t.Errorf("tenant %s: expected %d collections, got %d", tid, numCollections, len(collections))
		}

		stats, err := tm.GetTenantStats(tid)
		if err != nil {
			t.Errorf("tenant %s stats failed: %v", tid, err)
			continue
		}
		expectedDocs := numCollections * numDocs
		if stats.TotalDocuments != expectedDocs {
			t.Errorf("tenant %s: expected %d total docs, got %d", tid, expectedDocs, stats.TotalDocuments)
		}
	}
}

// ---------------------------------------------------------------------------
// TEST: Edge cases
// ---------------------------------------------------------------------------

func TestTenantEdgeCases(t *testing.T) {
	tm := NewTenantManager("")
	ctx := context.Background()

	// Empty tenant ID should fail
	_, err := tm.CreateCollection(ctx, "", testSchema("x", 4))
	if err == nil {
		t.Error("empty tenant ID should fail")
	}

	_, err = tm.GetCollection("", "x")
	if err == nil {
		t.Error("empty tenant ID GetCollection should fail")
	}

	// Non-existent tenant GetCollection should fail
	_, err = tm.GetCollection("ghost", "x")
	if err == nil {
		t.Error("non-existent tenant GetCollection should fail")
	}

	// Non-existent tenant ListCollections returns empty
	list := tm.ListCollections("ghost")
	if len(list) != 0 {
		t.Errorf("expected empty list for ghost tenant, got %v", list)
	}

	// DropTenant on non-existent tenant should fail
	err = tm.DropTenant(ctx, "ghost")
	if err == nil {
		t.Error("dropping non-existent tenant should fail")
	}
}
