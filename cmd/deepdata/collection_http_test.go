package main

import (
	"context"
	"crypto/sha256"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	vcollection "github.com/phenomenon0/vectordb/internal/collection"
)

func testFileSHA256(t *testing.T, path string) [sha256.Size]byte {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return sha256.Sum256(data)
}

func TestCollectionHTTPServerLoadDurableRefusesRawLegacyV2WithoutMutation(t *testing.T) {
	dir := t.TempDir()
	basePath := filepath.Join(dir, "index.gob.collections")
	manager := vcollection.NewCollectionManager(basePath)
	if _, err := manager.CreateCollection(context.Background(), vcollection.CollectionSchema{
		Name: "legacy",
		Fields: []vcollection.VectorField{{
			Name: "embedding", Type: vcollection.VectorTypeDense, Dim: 2,
			Index: vcollection.IndexConfig{Type: vcollection.IndexTypeFLAT},
		}},
	}); err != nil {
		t.Fatal(err)
	}
	if err := manager.Save(basePath + ".manager"); err != nil {
		t.Fatal(err)
	}
	tenants := vcollection.NewTenantManager(basePath)
	if _, err := tenants.CreateCollection(context.Background(), "tenant", vcollection.CollectionSchema{
		Name: "docs",
		Fields: []vcollection.VectorField{{
			Name: "embedding", Type: vcollection.VectorTypeDense, Dim: 2,
			Index: vcollection.IndexConfig{Type: vcollection.IndexTypeFLAT},
		}},
	}); err != nil {
		t.Fatal(err)
	}
	if err := tenants.Save(basePath + ".tenants"); err != nil {
		t.Fatal(err)
	}

	rawPaths := []string{basePath + ".manager", basePath + ".tenants"}
	before := make(map[string][sha256.Size]byte, len(rawPaths))
	for _, path := range rawPaths {
		before[path] = testFileSHA256(t, path)
	}

	server := NewCollectionHTTPServer(basePath)
	if err := server.LoadDurable(basePath); err == nil || !strings.Contains(err.Error(), "explicit offline migration") {
		t.Fatalf("raw legacy durable load error = %v", err)
	}
	if server.durableStore != nil {
		t.Fatal("raw legacy refusal opened a durable store")
	}
	for _, path := range rawPaths {
		if got := testFileSHA256(t, path); got != before[path] {
			t.Fatalf("raw legacy artifact changed during refusal: %s", path)
		}
	}
	for _, path := range []string{
		basePath + ".snapshot",
		basePath + ".initialized",
		basePath + ".lock",
		basePath + ".journal",
		basePath + ".journal.frozen",
	} {
		if _, err := os.Lstat(path); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("raw legacy refusal created %s: %v", path, err)
		}
	}
}

func TestCollectionHTTPServerLoadIsAllOrNothing(t *testing.T) {
	dir := t.TempDir()
	live := NewCollectionHTTPServer(filepath.Join(dir, "live"))
	ctx := context.Background()
	if _, err := live.manager.CreateCollection(ctx, vcollection.CollectionSchema{
		Name: "keep",
		Fields: []vcollection.VectorField{{
			Name: "embedding", Type: vcollection.VectorTypeDense, Dim: 4,
			Index: vcollection.IndexConfig{Type: vcollection.IndexTypeFLAT},
		}},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := live.tenantManager.CreateCollection(ctx, "keep-tenant", vcollection.CollectionSchema{
		Name: "docs",
		Fields: []vcollection.VectorField{{
			Name: "embedding", Type: vcollection.VectorTypeDense, Dim: 4,
			Index: vcollection.IndexConfig{Type: vcollection.IndexTypeFLAT},
		}},
	}); err != nil {
		t.Fatal(err)
	}

	basePath := filepath.Join(dir, "incoming")
	seed := NewCollectionHTTPServer(basePath)
	if err := seed.manager.Save(basePath + ".manager"); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(basePath+".tenants", []byte(`{"tenants":{"valid":{"collections":{}},"broken":null}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := live.Load(basePath); err == nil {
		t.Fatal("expected aggregate collection load to fail")
	}
	if !live.manager.HasCollection("keep") || live.manager.CollectionCount() != 1 {
		t.Fatalf("failed aggregate load changed V2 manager: %v", live.manager.ListCollections())
	}
	if got := live.tenantManager.ListTenants(); len(got) != 1 || got[0] != "keep-tenant" {
		t.Fatalf("failed aggregate load changed V3 tenants: %v", got)
	}
}
