package main

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	deepdatav3 "github.com/phenomenon0/vectordb/api/gen/deepdata/v3"
	vcollection "github.com/phenomenon0/vectordb/internal/collection"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
)

const (
	migrateTenantsHelperEnv = "DEEPDATA_MIGRATE_TENANTS_HELPER"
	migrateTenantsDirEnv    = "DEEPDATA_MIGRATE_TENANTS_DIR"
)

// TestMigrateTenantsProcessHelper is the re-exec target runMigrateTenantsProcess
// launches: same trick as TestCanonicalServerProcessHelper, but for the
// migrate-tenants subcommand instead of serve.
func TestMigrateTenantsProcessHelper(t *testing.T) {
	if os.Getenv(migrateTenantsHelperEnv) != "1" {
		return
	}
	os.Args = []string{"deepdata", "migrate-tenants", os.Getenv(migrateTenantsDirEnv)}
	main()
}

// runMigrateTenantsProcess runs `deepdata migrate-tenants dataDir` as a real
// subprocess and returns its exit code and combined output.
func runMigrateTenantsProcess(t *testing.T, dataDir string) (rc int, output string) {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^TestMigrateTenantsProcessHelper$")
	cmd.Env = processEnv(map[string]string{
		migrateTenantsHelperEnv: "1",
		migrateTenantsDirEnv:    dataDir,
	})
	out, err := cmd.CombinedOutput()
	if err == nil {
		return 0, string(out)
	}
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		return exitErr.ExitCode(), string(out)
	}
	t.Fatalf("run migrate-tenants helper: %v\n%s", err, out)
	return -1, string(out)
}

// buildSingleStoreFixture writes a 0.1 single-store layout (index.gob.collections.*)
// directly through the durable store package -- the only way to produce this
// layout now that canonical startup only ever creates the StoreSet one.
// Tenant "acme" gets one durable collection with two documents; tenant "beta"
// gets one ephemeral collection with a live (unpersisted) document, exercising
// the ADR 0009 reset-to-zero path migrate-tenants must report, not fail on.
func buildSingleStoreFixture(t *testing.T) (dataDir string, acmeIDs []uint64) {
	t.Helper()
	dataDir = filepath.Join(t.TempDir(), "state")
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		t.Fatal(err)
	}
	base := filepath.Join(dataDir, "index.gob.collections")
	store, err := vcollection.OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	durableSchema := vcollection.CollectionSchema{
		Name: "docs",
		Fields: []vcollection.VectorField{{
			Name: "embedding", Type: vcollection.VectorTypeDense, Dim: 2,
			Index: vcollection.IndexConfig{Type: vcollection.IndexTypeFLAT},
		}},
	}
	if _, err := store.Tenants().CreateCollection(ctx, "acme", durableSchema); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ {
		doc := &vcollection.Document{Vectors: map[string]vcollection.Vector{
			"embedding": {Dense: []float32{float32(i), 0}},
		}}
		if err := store.Tenants().AddDocument(ctx, "acme", "docs", doc); err != nil {
			t.Fatal(err)
		}
		acmeIDs = append(acmeIDs, doc.ID)
	}

	ephemeralSchema := durableSchema
	ephemeralSchema.Name = "frames"
	ephemeralSchema.Durability = vcollection.DurabilityEphemeral
	if _, err := store.Tenants().CreateCollection(ctx, "beta", ephemeralSchema); err != nil {
		t.Fatal(err)
	}
	if err := store.Tenants().AddDocument(ctx, "beta", "frames", &vcollection.Document{
		Vectors: map[string]vcollection.Vector{"embedding": {Dense: []float32{9, 9}}},
	}); err != nil {
		t.Fatal(err)
	}

	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	return dataDir, acmeIDs
}

// TestMigrateTenantsRefusalsOnOldAndEmptyLayouts covers the two argument-level
// refusals: nothing to migrate (no single-store state at all) and already
// migrated (running it twice).
func TestMigrateTenantsRefusalsOnOldAndEmptyLayouts(t *testing.T) {
	empty := t.TempDir()
	if rc, output := runMigrateTenantsProcess(t, empty); rc != 2 || !strings.Contains(output, "nothing to migrate") {
		t.Fatalf("empty data dir: rc=%d, want 2 with \"nothing to migrate\":\n%s", rc, output)
	}

	dataDir, _ := buildSingleStoreFixture(t)
	if rc, output := runMigrateTenantsProcess(t, dataDir); rc != 0 {
		t.Fatalf("first migrate-tenants rc=%d, want 0:\n%s", rc, output)
	}
	if rc, output := runMigrateTenantsProcess(t, dataDir); rc != 2 || !strings.Contains(output, "already migrated") {
		t.Fatalf("second migrate-tenants: rc=%d, want 2 with \"already migrated\":\n%s", rc, output)
	}
}

// TestCanonicalServeRefusesSingleStoreLayoutUntouched pins detectStoreLayout's
// serve-side half of this step: a server pointed at an un-migrated data
// directory refuses to start, names the fix, and never mutates the old files.
func TestCanonicalServeRefusesSingleStoreLayoutUntouched(t *testing.T) {
	dataDir, _ := buildSingleStoreFixture(t)
	base := filepath.Join(dataDir, "index.gob.collections")
	paths, err := filepath.Glob(base + ".*")
	if err != nil {
		t.Fatal(err)
	}
	beforeHash := make(map[string][32]byte, len(paths))
	for _, path := range paths {
		beforeHash[path] = testFileSHA256(t, path)
	}

	// Preflight runs before either listener binds, so fixed unused ports are fine.
	process := startCanonicalTestProcess(t, dataDir, "127.0.0.1:1", "127.0.0.1:2")
	defer process.stopIfRunning()
	select {
	case <-process.done:
		if err := process.waitError(); err == nil {
			t.Fatalf("serve on single-store layout exited successfully\n%s", process.output.String())
		}
	case <-time.After(10 * time.Second):
		_ = process.cmd.Process.Kill()
		_ = process.waitError()
		t.Fatalf("serve on single-store layout did not exit\n%s", process.output.String())
	}
	if !strings.Contains(process.output.String(), "migrate-tenants") {
		t.Fatalf("serve refusal did not point at migrate-tenants:\n%s", process.output.String())
	}
	for _, path := range paths {
		if got := testFileSHA256(t, path); got != beforeHash[path] {
			t.Fatalf("serve refusal changed single-store artifact %s", path)
		}
	}
}

// TestMigrateTenantsEndToEnd is the main path: migrate a two-tenant
// single-store layout (one collection ephemeral), then prove the result by
// starting a real server on it and using the gRPC client.
func TestMigrateTenantsEndToEnd(t *testing.T) {
	dataDir, acmeIDs := buildSingleStoreFixture(t)

	rc, output := runMigrateTenantsProcess(t, dataDir)
	if rc != 0 {
		t.Fatalf("migrate-tenants rc=%d, want 0:\n%s", rc, output)
	}
	if !strings.Contains(output, "acme") || !strings.Contains(output, "beta") {
		t.Fatalf("migration table is missing a tenant row:\n%s", output)
	}
	if !strings.Contains(output, "ephemeral") {
		t.Fatalf("migration output does not mention the reset ephemeral collection:\n%s", output)
	}

	httpAddress := unusedLoopbackAddress(t)
	grpcAddress := unusedLoopbackAddress(t)
	process := startCanonicalTestProcess(t, dataDir, httpAddress, grpcAddress)
	defer process.stopIfRunning()
	process.waitReady(t, httpAddress)

	connection, err := grpc.NewClient(grpcAddress, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	defer connection.Close()
	client := deepdatav3.NewDeepDataClient(connection)
	authed := func() (context.Context, context.CancelFunc) {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		return metadata.AppendToOutgoingContext(ctx, "authorization", "Bearer "+canonicalProcessAPIToken), cancel
	}

	for _, id := range acmeIDs {
		ctx, cancel := authed()
		_, err := client.GetDoc(ctx, &deepdatav3.GetDocRequest{TenantId: "acme", Collection: "docs", DocId: id})
		cancel()
		if err != nil {
			t.Fatalf("migrated document %d not found on the new layout: %v", id, err)
		}
	}

	ctx, cancel := authed()
	inserted, err := client.Insert(ctx, &deepdatav3.InsertRequest{
		TenantId: "acme", Collection: "docs",
		Vectors: map[string]*deepdatav3.VectorData{"embedding": denseProtoVector(5, 5)},
	})
	cancel()
	if err != nil {
		t.Fatal(err)
	}
	maxOld := acmeIDs[0]
	for _, id := range acmeIDs {
		if id > maxOld {
			maxOld = id
		}
	}
	if inserted.Id <= maxOld {
		t.Fatalf("post-migration insert id = %d, want greater than pre-migration max %d", inserted.Id, maxOld)
	}

	process.terminate(t)

	if rc, output := runMigrateTenantsProcess(t, dataDir); rc != 2 || !strings.Contains(output, "already migrated") {
		t.Fatalf("re-migrate after serving: rc=%d, want 2 with \"already migrated\":\n%s", rc, output)
	}
}

