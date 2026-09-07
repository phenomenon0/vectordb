package main

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	vcollection "github.com/phenomenon0/vectordb/internal/collection"
)

const (
	tenantTransferHelperEnv = "DEEPDATA_TENANT_TRANSFER_HELPER"
	tenantTransferArgsEnv   = "DEEPDATA_TENANT_TRANSFER_ARGS"
)

// TestTenantTransferProcessHelper is the re-exec target runTenantTransferProcess
// launches: same trick as TestMigrateTenantsProcessHelper, parameterized over
// which of the two subcommands to run.
func TestTenantTransferProcessHelper(t *testing.T) {
	subcommand := os.Getenv(tenantTransferHelperEnv)
	if subcommand == "" {
		return
	}
	os.Args = append([]string{"deepdata", subcommand}, strings.Split(os.Getenv(tenantTransferArgsEnv), "\x1f")...)
	main()
}

// runTenantTransferProcess runs `deepdata <subcommand> args...` as a real
// subprocess and returns its exit code and combined output.
func runTenantTransferProcess(t *testing.T, subcommand string, args ...string) (rc int, output string) {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^TestTenantTransferProcessHelper$")
	cmd.Env = processEnv(map[string]string{
		tenantTransferHelperEnv: subcommand,
		tenantTransferArgsEnv:   strings.Join(args, "\x1f"),
	})
	out, err := cmd.CombinedOutput()
	if err == nil {
		return 0, string(out)
	}
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		return exitErr.ExitCode(), string(out)
	}
	t.Fatalf("run %s helper: %v\n%s", subcommand, err, out)
	return -1, string(out)
}

// buildTenantTransferFixture writes a two-tenant StoreSet layout directly
// through the durable store package (the layout export-tenant/import-tenant
// operate on, unlike buildSingleStoreFixture's pre-migration one). acme gets
// two documents; globex gets one, and exists only to prove exporting acme
// never touches it.
func buildTenantTransferFixture(t *testing.T) (dataDir string, acmeIDs, globexIDs []uint64) {
	t.Helper()
	dataDir = filepath.Join(t.TempDir(), "state")
	tenantsDir := filepath.Join(dataDir, "index.gob.tenants")
	set, err := vcollection.OpenStoreSet(tenantsDir, vcollection.StoreLimits{MaxTenants: 8, MaxCollections: 8})
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	schema := vcollection.CollectionSchema{
		Name: "docs",
		Fields: []vcollection.VectorField{{
			Name: "dense", Type: vcollection.VectorTypeDense, Dim: 2,
			Index: vcollection.IndexConfig{Type: vcollection.IndexTypeFLAT},
		}},
	}
	if _, err := set.CreateCollection(ctx, "acme", schema); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ {
		doc := &vcollection.Document{Vectors: map[string]vcollection.Vector{"dense": {Dense: []float32{float32(i), 0}}}}
		if err := set.AddDocument(ctx, "acme", "docs", doc); err != nil {
			t.Fatal(err)
		}
		acmeIDs = append(acmeIDs, doc.ID)
	}
	if _, err := set.CreateCollection(ctx, "globex", schema); err != nil {
		t.Fatal(err)
	}
	doc := &vcollection.Document{Vectors: map[string]vcollection.Vector{"dense": {Dense: []float32{9, 9}}}}
	if err := set.AddDocument(ctx, "globex", "docs", doc); err != nil {
		t.Fatal(err)
	}
	globexIDs = append(globexIDs, doc.ID)
	if err := set.Close(); err != nil {
		t.Fatal(err)
	}
	return dataDir, acmeIDs, globexIDs
}

// tenantStoreID reads the store ID out of a tenant's initialization marker --
// the only place it is visible outside the collection package -- so a test
// can prove import-tenant minted a fresh one rather than copying the source's.
func tenantStoreID(t *testing.T, base string) string {
	t.Helper()
	data, err := os.ReadFile(base + ".initialized")
	if err != nil {
		t.Fatal(err)
	}
	var marker struct {
		StoreID string `json:"store_id"`
	}
	if err := json.Unmarshal(data, &marker); err != nil {
		t.Fatal(err)
	}
	if marker.StoreID == "" {
		t.Fatalf("marker at %s.initialized has no store_id", base)
	}
	return marker.StoreID
}

// Exporting one tenant must be read-only: every byte of both tenants' files
// must be exactly what it was before, and the leftover tenant (globex) must
// not even be inspected.
func TestExportTenantLeavesSourceUnchanged(t *testing.T) {
	dataDir, _, _ := buildTenantTransferFixture(t)
	tenantsDir := filepath.Join(dataDir, "index.gob.tenants")
	before := map[string][32]byte{}
	for _, glob := range []string{"acme.*", "globex.*"} {
		matches, err := filepath.Glob(filepath.Join(tenantsDir, glob))
		if err != nil {
			t.Fatal(err)
		}
		for _, path := range matches {
			before[path] = testFileSHA256(t, path)
		}
	}
	if len(before) == 0 {
		t.Fatal("fixture produced no tenant files to compare")
	}

	outDir := t.TempDir()
	acmeFile := filepath.Join(outDir, "acme.transfer")
	if rc, output := runTenantTransferProcess(t, "export-tenant", dataDir, "acme", acmeFile); rc != 0 {
		t.Fatalf("export-tenant acme rc=%d, want 0:\n%s", rc, output)
	}
	if _, err := os.Stat(acmeFile); err != nil {
		t.Fatalf("exported file missing: %v", err)
	}

	for path, hash := range before {
		if got := testFileSHA256(t, path); got != hash {
			t.Fatalf("export-tenant changed source artifact %s", path)
		}
	}
}

// A tenant open in-process (a running server or another migration/transfer
// command holding its flock) must fail export-tenant with the same hint
// migrate-tenants gives, not a confusing lock error.
func TestExportTenantRefusesWhileTenantIsOpen(t *testing.T) {
	dataDir, _, _ := buildTenantTransferFixture(t)
	tenantsDir := filepath.Join(dataDir, "index.gob.tenants")
	base := filepath.Join(tenantsDir, "acme")
	store, err := vcollection.OpenDurableStore(base, base)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Abort()

	rc, output := runTenantTransferProcess(t, "export-tenant", dataDir, "acme", filepath.Join(t.TempDir(), "acme.transfer"))
	if rc != 1 {
		t.Fatalf("export-tenant while open rc=%d, want 1:\n%s", rc, output)
	}
	if !strings.Contains(output, "is a server already running against it") {
		t.Fatalf("export-tenant refusal did not name the running-server hint:\n%s", output)
	}
}

// The main path: export acme, import it into a brand new data directory, and
// prove the result by serving it -- same document IDs, a fresh StoreID, and
// an insert that continues above the imported next_id.
func TestImportTenantIntoEmptyDirServes(t *testing.T) {
	sourceDir, acmeIDs, _ := buildTenantTransferFixture(t)
	sourceBase := filepath.Join(sourceDir, "index.gob.tenants", "acme")
	sourceStoreID := tenantStoreID(t, sourceBase)

	exportFile := filepath.Join(t.TempDir(), "acme.transfer")
	if rc, output := runTenantTransferProcess(t, "export-tenant", sourceDir, "acme", exportFile); rc != 0 {
		t.Fatalf("export-tenant rc=%d, want 0:\n%s", rc, output)
	}

	destDir := filepath.Join(t.TempDir(), "state")
	rc, output := runTenantTransferProcess(t, "import-tenant", destDir, "acme", exportFile)
	if rc != 0 {
		t.Fatalf("import-tenant rc=%d, want 0:\n%s", rc, output)
	}
	if !strings.Contains(output, "acme") {
		t.Fatalf("import-tenant output is missing the tenant row:\n%s", output)
	}

	destBase := filepath.Join(destDir, "index.gob.tenants", "acme")
	if destStoreID := tenantStoreID(t, destBase); destStoreID == sourceStoreID {
		t.Fatalf("imported tenant kept the source's StoreID %s; want a fresh one", destStoreID)
	}

	handler := newCanonicalSurfaceTestHandlerAt(t, filepath.Join(destDir, "index.gob"))
	for _, id := range acmeIDs {
		resp := canonicalCall(t, handler, http.MethodGet, "/v3/tenants/acme/collections/docs/docs/"+strconv.FormatUint(id, 10), "")
		if resp.Code != http.StatusOK {
			t.Fatalf("imported document %d not found: %d %s", id, resp.Code, resp.Body.String())
		}
	}

	resp := canonicalCall(t, handler, http.MethodPost, "/v3/tenants/acme/collections/docs/docs", `{"vectors":{"dense":[5,5]}}`)
	if resp.Code != http.StatusOK {
		t.Fatalf("insert after import = %d, want 200: %s", resp.Code, resp.Body.String())
	}
	var inserted struct {
		ID uint64 `json:"id"`
	}
	if err := json.Unmarshal(resp.Body.Bytes(), &inserted); err != nil {
		t.Fatal(err)
	}
	maxOld := acmeIDs[0]
	for _, id := range acmeIDs {
		if id > maxOld {
			maxOld = id
		}
	}
	if inserted.ID <= maxOld {
		t.Fatalf("post-import insert id = %d, want greater than pre-import max %d", inserted.ID, maxOld)
	}
}

// Importing onto a tenant id that already exists must refuse before touching
// anything, with rc 2 (an argument error, not a runtime failure).
func TestImportTenantOntoExistingTenantRefuses(t *testing.T) {
	dataDir, _, _ := buildTenantTransferFixture(t)
	tenantsDir := filepath.Join(dataDir, "index.gob.tenants")
	exportFile := filepath.Join(t.TempDir(), "acme.transfer")
	if rc, output := runTenantTransferProcess(t, "export-tenant", dataDir, "acme", exportFile); rc != 0 {
		t.Fatalf("export-tenant rc=%d, want 0:\n%s", rc, output)
	}

	before := map[string][32]byte{}
	matches, err := filepath.Glob(filepath.Join(tenantsDir, "acme.*"))
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range matches {
		before[path] = testFileSHA256(t, path)
	}

	rc, output := runTenantTransferProcess(t, "import-tenant", dataDir, "acme", exportFile)
	if rc != 2 {
		t.Fatalf("import-tenant onto existing tenant rc=%d, want 2:\n%s", rc, output)
	}
	if !strings.Contains(output, "already exists") {
		t.Fatalf("import-tenant refusal did not say the tenant already exists:\n%s", output)
	}

	after, err := filepath.Glob(filepath.Join(tenantsDir, "acme.*"))
	if err != nil {
		t.Fatal(err)
	}
	if len(after) != len(before) {
		t.Fatalf("import-tenant onto an existing tenant changed its file set: before=%v after=%v", matches, after)
	}
	for path, hash := range before {
		if got := testFileSHA256(t, path); got != hash {
			t.Fatalf("import-tenant onto an existing tenant modified %s", path)
		}
	}
	if leftover, _ := filepath.Glob(filepath.Join(tenantsDir, ".import-*")); len(leftover) != 0 {
		t.Fatalf("import-tenant left temp files behind: %v", leftover)
	}
}

// A truncated (corrupt) snapshot file must fail the checksum/header
// validation inside OpenDurableStore, not silently mint a broken tenant, and
// must leave no trace in the target tenant directory.
func TestImportTenantTruncatedFileFails(t *testing.T) {
	sourceDir, _, _ := buildTenantTransferFixture(t)
	exportFile := filepath.Join(t.TempDir(), "acme.transfer")
	if rc, output := runTenantTransferProcess(t, "export-tenant", sourceDir, "acme", exportFile); rc != 0 {
		t.Fatalf("export-tenant rc=%d, want 0:\n%s", rc, output)
	}
	full, err := os.ReadFile(exportFile)
	if err != nil {
		t.Fatal(err)
	}
	if len(full) < 2 {
		t.Fatal("exported file is too small to truncate meaningfully")
	}
	truncatedFile := filepath.Join(t.TempDir(), "acme.truncated")
	if err := os.WriteFile(truncatedFile, full[:len(full)/2], 0o600); err != nil {
		t.Fatal(err)
	}

	destDir := filepath.Join(t.TempDir(), "state")
	rc, output := runTenantTransferProcess(t, "import-tenant", destDir, "acme", truncatedFile)
	if rc != 1 {
		t.Fatalf("import-tenant of a truncated file rc=%d, want 1:\n%s", rc, output)
	}

	tenantsDir := filepath.Join(destDir, "index.gob.tenants")
	leftover, err := filepath.Glob(filepath.Join(tenantsDir, "*"))
	if err != nil {
		t.Fatal(err)
	}
	if len(leftover) != 0 {
		t.Fatalf("import-tenant of a truncated file left files behind: %v", leftover)
	}
}
