package main

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"text/tabwriter"

	vcollection "github.com/phenomenon0/vectordb/internal/collection"
	"github.com/phenomenon0/vectordb/internal/logging"
)

// tenantTransferSuffixes mirrors the fixed suffix list
// internal/collection/store_set.go's StoreSet.DeleteTenant sweeps a tenant's
// files with (tenantArtifactSuffixes, unexported). Duplicated here rather
// than exported across the package boundary for two subcommands' temp-file
// cleanup; keep it in sync if that list ever changes.
var tenantTransferSuffixes = []string{
	".journal", ".journal.frozen", ".usage.json", ".snapshot", ".initialized", ".lock", "-replica",
}

// removeTenantArtifacts best-effort removes every file a store opened at
// base could have created, for cleaning up an abandoned temp store.
func removeTenantArtifacts(base string) {
	for _, suffix := range tenantTransferSuffixes {
		_ = os.Remove(base + suffix)
	}
}

func randomHex(n int) string {
	b := make([]byte, n)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

func printTenantTransferTable(w io.Writer, tenant string, infos []vcollection.CollectionInfo) {
	documents := 0
	for _, info := range infos {
		documents += info.DocCount
	}
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "tenant\tcollections\tdocuments")
	fmt.Fprintf(tw, "%s\t%d\t%d\n", tenant, len(infos), documents)
	tw.Flush()
}

// runExportTenant is `deepdata export-tenant <data-dir> <tenant> <file>`: an
// offline, read-only copy of one tenant's durable state into a single
// self-contained file, so import-tenant can hand it to another server. It
// never checkpoints or otherwise mutates the source tenant.
//
// Like migrate-tenants, it opens the tenant's store to export it, which takes
// the same per-tenant flock a running `deepdata serve` or `deepdata
// replicate` holds -- so one of those already running against this tenant
// fails this the same way it fails migrate-tenants.
func runExportTenant(args []string, stdout, stderr io.Writer, logger *logging.Logger) int {
	fs := flag.NewFlagSet("export-tenant", flag.ExitOnError)
	fs.SetOutput(stderr)
	fs.Parse(args)
	if fs.NArg() != 3 {
		fmt.Fprintln(stderr, "export-tenant: usage: deepdata export-tenant <data-dir> <tenant> <file>")
		return 2
	}
	dataDir := resolveDataDir(os.Getenv("VECTORDB_BASE_DIR"), fs.Arg(0))
	tenant := fs.Arg(1)
	file := fs.Arg(2)
	tenantsDir := filepath.Join(dataDir, "index.gob") + ".tenants"

	if !vcollection.ValidTenantID(tenant) {
		fmt.Fprintf(stderr, "export-tenant: %q is not a valid tenant id\n", tenant)
		return 2
	}
	base := filepath.Join(tenantsDir, tenant)
	if _, err := os.Stat(base + ".initialized"); os.IsNotExist(err) {
		fmt.Fprintf(stderr, "export-tenant: tenant %q does not exist under %s\n", tenant, tenantsDir)
		return 2
	} else if err != nil {
		logger.Error("export-tenant: cannot inspect tenant marker", "path", base+".initialized", "error", err)
		return 1
	}

	store, err := vcollection.OpenDurableStore(base, base)
	if err != nil {
		fmt.Fprintf(stderr, "export-tenant: cannot open %s (is a server already running against it?): %v\n", base, err)
		return 1
	}
	aborted := false
	abort := func() {
		if aborted {
			return
		}
		aborted = true
		if err := store.Abort(); err != nil {
			logger.Error("export-tenant: releasing tenant store lock", "error", err)
		}
	}
	defer abort()

	infos := store.Tenants().ListCollectionInfos(tenant)

	tmpBase := filepath.Join(filepath.Dir(file), ".export-"+tenant+"-"+randomHex(8))
	if err := vcollection.ExportTenantSnapshot(store, tenant, tmpBase); err != nil {
		removeTenantArtifacts(tmpBase)
		fmt.Fprintf(stderr, "export-tenant: export snapshot: %v\n", err)
		return 1
	}
	if err := os.Rename(tmpBase+".snapshot", file); err != nil {
		removeTenantArtifacts(tmpBase)
		fmt.Fprintf(stderr, "export-tenant: write %s: %v\n", file, err)
		return 1
	}
	os.Remove(tmpBase + ".initialized")
	abort() // release the source lock now; the copy is already safely in place.

	printTenantTransferTable(stdout, tenant, infos)
	fmt.Fprintf(stdout, "wrote %s\n", file)
	fmt.Fprintln(stdout, "note: ephemeral collections' documents are memory-only and are not in this file")
	return 0
}

// runImportTenant is `deepdata import-tenant <data-dir> <tenant> <file>`, the
// other half of export-tenant. <file> is validated as a real tenant snapshot
// (checksum, header) by opening it before anything under <data-dir> is
// created, and the imported tenant always gets a fresh StoreID: two servers
// that both imported the same file must not look like the same store to the
// node transport.
func runImportTenant(args []string, stdout, stderr io.Writer, logger *logging.Logger) int {
	fs := flag.NewFlagSet("import-tenant", flag.ExitOnError)
	fs.SetOutput(stderr)
	fs.Parse(args)
	if fs.NArg() != 3 {
		fmt.Fprintln(stderr, "import-tenant: usage: deepdata import-tenant <data-dir> <tenant> <file>")
		return 2
	}
	dataDir := resolveDataDir(os.Getenv("VECTORDB_BASE_DIR"), fs.Arg(0))
	tenant := fs.Arg(1)
	file := fs.Arg(2)
	tenantsDir := filepath.Join(dataDir, "index.gob") + ".tenants"

	if !vcollection.ValidTenantID(tenant) {
		fmt.Fprintf(stderr, "import-tenant: %q is not a valid tenant id\n", tenant)
		return 2
	}
	base := filepath.Join(tenantsDir, tenant)
	if _, err := os.Stat(base + ".initialized"); err == nil {
		fmt.Fprintf(stderr, "import-tenant: tenant %q already exists here; delete it or choose another id\n", tenant)
		return 2
	} else if !os.IsNotExist(err) {
		logger.Error("import-tenant: cannot inspect tenant marker", "path", base+".initialized", "error", err)
		return 1
	}
	src, err := os.Open(file)
	if err != nil {
		fmt.Fprintf(stderr, "import-tenant: %v\n", err)
		return 2
	}
	defer src.Close()

	if err := os.MkdirAll(tenantsDir, 0o750); err != nil {
		fmt.Fprintf(stderr, "import-tenant: create %s: %v\n", tenantsDir, err)
		return 1
	}

	tmpBase := filepath.Join(tenantsDir, ".import-"+tenant+"-"+randomHex(8))
	dst, err := os.OpenFile(tmpBase+".snapshot", os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		fmt.Fprintf(stderr, "import-tenant: stage %s: %v\n", file, err)
		return 1
	}
	_, copyErr := io.Copy(dst, src)
	closeErr := dst.Close()
	if err := errors.Join(copyErr, closeErr); err != nil {
		removeTenantArtifacts(tmpBase)
		fmt.Fprintf(stderr, "import-tenant: stage %s: %v\n", file, err)
		return 1
	}

	// Opening tmpBase validates the copy is a real, uncorrupted snapshot
	// (checksum, header) before base ever exists.
	tmpStore, err := vcollection.OpenDurableStore(tmpBase, tmpBase)
	if err != nil {
		removeTenantArtifacts(tmpBase)
		fmt.Fprintf(stderr, "import-tenant: %s is not a valid tenant snapshot: %v\n", file, err)
		return 1
	}
	tmpAborted := false
	abortTmp := func() {
		if tmpAborted {
			return
		}
		tmpAborted = true
		if err := tmpStore.Abort(); err != nil {
			logger.Error("import-tenant: releasing staged store lock", "error", err)
		}
	}
	defer func() {
		abortTmp()
		removeTenantArtifacts(tmpBase)
	}()

	// base is only ever created here, by minting a fresh StoreID for it --
	// never by copying tmpBase's files directly.
	if err := vcollection.ExportTenantSnapshot(tmpStore, tenant, base); err != nil {
		fmt.Fprintf(stderr, "import-tenant: %s does not contain tenant %q: %v\n", file, tenant, err)
		return 1
	}
	abortTmp()
	removeTenantArtifacts(tmpBase)

	store, err := vcollection.OpenDurableStore(base, base)
	if err != nil {
		fmt.Fprintf(stderr, "import-tenant: verify %s: %v\n", base, err)
		return 1
	}
	infos := store.Tenants().ListCollectionInfos(tenant)
	if err := store.Close(); err != nil {
		fmt.Fprintf(stderr, "import-tenant: close %s: %v\n", base, err)
		return 1
	}

	printTenantTransferTable(stdout, tenant, infos)
	fmt.Fprintf(stdout, "imported into %s\n", base)
	return 0
}
