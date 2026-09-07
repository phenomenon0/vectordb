package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"text/tabwriter"

	vcollection "github.com/phenomenon0/vectordb/internal/collection"
	"github.com/phenomenon0/vectordb/internal/logging"
)

// runMigrateTenants is the `deepdata migrate-tenants <data-dir>` subcommand:
// a one-time offline rewrite of the 0.1 single-store layout
// (index.gob.collections.*) into the canonical per-tenant StoreSet layout
// (index.gob.tenants/<id>.*). detectStoreLayout is what tells an operator
// they need this; this command is the only thing that runs it.
//
// It never touches the old store beyond opening it read-write to export each
// tenant and then aborting (no checkpoint, no journal deletion): the old
// prefix is the rollback path, untouched on both success and failure.
func runMigrateTenants(args []string, stdout, stderr io.Writer, logger *logging.Logger) int {
	fs := flag.NewFlagSet("migrate-tenants", flag.ExitOnError)
	fs.SetOutput(stderr)
	fs.Parse(args)
	if fs.NArg() != 1 {
		fmt.Fprintln(stderr, "migrate-tenants: usage: deepdata migrate-tenants <data-dir>")
		return 2
	}

	// Same resolution loadServerConfig uses to turn a data-dir into
	// cfg.IndexPath (config.go), so this command and `serve` always agree on
	// where a directory's store files live.
	dataDir := resolveDataDir(os.Getenv("VECTORDB_BASE_DIR"), fs.Arg(0))
	indexPath := filepath.Join(dataDir, "index.gob")
	basePath := indexPath + ".collections"
	tenantsDir := indexPath + ".tenants"

	if info, err := os.Stat(tenantsDir); err == nil && info.IsDir() {
		fmt.Fprintf(stderr, "migrate-tenants: %s already migrated\n", dataDir)
		return 2
	} else if err != nil && !os.IsNotExist(err) {
		logger.Error("migrate-tenants: cannot inspect tenant store directory", "path", tenantsDir, "error", err)
		return 1
	}

	legacyV2, err := legacyV2CollectionArtifacts(basePath)
	if err != nil {
		logger.Error("migrate-tenants: cannot inspect legacy V2 artifacts", "error", err)
		return 1
	}
	if len(legacyV2) > 0 {
		fmt.Fprintf(stderr, "migrate-tenants: raw legacy V2 collection artifacts need their own offline migration first: %v\n", legacyV2)
		return 2
	}

	if _, err := os.Stat(basePath + ".initialized"); os.IsNotExist(err) {
		fmt.Fprintf(stderr, "migrate-tenants: nothing to migrate: %s has no single-store collection state\n", dataDir)
		return 2
	} else if err != nil {
		logger.Error("migrate-tenants: cannot inspect single-store marker", "path", basePath+".initialized", "error", err)
		return 1
	}

	// OpenDurableStore takes the store's flock, so a `deepdata serve` (or
	// another migrate-tenants) already holding it fails this loudly rather
	// than racing it.
	store, err := vcollection.OpenDurableStore(basePath, basePath)
	if err != nil {
		fmt.Fprintf(stderr, "migrate-tenants: cannot open %s (is a server already running against it?): %v\n", basePath, err)
		return 1
	}
	aborted := false
	abort := func() {
		if aborted {
			return
		}
		aborted = true
		if err := store.Abort(); err != nil {
			logger.Error("migrate-tenants: releasing old store lock", "error", err)
		}
	}
	defer abort()

	if n, err := store.LegacyCollectionCount(); err != nil {
		fmt.Fprintf(stderr, "migrate-tenants: %v\n", err)
		return 1
	} else if n != 0 {
		fmt.Fprintf(stderr, "migrate-tenants: %s has %d pre-tenant collection(s) that need their own migration first\n", basePath, n)
		return 1
	}

	tenantIDs := store.Tenants().ListTenants()
	for _, id := range tenantIDs {
		if !vcollection.ValidTenantID(id) {
			fmt.Fprintf(stderr, "migrate-tenants: tenant %q is not a valid tenant ID; refusing before writing anything\n", id)
			return 1
		}
	}

	type tenantExport struct {
		id       string
		oldInfos []vcollection.CollectionInfo
	}
	exports := make([]tenantExport, 0, len(tenantIDs))
	for _, id := range tenantIDs {
		exports = append(exports, tenantExport{id: id, oldInfos: store.Tenants().ListCollectionInfos(id)})
	}

	if err := os.MkdirAll(tenantsDir, 0o750); err != nil {
		fmt.Fprintf(stderr, "migrate-tenants: create %s: %v\n", tenantsDir, err)
		return 1
	}
	// tenantsDir did not exist a moment ago (checked above), so this run
	// always owns it: any failure from here on cleans it back up.
	rollback := func() { _ = os.RemoveAll(tenantsDir) }

	for _, e := range exports {
		if err := vcollection.ExportTenantSnapshot(store, e.id, filepath.Join(tenantsDir, e.id)); err != nil {
			fmt.Fprintf(stderr, "migrate-tenants: tenant %q: export snapshot: %v\n", e.id, err)
			rollback()
			return 1
		}
	}
	abort() // release the old store's lock now; never checkpoint or touch it further.

	cfg, _ := loadServerConfig(nil, os.Getenv)
	newSet, err := vcollection.OpenStoreSet(tenantsDir, vcollection.StoreLimits{
		MaxTenants:           cfg.Limits.MaxTenants,
		MaxCollections:       cfg.Limits.MaxCollections,
		MaxTenantDocuments:   cfg.Limits.MaxTenantDocuments,
		MaxTenantBytes:       cfg.Limits.MaxTenantBytes,
		MaxTenantCollections: cfg.Limits.MaxTenantCollections,
	})
	if err != nil {
		fmt.Fprintf(stderr, "migrate-tenants: reopen migrated store: %v\n", err)
		rollback()
		return 1
	}

	type row struct {
		id                     string
		collections, documents int
		ephemeral              int
	}
	rows := make([]row, 0, len(exports))
	fail := func(format string, args ...any) int {
		fmt.Fprintf(stderr, "migrate-tenants: "+format+"\n", args...)
		_ = newSet.Close()
		rollback()
		return 1
	}
	for _, e := range exports {
		newInfos, err := newSet.ListCollectionInfosChecked(e.id)
		if err != nil {
			return fail("tenant %q: verify: %v", e.id, err)
		}
		if len(newInfos) != len(e.oldInfos) {
			return fail("tenant %q: has %d collections after migration, want %d", e.id, len(newInfos), len(e.oldInfos))
		}
		newByName := make(map[string]vcollection.CollectionInfo, len(newInfos))
		for _, info := range newInfos {
			newByName[info.Name] = info
		}
		r := row{id: e.id}
		for _, old := range e.oldInfos {
			neu, ok := newByName[old.Name]
			if !ok {
				return fail("tenant %q: collection %q missing after migration", e.id, old.Name)
			}
			// Durability class E (ADR 0009): an ephemeral collection's
			// documents are memory only and were never journaled, so the
			// export snapshot always carries zero for it regardless of how
			// many were live when this command ran.
			if old.Durability == vcollection.DurabilityEphemeral {
				r.ephemeral++
				if neu.DocCount != 0 {
					return fail("tenant %q: ephemeral collection %q kept %d docs, want 0", e.id, old.Name, neu.DocCount)
				}
			} else if neu.DocCount != old.DocCount {
				return fail("tenant %q: collection %q has %d docs, want %d", e.id, old.Name, neu.DocCount, old.DocCount)
			}
			r.documents += neu.DocCount
		}
		r.collections = len(newInfos)
		rows = append(rows, r)
	}

	if err := newSet.Close(); err != nil {
		fmt.Fprintf(stderr, "migrate-tenants: close migrated store: %v\n", err)
		rollback()
		return 1
	}

	w := tabwriter.NewWriter(stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "tenant\tcollections\tdocuments")
	for _, r := range rows {
		note := ""
		if r.ephemeral > 0 {
			note = fmt.Sprintf("\t(%d ephemeral collection(s) reset to 0 docs)", r.ephemeral)
		}
		fmt.Fprintf(w, "%s\t%d\t%d%s\n", r.id, r.collections, r.documents, note)
	}
	w.Flush()
	fmt.Fprintln(stdout, "note: usage stats are not migrated; each tenant's quota usage re-accretes from zero under the new layout")
	fmt.Fprintf(stdout, "next: start deepdata serve; %s is untouched, delete it once you are satisfied\n", basePath)
	return 0
}
