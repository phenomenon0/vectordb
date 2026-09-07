package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"text/tabwriter"

	vcollection "github.com/phenomenon0/vectordb/internal/collection"
	"github.com/phenomenon0/vectordb/internal/logging"
	"github.com/phenomenon0/vectordb/internal/replication"
)

// promoteMarkerSuffix mirrors internal/replication/marker.go's unexported
// markerSuffix -- the same duplication tenant_transfer.go already carries in
// tenantTransferSuffixes, for the one place outside that package that needs
// to find or remove a tenant's marker file by name rather than through a
// DurableStore it has open.
const promoteMarkerSuffix = "-replica"

// runPromote is the `deepdata promote <data-dir> [--tenant id]` subcommand:
// an offline, operator-driven fencing of one or more standby tenants into
// leaders of their own history.
//
// It is the only thing that ever bumps a tenant's epoch. Promotion is a
// command, not a vote: the new epoch is what then fences the old leader out
// if it keeps writing past the point promotion happened at (Follower.Bind/
// Follow resync instead of applying a demoted leader's stream). The StoreID
// is left untouched -- it is the history's identity, and it is what lets a
// standby that never wrote past the promotion LSN keep following under the
// bumped epoch instead of being treated as a stranger.
//
// All-or-nothing across every candidate tenant: every candidate's store is
// opened first, and no epoch sidecar is written until every one of them
// opened. A directory a running serve or follower still holds fails loud
// before anything changes.
func runPromote(args []string, stdout, stderr io.Writer, logger *logging.Logger) int {
	fs := flag.NewFlagSet("promote", flag.ExitOnError)
	fs.SetOutput(stderr)
	tenantFlag := fs.String("tenant", "", "promote only this tenant")
	fs.Parse(args)
	if fs.NArg() != 1 {
		fmt.Fprintln(stderr, "promote: usage: deepdata promote <data-dir> [--tenant id]")
		return 2
	}

	// Same resolution migrate-tenants and export-tenant use, so this command
	// always agrees with `serve` on where a directory's tenants live.
	dataDir := resolveDataDir(os.Getenv("VECTORDB_BASE_DIR"), fs.Arg(0))
	tenantsDir := filepath.Join(dataDir, "index.gob") + ".tenants"

	var candidates []string
	if *tenantFlag != "" {
		if !vcollection.ValidTenantID(*tenantFlag) {
			fmt.Fprintf(stderr, "promote: %q is not a valid tenant id\n", *tenantFlag)
			return 2
		}
		if _, err := os.Stat(filepath.Join(tenantsDir, *tenantFlag+promoteMarkerSuffix)); os.IsNotExist(err) {
			fmt.Fprintf(stderr, "promote: tenant %q is not a standby under %s\n", *tenantFlag, tenantsDir)
			return 2
		} else if err != nil {
			logger.Error("promote: cannot inspect replica marker", "tenant", *tenantFlag, "error", err)
			return 1
		}
		candidates = []string{*tenantFlag}
	} else {
		entries, err := os.ReadDir(tenantsDir)
		if err != nil && !os.IsNotExist(err) {
			logger.Error("promote: cannot list tenant store directory", "path", tenantsDir, "error", err)
			return 1
		}
		for _, e := range entries {
			if e.IsDir() {
				continue
			}
			if id, ok := strings.CutSuffix(e.Name(), promoteMarkerSuffix); ok {
				candidates = append(candidates, id)
			}
		}
		sort.Strings(candidates)
	}
	if len(candidates) == 0 {
		fmt.Fprintf(stderr, "promote: nothing to promote: no standby tenants under %s\n", tenantsDir)
		return 2
	}

	type openTenant struct {
		id    string
		base  string
		store *vcollection.DurableStore
	}
	opened := make([]openTenant, 0, len(candidates))
	abortOpened := func() {
		for _, o := range opened {
			if err := o.store.Abort(); err != nil {
				logger.Error("promote: releasing tenant store lock", "tenant", o.id, "error", err)
			}
		}
	}

	// Phase 1: open every candidate's store. This takes the same per-tenant
	// flock a running `deepdata serve` or `deepdata replicate` holds, so one
	// of those already running against a candidate fails here, loudly,
	// before anything on disk changes.
	for _, id := range candidates {
		base := filepath.Join(tenantsDir, id)
		store, err := vcollection.OpenDurableStore(base, base)
		if err != nil {
			abortOpened()
			if errors.Is(err, vcollection.ErrCollectionStoreLocked) {
				fmt.Fprintf(stderr, "promote: tenant %q is held by a running serve or follower; stop it first\n", id)
			} else {
				fmt.Fprintf(stderr, "promote: cannot open tenant %q: %v\n", id, err)
			}
			return 1
		}
		opened = append(opened, openTenant{id: id, base: base, store: store})
	}

	// Phase 2: fence every candidate by writing its next epoch at its
	// current LSN. Markers are still untouched here -- a crash mid-loop
	// leaves every candidate looking like a standby, and re-running promote
	// just fences it again.
	type row struct {
		id    string
		lsn   uint64
		epoch uint64
	}
	rows := make([]row, 0, len(opened))
	for _, o := range opened {
		pos, err := o.store.JournalStatus()
		if err != nil {
			abortOpened()
			fmt.Fprintf(stderr, "promote: tenant %q: %v\n", o.id, err)
			return 1
		}
		local, err := replication.ReadEpoch(o.base)
		if err != nil {
			abortOpened()
			fmt.Fprintf(stderr, "promote: tenant %q: %v\n", o.id, err)
			return 1
		}
		next := replication.Epoch{Number: local.Number + 1, StartLSN: pos.LatestLSN}
		if err := replication.WriteEpoch(o.base, next); err != nil {
			abortOpened()
			fmt.Fprintf(stderr, "promote: tenant %q: %v\n", o.id, err)
			return 1
		}
		rows = append(rows, row{id: o.id, lsn: pos.LatestLSN, epoch: next.Number})
	}

	// Phase 3: every epoch sidecar landed. Only now flip each candidate from
	// standby to leader by dropping its marker (no checkpoint -- the store
	// is done being opened, not written to), and release its lock. Every
	// store is aborted regardless of whether its marker removal succeeds, so
	// a failure here never leaks a lock; a candidate whose marker survives
	// still looks like a standby and promote can simply be re-run on it.
	failed := false
	for _, o := range opened {
		if err := os.Remove(o.base + promoteMarkerSuffix); err != nil {
			fmt.Fprintf(stderr, "promote: tenant %q: remove replica marker: %v\n", o.id, err)
			failed = true
		}
		if err := o.store.Abort(); err != nil {
			logger.Error("promote: releasing tenant store lock", "tenant", o.id, "error", err)
		}
	}
	if failed {
		return 1
	}

	w := tabwriter.NewWriter(stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "tenant\tpromoted_at_lsn\tepoch")
	for _, r := range rows {
		fmt.Fprintf(w, "%s\t%d\t%d\n", r.id, r.lsn, r.epoch)
	}
	w.Flush()
	fmt.Fprintln(stdout, "next: start deepdata serve on this directory; point standbys at it")
	return 0
}
