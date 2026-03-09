// vectordb-cli is a command-line tool for managing a VectorDB server.
//
// Usage:
//
//	vectordb-cli <command> [flags]
//
// Commands:
//
//	health        Check server health
//	insert        Insert a document
//	query         Search for similar documents
//	delete        Delete a document by ID
//	collections   List collections
//	stats         Show server statistics
//	import           Bulk import from JSONL file
//	import-obsidian  Import Obsidian vault with graph metadata
//	export           Export collection to JSONL file
//	compact       Trigger index compaction
//	gentoken      Generate a JWT token
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"regexp"
	"strings"
	"syscall"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/phenomenon0/vectordb/client"
)

var version = "dev"

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	cmd := os.Args[1]
	// Strip the subcommand from args so flag.Parse works on the rest
	os.Args = append(os.Args[:1], os.Args[2:]...)

	switch cmd {
	case "health":
		cmdHealth()
	case "insert":
		cmdInsert()
	case "query", "search":
		cmdQuery()
	case "delete":
		cmdDelete()
	case "collections", "list":
		cmdCollections()
	case "stats":
		cmdStats()
	case "import":
		cmdImport()
	case "import-obsidian":
		cmdImportObsidian()
	case "export":
		cmdExport()
	case "compact":
		cmdCompact()
	case "gentoken":
		cmdGentoken()
	case "version":
		fmt.Printf("vectordb-cli %s\n", version)
	case "help", "-h", "--help":
		printUsage()
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n\n", cmd)
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Fprintf(os.Stderr, `vectordb-cli — VectorDB command-line tool

Usage: vectordb-cli <command> [flags]

Commands:
  health        Check server health and stats
  insert        Insert a document (--doc, --id, --collection, --meta)
  query         Search for similar documents (--query, --top-k, --collection)
  delete        Delete a document by ID (--id)
  collections   List all collections with stats
  stats         Show detailed server statistics
  import           Bulk import from JSONL file (--file, --collection)
  import-obsidian  Import Obsidian vault or markdown folder (--vault)
  export           Export collection to JSONL (--collection, --output)
  compact       Trigger index compaction
  gentoken      Generate a JWT authentication token
  version       Print version

Global flags (set via environment):
  VECTORDB_URL      Server URL (default: http://localhost:8080)
  VECTORDB_TOKEN    Bearer token for authentication

Examples:
  vectordb-cli health
  vectordb-cli insert --doc "Hello world" --collection docs
  vectordb-cli query --query "search term" --top-k 5
  vectordb-cli import --file data.jsonl --collection my-docs
  vectordb-cli export --collection docs --output backup.jsonl
`)
}

func newClient() *client.Client {
	url := os.Getenv("VECTORDB_URL")
	if url == "" {
		url = "http://localhost:8080"
	}
	var opts []client.Option
	if token := os.Getenv("VECTORDB_TOKEN"); token != "" {
		opts = append(opts, client.WithToken(token))
	}
	return client.New(url, opts...)
}

func ctx() context.Context {
	c, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	// cancel is intentionally not deferred — the process exits after each command.
	// Suppress the linter by using cancel in a runtime.SetFinalizer-like pattern.
	go func() { <-c.Done(); cancel() }()
	return c
}

func fatal(msg string, err error) {
	var apiErr *client.APIError
	if errors.As(err, &apiErr) {
		fmt.Fprintf(os.Stderr, "error: %s (HTTP %d)\n", apiErr.Message, apiErr.StatusCode)
	} else {
		fmt.Fprintf(os.Stderr, "error: %s: %v\n", msg, err)
	}
	os.Exit(1)
}

// --- Commands ---

func cmdHealth() {
	c := newClient()
	resp, err := c.Health(ctx())
	if err != nil {
		fatal("health check failed", err)
	}
	out, _ := json.MarshalIndent(resp, "", "  ")
	fmt.Println(string(out))
}

func cmdInsert() {
	fs := flag.NewFlagSet("insert", flag.ExitOnError)
	doc := fs.String("doc", "", "Document text to insert (required)")
	id := fs.String("id", "", "Document ID (auto-generated if empty)")
	collection := fs.String("collection", "", "Collection name")
	metaStr := fs.String("meta", "", "Metadata as key=value pairs, comma-separated")
	upsert := fs.Bool("upsert", false, "Upsert mode (update if ID exists)")
	fs.Parse(os.Args[1:])

	if *doc == "" {
		fmt.Fprintln(os.Stderr, "error: --doc is required")
		fs.Usage()
		os.Exit(1)
	}

	meta := parseMeta(*metaStr)

	c := newClient()
	resp, err := c.Insert(ctx(), client.InsertRequest{
		Doc:        *doc,
		ID:         *id,
		Collection: *collection,
		Meta:       meta,
		Upsert:     *upsert,
	})
	if err != nil {
		fatal("insert failed", err)
	}
	fmt.Printf("inserted: %s\n", resp.ID)
}

func cmdQuery() {
	fs := flag.NewFlagSet("query", flag.ExitOnError)
	query := fs.String("query", "", "Search query text (required)")
	topK := fs.Int("top-k", 5, "Number of results")
	collection := fs.String("collection", "", "Collection to search")
	includeMeta := fs.Bool("meta", false, "Include metadata in results")
	mode := fs.String("mode", "ann", "Search mode: ann or scan")
	scoreMode := fs.String("score-mode", "vector", "Score mode: vector, hybrid, or lexical")
	outputJSON := fs.Bool("json", false, "Output raw JSON")
	fs.Parse(os.Args[1:])

	if *query == "" {
		fmt.Fprintln(os.Stderr, "error: --query is required")
		fs.Usage()
		os.Exit(1)
	}

	c := newClient()
	resp, err := c.Query(ctx(), client.QueryRequest{
		Query:       *query,
		TopK:        *topK,
		Collection:  *collection,
		IncludeMeta: *includeMeta,
		Mode:        *mode,
		ScoreMode:   *scoreMode,
	})
	if err != nil {
		fatal("query failed", err)
	}

	if *outputJSON {
		out, _ := json.MarshalIndent(resp, "", "  ")
		fmt.Println(string(out))
		return
	}

	fmt.Printf("Results (%d):\n", len(resp.IDs))
	for i, id := range resp.IDs {
		score := float32(0)
		if i < len(resp.Scores) {
			score = resp.Scores[i]
		}
		doc := ""
		if i < len(resp.Docs) {
			doc = resp.Docs[i]
			if len(doc) > 120 {
				doc = doc[:120] + "..."
			}
		}
		fmt.Printf("  %d. [%.4f] %s\n     %s\n", i+1, score, id, doc)
		if *includeMeta && i < len(resp.Meta) && len(resp.Meta[i]) > 0 {
			for k, v := range resp.Meta[i] {
				fmt.Printf("     %s: %s\n", k, v)
			}
		}
	}
}

func cmdDelete() {
	fs := flag.NewFlagSet("delete", flag.ExitOnError)
	id := fs.String("id", "", "Document ID to delete (required)")
	fs.Parse(os.Args[1:])

	if *id == "" {
		fmt.Fprintln(os.Stderr, "error: --id is required")
		fs.Usage()
		os.Exit(1)
	}

	c := newClient()
	resp, err := c.Delete(ctx(), client.DeleteRequest{ID: *id})
	if err != nil {
		fatal("delete failed", err)
	}
	fmt.Printf("deleted: %s\n", resp.Deleted)
}

func cmdCollections() {
	c := newClient()
	resp, err := c.ListCollections(ctx())
	if err != nil {
		// Fall back to health endpoint if admin endpoint is unavailable
		var apiErr *client.APIError
		if errors.As(err, &apiErr) && (apiErr.StatusCode == 403 || apiErr.StatusCode == 404) {
			health, herr := c.Health(ctx())
			if herr != nil {
				fatal("failed to get collections", herr)
			}
			fmt.Printf("Total vectors: %d (active: %d, deleted: %d)\n", health.Total, health.Active, health.Deleted)
			fmt.Printf("HNSW nodes:    %d\n", health.HNSWIDs)
			return
		}
		fatal("failed to get collections", err)
	}
	fmt.Printf("Collections (%d):\n", resp.Count)
	for _, coll := range resp.Collections {
		name, _ := coll["name"].(string)
		if name == "" {
			name = fmt.Sprintf("%v", coll)
		}
		fmt.Printf("  - %s\n", name)
	}
}

func cmdStats() {
	c := newClient()
	resp, err := c.Health(ctx())
	if err != nil {
		fatal("stats failed", err)
	}
	out, _ := json.MarshalIndent(resp, "", "  ")
	fmt.Println(string(out))
}

func cmdImport() {
	fs := flag.NewFlagSet("import", flag.ExitOnError)
	file := fs.String("file", "", "JSONL file to import (required)")
	collection := fs.String("collection", "", "Target collection")
	batchSize := fs.Int("batch-size", 100, "Batch size for bulk insert")
	upsert := fs.Bool("upsert", false, "Upsert mode")
	fs.Parse(os.Args[1:])

	if *file == "" {
		fmt.Fprintln(os.Stderr, "error: --file is required")
		fs.Usage()
		os.Exit(1)
	}

	f, err := os.Open(*file)
	if err != nil {
		fatal("open file", err)
	}
	defer f.Close()

	c := newClient()
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 1<<20), 1<<20) // 1MB line limit

	var batch []client.BatchDoc
	total := 0

	for scanner.Scan() {
		line := scanner.Text()
		if strings.TrimSpace(line) == "" {
			continue
		}

		var doc struct {
			ID         string            `json:"id"`
			Doc        string            `json:"doc"`
			Text       string            `json:"text"` // alias
			Meta       map[string]string `json:"meta"`
			Collection string            `json:"collection"`
		}
		if err := json.Unmarshal([]byte(line), &doc); err != nil {
			fmt.Fprintf(os.Stderr, "warning: skip invalid line %d: %v\n", total+1, err)
			continue
		}

		text := doc.Doc
		if text == "" {
			text = doc.Text
		}
		if text == "" {
			continue
		}

		coll := doc.Collection
		if coll == "" {
			coll = *collection
		}

		batch = append(batch, client.BatchDoc{
			ID:         doc.ID,
			Doc:        text,
			Meta:       doc.Meta,
			Collection: coll,
		})

		if len(batch) >= *batchSize {
			if _, err := c.BatchInsert(ctx(), client.BatchInsertRequest{Docs: batch, Upsert: *upsert}); err != nil {
				fatal("batch insert", err)
			}
			total += len(batch)
			fmt.Fprintf(os.Stderr, "\r  imported %d documents...", total)
			batch = batch[:0]
		}
	}

	// Flush remaining
	if len(batch) > 0 {
		if _, err := c.BatchInsert(ctx(), client.BatchInsertRequest{Docs: batch, Upsert: *upsert}); err != nil {
			fatal("batch insert", err)
		}
		total += len(batch)
	}

	if err := scanner.Err(); err != nil {
		fatal("read file", err)
	}

	fmt.Fprintf(os.Stderr, "\r")
	fmt.Printf("imported %d documents from %s\n", total, *file)
}

func cmdExport() {
	fs := flag.NewFlagSet("export", flag.ExitOnError)
	collection := fs.String("collection", "", "Collection to export")
	output := fs.String("output", "", "Output file (default: stdout)")
	topK := fs.Int("limit", 10000, "Maximum documents to export")
	fs.Parse(os.Args[1:])

	c := newClient()

	resp, err := c.Query(ctx(), client.QueryRequest{
		Query:       "", // empty query returns all in scan mode
		TopK:        *topK,
		Collection:  *collection,
		IncludeMeta: true,
		Mode:        "scan",
	})
	if err != nil {
		fatal("export query", err)
	}

	var w *os.File
	if *output != "" && *output != "-" {
		w, err = os.Create(*output)
		if err != nil {
			fatal("create output file", err)
		}
		defer w.Close()
	} else {
		w = os.Stdout
	}

	enc := json.NewEncoder(w)
	for i, id := range resp.IDs {
		doc := ""
		if i < len(resp.Docs) {
			doc = resp.Docs[i]
		}
		var meta map[string]string
		if i < len(resp.Meta) {
			meta = resp.Meta[i]
		}
		rec := map[string]any{
			"id":  id,
			"doc": doc,
		}
		if len(meta) > 0 {
			rec["meta"] = meta
		}
		if *collection != "" {
			rec["collection"] = *collection
		}
		enc.Encode(rec)
	}

	if *output != "" && *output != "-" {
		fmt.Fprintf(os.Stderr, "exported %d documents to %s\n", len(resp.IDs), *output)
	}
}

func cmdCompact() {
	c := newClient()
	fmt.Println("Triggering compaction...")
	resp, err := c.Compact(ctx())
	if err != nil {
		fatal("compact failed", err)
	}
	if resp.OK {
		fmt.Println("Compaction completed successfully.")
	} else {
		fmt.Println("Compaction returned without error but OK=false. Check server logs.")
	}
}

func cmdGentoken() {
	fs := flag.NewFlagSet("gentoken", flag.ExitOnError)
	tenant := fs.String("tenant", "default", "Tenant ID")
	permissions := fs.String("permissions", "read,write", "Comma-separated permissions (read,write,admin)")
	collections := fs.String("collections", "", "Comma-separated allowed collections (empty = all)")
	secret := fs.String("secret", os.Getenv("JWT_SECRET"), "JWT signing secret (or set JWT_SECRET)")
	expires := fs.Duration("expires", 24*time.Hour, "Token expiration (e.g., 24h, 720h)")
	outputJSON := fs.Bool("json", false, "Output as JSON")
	fs.Parse(os.Args[1:])

	if *secret == "" {
		fmt.Fprintln(os.Stderr, "error: JWT secret required. Set JWT_SECRET env var or use --secret")
		fs.Usage()
		os.Exit(1)
	}

	perms := strings.Split(*permissions, ",")
	for i := range perms {
		perms[i] = strings.TrimSpace(perms[i])
	}

	var colls []string
	if *collections != "" {
		colls = strings.Split(*collections, ",")
		for i := range colls {
			colls[i] = strings.TrimSpace(colls[i])
		}
	}

	now := time.Now()
	claims := jwt.MapClaims{
		"tenant_id":   *tenant,
		"permissions": perms,
		"iss":         "vectordb",
		"iat":         now.Unix(),
		"exp":         now.Add(*expires).Unix(),
	}
	if len(colls) > 0 {
		claims["collections"] = colls
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	signed, err := token.SignedString([]byte(*secret))
	if err != nil {
		fatal("sign token", err)
	}

	if *outputJSON {
		out := map[string]any{
			"token":       signed,
			"tenant_id":   *tenant,
			"permissions": perms,
			"collections": colls,
			"expires_at":  now.Add(*expires).Format(time.RFC3339),
		}
		b, _ := json.MarshalIndent(out, "", "  ")
		fmt.Println(string(b))
	} else {
		fmt.Println(signed)
	}
}

// --- Obsidian Import ---

type obsidianNote struct {
	ID            string
	Content       string
	Meta          map[string]string
	OutgoingLinks []string
	FileSize      int64
	ModTime       time.Time
}

// noteName returns the bare note name (filename without .md extension).
func noteName(id string) string {
	return strings.TrimSuffix(filepath.Base(id), ".md")
}

var (
	reWikiLink  = regexp.MustCompile(`\[\[([^\]|]+)(?:\|[^\]]+)?\]\]`)
	reInlineTag = regexp.MustCompile(`(?:^|\s)#([\w\-/]+)`)
	reZettelID  = regexp.MustCompile(`^\d{12,14}`)
)

// syncState tracks file mtimes for incremental sync.
type syncState struct {
	Mtimes map[string]time.Time `json:"mtimes"`
}

func loadSyncState(path string) syncState {
	var s syncState
	s.Mtimes = make(map[string]time.Time)
	data, err := os.ReadFile(path)
	if err != nil {
		return s
	}
	json.Unmarshal(data, &s)
	if s.Mtimes == nil {
		s.Mtimes = make(map[string]time.Time)
	}
	return s
}

func saveSyncState(path string, s syncState) error {
	data, err := json.Marshal(s)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// importStats tracks summary statistics for an import run.
type importStats struct {
	Total       int
	Folders     map[string]bool
	Tags        map[string]bool
	Links       int
	Backlinks   int
	LargestPath string
	LargestSize int64
}

func (s *importStats) collect(notes []obsidianNote) {
	s.Folders = make(map[string]bool)
	s.Tags = make(map[string]bool)
	s.Total = len(notes)
	for _, note := range notes {
		if f := note.Meta["folder"]; f != "" && f != "." {
			s.Folders[f] = true
		}
		if t := note.Meta["tags"]; t != "" {
			for _, tag := range strings.Split(t, ",") {
				s.Tags[tag] = true
			}
		}
		s.Links += len(note.OutgoingLinks)
		if bl := note.Meta["backlinks"]; bl != "" {
			s.Backlinks += len(strings.Split(bl, ","))
		}
		if note.FileSize > s.LargestSize {
			s.LargestSize = note.FileSize
			s.LargestPath = note.ID
		}
	}
}

func (s *importStats) print(vaultName, collection string) {
	fmt.Printf("imported %d notes from vault %q into collection %q\n", s.Total, vaultName, collection)
	fmt.Printf("  folders: %d | tags: %d unique | links: %d | backlinks: %d\n",
		len(s.Folders), len(s.Tags), s.Links, s.Backlinks)
	if s.LargestPath != "" {
		sizeStr := fmt.Sprintf("%dB", s.LargestSize)
		if s.LargestSize >= 1024 {
			sizeStr = fmt.Sprintf("%dKB", s.LargestSize/1024)
		}
		fmt.Printf("  largest: %q (%s)\n", s.LargestPath, sizeStr)
	}
}

func cmdImportObsidian() {
	fs := flag.NewFlagSet("import-obsidian", flag.ExitOnError)
	vault := fs.String("vault", "", "Path to Obsidian vault or markdown folder (required)")
	collection := fs.String("collection", "obsidian", "Target collection")
	batchSize := fs.Int("batch-size", 100, "Docs per batch insert")
	upsert := fs.Bool("upsert", false, "Update existing docs (idempotent re-import)")
	stripFM := fs.Bool("strip-frontmatter", false, "Remove YAML frontmatter from doc text")
	direct := fs.Bool("direct", false, "Force direct file reading (auto-detected if not set)")
	exclude := fs.String("exclude", ".obsidian,.git,.trash,.space,.views", "Dirs to skip (direct mode)")
	watch := fs.Bool("watch", false, "Watch for changes and re-sync automatically")
	watchInterval := fs.Duration("watch-interval", 30*time.Second, "Interval between watch polls")
	incremental := fs.Bool("incremental", false, "Skip unchanged files (by mtime)")
	stateFile := fs.String("state-file", "", "Sync state file path (default: <vault>/.deepdata-sync)")
	prune := fs.Bool("prune", false, "Delete orphaned docs not in vault")
	fs.Parse(os.Args[1:])

	if *vault == "" {
		fmt.Fprintln(os.Stderr, "error: --vault is required")
		fs.Usage()
		os.Exit(1)
	}

	vaultPath, err := filepath.Abs(*vault)
	if err != nil {
		fatal("resolve vault path", err)
	}
	info, err := os.Stat(vaultPath)
	if err != nil || !info.IsDir() {
		fmt.Fprintf(os.Stderr, "error: vault path %q is not a directory\n", vaultPath)
		os.Exit(1)
	}

	// Auto-detect: is this an Obsidian vault or a plain markdown folder?
	_, obsidianDirErr := os.Stat(filepath.Join(vaultPath, ".obsidian"))
	isObsidianVault := obsidianDirErr == nil
	source := "obsidian"
	if !isObsidianVault {
		source = "markdown"
	}

	// Auto-detect direct mode: use direct if explicitly requested, if it's
	// a plain folder, or if the obsidian CLI isn't available.
	useDirect := *direct || !isObsidianVault
	if !useDirect && !*direct {
		if _, err := exec.LookPath("obsidian"); err != nil {
			useDirect = true
		}
	}

	vaultName := filepath.Base(vaultPath)
	excludeSet := make(map[string]bool)
	for _, d := range strings.Split(*exclude, ",") {
		d = strings.TrimSpace(d)
		if d != "" {
			excludeSet[d] = true
		}
	}

	// Resolve state file path for incremental sync
	syncFilePath := *stateFile
	if syncFilePath == "" {
		syncFilePath = filepath.Join(vaultPath, ".deepdata-sync")
	}

	if useDirect {
		if isObsidianVault {
			fmt.Fprintf(os.Stderr, "reading vault %q directly (obsidian CLI not used)\n", vaultName)
		} else {
			fmt.Fprintf(os.Stderr, "reading markdown folder %q\n", vaultName)
		}
	}

	// Single shared client for all operations in this run.
	c := newClient()

	// Build the import function used by both single-run and watch mode.
	doImport := func(prevMtimes map[string]time.Time) map[string]time.Time {
		// Pass 1: Collect notes
		var notes []obsidianNote
		var collectErr error
		if useDirect {
			notes, collectErr = collectNotesDirect(vaultPath, vaultName, source, excludeSet, *stripFM)
		} else {
			notes, collectErr = collectNotesFromObsidianCLI(vaultPath, vaultName, *stripFM)
			if collectErr != nil {
				fmt.Fprintf(os.Stderr, "warning: obsidian CLI failed: %v\nfalling back to direct file reading\n", collectErr)
				notes, collectErr = collectNotesDirect(vaultPath, vaultName, source, excludeSet, *stripFM)
			}
		}
		if collectErr != nil {
			fmt.Fprintf(os.Stderr, "error: collect notes: %v\n", collectErr)
			return prevMtimes
		}

		if len(notes) == 0 {
			fmt.Println("no notes found in vault")
			return prevMtimes
		}

		// Build current mtime map from stored ModTime (no re-stat)
		currentMtimes := make(map[string]time.Time, len(notes))
		for _, note := range notes {
			currentMtimes[note.ID] = note.ModTime
		}

		// Incremental: filter to changed/new notes only
		var toUpsert []obsidianNote
		if *incremental && len(prevMtimes) > 0 {
			// Find changed or new files
			changedSet := make(map[string]bool)
			for _, note := range notes {
				prev, existed := prevMtimes[note.ID]
				if !existed || !note.ModTime.Equal(prev) {
					changedSet[note.ID] = true
				}
			}

			if len(changedSet) == 0 {
				// Check for deletions even if no changes
				deleted := 0
				if *prune {
					deleted = pruneOrphans(c, notes, *collection)
				}
				if deleted > 0 {
					fmt.Printf("no changed notes, pruned %d orphans\n", deleted)
				} else {
					fmt.Println("no changes detected")
				}
				saveSyncState(syncFilePath, syncState{Mtimes: currentMtimes})
				return currentMtimes
			}

			// Compute backlinks on full set (backlinks depend on global state)
			computeBacklinks(notes)

			// Pre-compute set of changed note names for O(1) lookup
			changedNames := make(map[string]bool, len(changedSet))
			for cid := range changedSet {
				changedNames[noteName(cid)] = true
			}

			// Collect changed notes + their backlink-affected neighbors
			for _, note := range notes {
				if changedSet[note.ID] {
					toUpsert = append(toUpsert, note)
					continue
				}
				// If any of this note's backlinks are from a changed note,
				// its backlink metadata needs re-upserting
				if bl := note.Meta["backlinks"]; bl != "" {
					for _, link := range strings.Split(bl, ",") {
						if changedNames[link] {
							toUpsert = append(toUpsert, note)
							break
						}
					}
				}
			}

			newCount := 0
			for id := range changedSet {
				if _, existed := prevMtimes[id]; !existed {
					newCount++
				}
			}
			fmt.Fprintf(os.Stderr, "incremental: %d changed, %d new\n", len(changedSet)-newCount, newCount)
		} else {
			// Full import
			computeBacklinks(notes)
			toUpsert = notes
			fmt.Fprintf(os.Stderr, "collected %d notes\n", len(notes))
		}

		// Batch insert
		total, err := batchUpsertNotes(c, toUpsert, *collection, *batchSize, *upsert || *incremental)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: batch insert: %v\n", err)
			return prevMtimes
		}

		// Prune orphaned docs
		pruned := 0
		if *prune {
			pruned = pruneOrphans(c, notes, *collection)
		}

		// Save sync state
		saveSyncState(syncFilePath, syncState{Mtimes: currentMtimes})

		// Print summary stats
		var stats importStats
		stats.collect(notes)
		if *incremental && len(prevMtimes) > 0 {
			fmt.Printf("synced %d notes to collection %q", total, *collection)
			if pruned > 0 {
				fmt.Printf(", pruned %d orphans", pruned)
			}
			fmt.Println()
		} else {
			stats.print(vaultName, *collection)
			if pruned > 0 {
				fmt.Printf("  pruned: %d orphaned docs\n", pruned)
			}
		}

		return currentMtimes
	}

	// Watch mode: loop with interval
	if *watch {
		// Enable incremental and upsert implicitly in watch mode
		*incremental = true
		*upsert = true

		// Load previous state for incremental first cycle
		state := loadSyncState(syncFilePath)
		prevMtimes := state.Mtimes

		fmt.Fprintf(os.Stderr, "watching vault %q (interval: %s) — press Ctrl+C to stop\n", vaultName, *watchInterval)

		// Initial sync (incremental if state exists, full otherwise)
		prevMtimes = doImport(prevMtimes)

		// Set up signal handler
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

		ticker := time.NewTicker(*watchInterval)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				prevMtimes = doImport(prevMtimes)
			case <-sigCh:
				fmt.Fprintf(os.Stderr, "\nstopping watch\n")
				return
			}
		}
	}

	// Single run mode
	var prevMtimes map[string]time.Time
	if *incremental {
		state := loadSyncState(syncFilePath)
		prevMtimes = state.Mtimes
	}
	doImport(prevMtimes)
}

// batchUpsertNotes inserts notes in batches, returning the total count inserted.
func batchUpsertNotes(c *client.Client, notes []obsidianNote, collection string, batchSize int, upsert bool) (int, error) {
	var batch []client.BatchDoc
	total := 0
	for _, note := range notes {
		batch = append(batch, client.BatchDoc{
			ID:         note.ID,
			Doc:        note.Content,
			Meta:       note.Meta,
			Collection: collection,
		})
		if len(batch) >= batchSize {
			if _, err := c.BatchInsert(ctx(), client.BatchInsertRequest{Docs: batch, Upsert: upsert}); err != nil {
				return total, err
			}
			total += len(batch)
			fmt.Fprintf(os.Stderr, "\r  imported %d/%d notes...", total, len(notes))
			batch = batch[:0]
		}
	}
	if len(batch) > 0 {
		if _, err := c.BatchInsert(ctx(), client.BatchInsertRequest{Docs: batch, Upsert: upsert}); err != nil {
			return total, err
		}
		total += len(batch)
	}
	fmt.Fprintf(os.Stderr, "\r")
	return total, nil
}

// pruneOrphans deletes docs from the collection that no longer have a matching
// file in the vault. Returns the number of pruned docs.
func pruneOrphans(c *client.Client, currentNotes []obsidianNote, collection string) int {
	// Build set of current note IDs (relative paths)
	currentIDs := make(map[string]bool, len(currentNotes))
	for _, note := range currentNotes {
		currentIDs[note.ID] = true
	}

	pruned := 0
	offset := 0
	limit := 500

	for {
		resp, err := c.Scroll(ctx(), client.ScrollRequest{
			Collection: collection,
			Limit:      limit,
			Offset:     offset,
		})
		if err != nil {
			fmt.Fprintf(os.Stderr, "warning: prune scroll failed: %v\n", err)
			break
		}
		if len(resp.IDs) == 0 {
			break
		}

		for i, id := range resp.IDs {
			// Check if this doc has obsidian/markdown source metadata
			var docSource string
			if i < len(resp.Meta) && resp.Meta[i] != nil {
				docSource = resp.Meta[i]["source"]
			}
			if docSource != "obsidian" && docSource != "markdown" {
				continue
			}
			// Check if the doc's ID (which is the relative path) still exists in vault
			if !currentIDs[id] {
				if _, err := c.Delete(ctx(), client.DeleteRequest{ID: id}); err != nil {
					fmt.Fprintf(os.Stderr, "warning: failed to prune %s: %v\n", id, err)
				} else {
					pruned++
				}
			}
		}

		if resp.Next <= offset || len(resp.IDs) < limit {
			break
		}
		offset = resp.Next
	}

	return pruned
}

func collectNotesDirect(vaultPath, vaultName, source string, excludeSet map[string]bool, stripFM bool) ([]obsidianNote, error) {
	var notes []obsidianNote
	err := filepath.WalkDir(vaultPath, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil // skip unreadable
		}
		if d.IsDir() {
			if excludeSet[d.Name()] {
				return filepath.SkipDir
			}
			return nil
		}
		if filepath.Ext(path) != ".md" {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return nil
		}
		if info.Size() == 0 || info.Size() > 1<<20 {
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return nil
		}
		rel, _ := filepath.Rel(vaultPath, path)
		note := parseObsidianNote(rel, string(data), vaultName, source, info.ModTime(), info.Size(), stripFM)
		notes = append(notes, note)
		return nil
	})
	return notes, err
}

func collectNotesFromObsidianCLI(vaultPath, vaultName string, stripFM bool) ([]obsidianNote, error) {
	// Try to list notes via obsidian CLI search
	cmd := exec.Command("obsidian", "search", `query=*`, "format=json")
	cmd.Env = append(os.Environ(), "OBSIDIAN_VAULT="+vaultPath)
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("obsidian CLI failed (is Obsidian running?): %w\ntip: use --direct to read files without Obsidian", err)
	}

	var searchResults []struct {
		File string `json:"file"`
	}
	if err := json.Unmarshal(out, &searchResults); err != nil {
		return nil, fmt.Errorf("parse obsidian search output: %w", err)
	}

	var notes []obsidianNote
	for _, sr := range searchResults {
		if filepath.Ext(sr.File) != ".md" {
			continue
		}
		// Read file content via obsidian CLI
		readCmd := exec.Command("obsidian", "read", "file="+sr.File)
		readCmd.Env = append(os.Environ(), "OBSIDIAN_VAULT="+vaultPath)
		content, err := readCmd.Output()
		if err != nil {
			fmt.Fprintf(os.Stderr, "warning: could not read %s: %v\n", sr.File, err)
			continue
		}
		if len(content) == 0 || len(content) > 1<<20 {
			continue
		}

		fullPath := filepath.Join(vaultPath, sr.File)
		var modTime time.Time
		var fileSize int64
		if info, err := os.Stat(fullPath); err == nil {
			modTime = info.ModTime()
			fileSize = info.Size()
		}

		note := parseObsidianNote(sr.File, string(content), vaultName, "obsidian", modTime, fileSize, stripFM)
		notes = append(notes, note)
	}
	return notes, nil
}

func parseObsidianNote(relPath, content, vaultName, source string, modTime time.Time, fileSize int64, stripFM bool) obsidianNote {
	meta := map[string]string{
		"source": source,
		"vault":  vaultName,
		"path":   relPath,
		"folder": filepath.Dir(relPath),
	}

	baseName := noteName(relPath)
	meta["title"] = baseName

	// Parse YAML frontmatter (single pass)
	body := content
	var fmTags []string
	if strings.HasPrefix(content, "---\n") {
		endIdx := strings.Index(content[4:], "\n---")
		if endIdx >= 0 {
			fmBlock := content[4 : 4+endIdx]
			body = content[4+endIdx+4:] // after closing ---\n

			inTags := false
			for _, line := range strings.Split(fmBlock, "\n") {
				trimmed := strings.TrimSpace(line)
				if trimmed == "" || strings.HasPrefix(trimmed, "#") {
					continue
				}

				// Handle multi-line tag list items
				if inTags {
					if strings.HasPrefix(trimmed, "- ") {
						t := strings.TrimSpace(strings.TrimPrefix(trimmed, "- "))
						t = strings.TrimPrefix(t, "#")
						if t != "" {
							fmTags = append(fmTags, t)
						}
						continue
					}
					inTags = false
				}

				k, v, ok := strings.Cut(trimmed, ":")
				if !ok {
					continue
				}
				k = strings.TrimSpace(k)
				v = strings.TrimSpace(v)
				switch k {
				case "title":
					meta["title"] = strings.Trim(v, "\"'")
				case "author":
					meta["author"] = strings.NewReplacer("[[", "", "]]", "").Replace(strings.Trim(v, "\"'"))
				case "created", "date":
					meta["created"] = strings.Trim(v, "\"'")
				case "zettelkasten_id", "uid", "zettel":
					meta["zettelkasten_id"] = strings.Trim(v, "\"'")
				case "tags":
					// tags can be: [a, b, c] or a, b, c (inline) or empty (multi-line follows)
					v = strings.Trim(v, "[]\"'")
					if v == "" {
						inTags = true // next lines are "- tag" items
					} else {
						for _, t := range strings.Split(v, ",") {
							t = strings.TrimSpace(t)
							t = strings.TrimPrefix(t, "#")
							if t != "" {
								fmTags = append(fmTags, t)
							}
						}
					}
				}
			}
		}
	}

	// Extract inline #tags from body (skip headings: lines starting with #)
	for _, line := range strings.Split(body, "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "#") && (len(trimmed) < 2 || trimmed[1] == ' ' || trimmed[1] == '#') {
			continue // heading
		}
		for _, m := range reInlineTag.FindAllStringSubmatch(line, -1) {
			fmTags = append(fmTags, m[1])
		}
	}

	// Deduplicate tags
	if len(fmTags) > 0 {
		seen := make(map[string]bool)
		var unique []string
		for _, t := range fmTags {
			lower := strings.ToLower(t)
			if !seen[lower] {
				seen[lower] = true
				unique = append(unique, t)
			}
		}
		meta["tags"] = strings.Join(unique, ",")
	}

	// Extract [[wiki-links]]
	var outgoing []string
	seen := make(map[string]bool)
	for _, m := range reWikiLink.FindAllStringSubmatch(body, -1) {
		link := strings.TrimSpace(m[1])
		// Strip path prefix if present (e.g., folder/note -> note)
		if idx := strings.LastIndex(link, "/"); idx >= 0 {
			link = link[idx+1:]
		}
		if !seen[link] {
			seen[link] = true
			outgoing = append(outgoing, link)
		}
	}
	if len(outgoing) > 0 {
		meta["outgoing_links"] = strings.Join(outgoing, ",")
	}

	// Detect Zettelkasten ID from filename if not in frontmatter
	if meta["zettelkasten_id"] == "" {
		if m := reZettelID.FindString(baseName); m != "" {
			meta["zettelkasten_id"] = m
		}
	}

	// Created from modTime if not in frontmatter
	if meta["created"] == "" && !modTime.IsZero() {
		meta["created"] = modTime.Format("2006-01-02")
	}

	// Optionally strip frontmatter from content
	docContent := content
	if stripFM {
		docContent = body
	}

	return obsidianNote{
		ID:            relPath,
		Content:       docContent,
		Meta:          meta,
		OutgoingLinks: outgoing,
		FileSize:      fileSize,
		ModTime:       modTime,
	}
}

func computeBacklinks(notes []obsidianNote) {
	// Build reverse map: link target name -> list of source note names
	backlinks := map[string][]string{}
	for _, note := range notes {
		name := noteName(note.ID)
		for _, link := range note.OutgoingLinks {
			backlinks[link] = append(backlinks[link], name)
		}
	}
	// Apply to each note
	for i, note := range notes {
		name := noteName(note.ID)
		if bl := backlinks[name]; len(bl) > 0 {
			notes[i].Meta["backlinks"] = strings.Join(bl, ",")
		}
	}
}

// parseMeta parses "key1=val1,key2=val2" into a map.
func parseMeta(s string) map[string]string {
	if s == "" {
		return nil
	}
	m := make(map[string]string)
	for _, pair := range strings.Split(s, ",") {
		parts := strings.SplitN(pair, "=", 2)
		if len(parts) == 2 {
			m[strings.TrimSpace(parts[0])] = strings.TrimSpace(parts[1])
		}
	}
	return m
}
