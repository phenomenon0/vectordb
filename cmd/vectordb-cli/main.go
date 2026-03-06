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
//	import        Bulk import from JSONL file
//	export        Export collection to JSONL file
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
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/phenomenon0/Agent-GO/vectordb/client"
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
  import        Bulk import from JSONL file (--file, --collection)
  export        Export collection to JSONL (--collection, --output)
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
