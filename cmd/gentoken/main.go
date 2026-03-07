package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/phenomenon0/vectordb/internal/security"
)

func main() {
	if len(os.Args) > 1 && os.Args[1] == "preset" {
		generatePresetTokens()
		return
	}

	// Parse command-line flags
	tenantID := flag.String("tenant", "test-tenant", "Tenant ID")
	permissions := flag.String("permissions", "read,write", "Comma-separated permissions (read,write,admin)")
	collections := flag.String("collections", "", "Comma-separated collections (empty = all collections)")
	secret := flag.String("secret", os.Getenv("JWT_SECRET"), "JWT secret key")
	issuer := flag.String("issuer", "vectordb", "JWT issuer")
	expiresIn := flag.Duration("expires", 24*time.Hour, "Token expiration duration (e.g., 24h, 168h for 7d)")
	outputJSON := flag.Bool("json", false, "Output as JSON")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Generate JWT tokens for VectorDB multi-tenant authentication\n\n")
		fmt.Fprintf(os.Stderr, "Usage:\n")
		fmt.Fprintf(os.Stderr, "  %s [options]           Generate custom token\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s preset             Generate preset tokens for dev/testing\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  # Generate admin token for all collections\n")
		fmt.Fprintf(os.Stderr, "  %s -tenant=acme-corp -permissions=admin -secret=my-secret\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  # Generate read-only token for specific collections\n")
		fmt.Fprintf(os.Stderr, "  %s -tenant=customer-1 -permissions=read -collections=docs,images\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  # Generate preset tokens\n")
		fmt.Fprintf(os.Stderr, "  JWT_SECRET=my-secret %s preset\n\n", os.Args[0])
	}

	flag.Parse()

	if *secret == "" {
		fmt.Fprintf(os.Stderr, "Error: JWT secret required. Set JWT_SECRET env var or use -secret flag\n")
		os.Exit(1)
	}

	// Parse permissions
	perms := []string{}
	if *permissions != "" {
		perms = strings.Split(*permissions, ",")
		for i := range perms {
			perms[i] = strings.TrimSpace(perms[i])
		}
	}

	// Parse collections
	colls := []string{}
	if *collections != "" {
		colls = strings.Split(*collections, ",")
		for i := range colls {
			colls[i] = strings.TrimSpace(colls[i])
		}
	}

	// Create JWT manager
	jwtMgr := security.NewJWTManager(*secret, *issuer)

	// Generate token
	token, err := jwtMgr.GenerateTenantToken(*tenantID, perms, colls, *expiresIn)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error generating token: %v\n", err)
		os.Exit(1)
	}

	// Output token
	if *outputJSON {
		output := map[string]interface{}{
			"token":       token,
			"tenant_id":   *tenantID,
			"permissions": perms,
			"collections": colls,
			"expires_in":  expiresIn.String(),
			"issued_at":   time.Now().Format(time.RFC3339),
			"expires_at":  time.Now().Add(*expiresIn).Format(time.RFC3339),
		}
		jsonBytes, _ := json.MarshalIndent(output, "", "  ")
		fmt.Println(string(jsonBytes))
	} else {
		fmt.Println("=== JWT Token Generated ===")
		fmt.Printf("Tenant ID:    %s\n", *tenantID)
		fmt.Printf("Permissions:  %v\n", perms)
		if len(colls) > 0 {
			fmt.Printf("Collections:  %v\n", colls)
		} else {
			fmt.Printf("Collections:  (all)\n")
		}
		fmt.Printf("Expires:      %s (%s from now)\n", time.Now().Add(*expiresIn).Format(time.RFC3339), expiresIn.String())
		fmt.Println("\n=== Token (use as Bearer token) ===")
		fmt.Println(token)
		fmt.Println("\n=== Usage Example ===")
		fmt.Printf("curl -H \"Authorization: Bearer %s\" http://localhost:8080/query \\\n", token)
		fmt.Println("  -X POST -H \"Content-Type: application/json\" \\")
		fmt.Println("  -d '{\"query\": \"test\", \"top_k\": 3}'")
	}
}

func generatePresetTokens() {
	secret := os.Getenv("JWT_SECRET")
	if secret == "" {
		fmt.Println("Error: JWT_SECRET environment variable required")
		os.Exit(1)
	}

	jwtMgr := security.NewJWTManager(secret, "vectordb")

	fmt.Println("=== Preset JWT Tokens for Development ===")

	// Admin token
	adminToken, _ := jwtMgr.GenerateTenantToken("admin", []string{"admin"}, []string{}, 24*365*time.Hour)
	fmt.Println("1. ADMIN TOKEN (full access, 1 year)")
	fmt.Printf("   Tenant: admin\n")
	fmt.Printf("   Token:  %s\n\n", adminToken)

	// Read-only token
	readToken, _ := jwtMgr.GenerateTenantToken("viewer", []string{"read"}, []string{}, 7*24*time.Hour)
	fmt.Println("2. READ-ONLY TOKEN (7 days)")
	fmt.Printf("   Tenant: viewer\n")
	fmt.Printf("   Token:  %s\n\n", readToken)

	// Read-write token
	rwToken, _ := jwtMgr.GenerateTenantToken("editor", []string{"read", "write"}, []string{}, 30*24*time.Hour)
	fmt.Println("3. READ-WRITE TOKEN (30 days)")
	fmt.Printf("   Tenant: editor\n")
	fmt.Printf("   Token:  %s\n\n", rwToken)

	// Limited collections token
	limitedToken, _ := jwtMgr.GenerateTenantToken("partner", []string{"read", "write"}, []string{"public", "shared"}, 24*time.Hour)
	fmt.Println("4. LIMITED COLLECTIONS TOKEN (24 hours)")
	fmt.Printf("   Tenant:      partner\n")
	fmt.Printf("   Collections: public, shared\n")
	fmt.Printf("   Token:       %s\n\n", limitedToken)

	fmt.Println("=== Save These Tokens for Testing ===")
	fmt.Println("export ADMIN_TOKEN=\"" + adminToken + "\"")
	fmt.Println("export READ_TOKEN=\"" + readToken + "\"")
	fmt.Println("export WRITE_TOKEN=\"" + rwToken + "\"")
	fmt.Println("export LIMITED_TOKEN=\"" + limitedToken + "\"")
}
