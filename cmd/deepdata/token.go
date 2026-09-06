package main

import (
	"flag"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/phenomenon0/vectordb/internal/security"
)

// runToken is the `deepdata token` subcommand: mint a tenant JWT offline
// with the server's own JWT_SECRET.
func runToken(args []string, stdout, stderr io.Writer, getenv func(string) string) int {
	fs := flag.NewFlagSet("token", flag.ContinueOnError)
	fs.SetOutput(stderr)
	tenant := fs.String("tenant", "", "tenant ID (required)")
	permissions := fs.String("permissions", "read,write", "comma-separated permissions (read,write,admin)")
	collections := fs.String("collections", "", "comma-separated collection allowlist (empty = all)")
	ttl := fs.Duration("ttl", 24*time.Hour, "token lifetime")
	serverAdmin := fs.Bool("server-admin", false, "mint a global server-administrator token")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if strings.TrimSpace(*tenant) == "" {
		fmt.Fprintln(stderr, "token: --tenant is required")
		return 2
	}

	secret := getenv("JWT_SECRET")
	if err := validateCanonicalCredential("JWT_SECRET", secret); err != nil {
		fmt.Fprintln(stderr, "JWT_SECRET is required to mint tokens (32+ bytes)")
		return 2
	}
	issuer := getenv("JWT_ISSUER")
	if issuer == "" {
		issuer = "vectordb"
	}

	token, err := security.NewJWTManager(secret, issuer).SignTenantClaims(security.TenantClaims{
		TenantID:    *tenant,
		Permissions: splitCSV(*permissions),
		Collections: splitCSV(*collections),
		ServerAdmin: *serverAdmin,
	}, *ttl)
	if err != nil {
		fmt.Fprintf(stderr, "token: failed to sign token: %v\n", err)
		return 1
	}
	fmt.Fprintln(stdout, token)
	return 0
}

// splitCSV trims and drops empty entries from a comma-separated flag value.
func splitCSV(s string) []string {
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}
