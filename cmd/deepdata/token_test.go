package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/phenomenon0/vectordb/internal/security"
)

func TestTokenSubcommandMintsVerifiableJWT(t *testing.T) {
	const secret = "0123456789abcdef0123456789abcdef" // gitleaks:allow -- deterministic test-only credential
	getenv := mapGetenv(map[string]string{"JWT_SECRET": secret})

	var stdout, stderr bytes.Buffer
	args := []string{
		"--tenant", "acme",
		"--permissions", "read,write,admin",
		"--collections", "docs,images",
		"--ttl", "2h",
		"--server-admin",
	}
	if code := runToken(args, &stdout, &stderr, getenv); code != 0 {
		t.Fatalf("runToken exit = %d, want 0 (stderr: %s)", code, stderr.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("stderr = %q, want empty", stderr.String())
	}
	out := stdout.String()
	if strings.Count(out, "\n") != 1 || !strings.HasSuffix(out, "\n") {
		t.Fatalf("stdout = %q, want exactly one line", out)
	}
	token := strings.TrimSuffix(out, "\n")

	tenantCtx, err := security.NewJWTManager(secret, "vectordb").ValidateTenantToken(token)
	if err != nil {
		t.Fatalf("minted token does not validate: %v", err)
	}
	if tenantCtx.TenantID != "acme" {
		t.Fatalf("TenantID = %q, want acme", tenantCtx.TenantID)
	}
	for _, perm := range []string{"read", "write", "admin"} {
		if !tenantCtx.Permissions[perm] {
			t.Fatalf("permission %q missing from %v", perm, tenantCtx.Permissions)
		}
	}
	for _, coll := range []string{"docs", "images"} {
		if !tenantCtx.Collections[coll] {
			t.Fatalf("collection %q missing from %v", coll, tenantCtx.Collections)
		}
	}
	if !tenantCtx.IsServerAdmin {
		t.Fatal("IsServerAdmin = false, want true")
	}
}

func TestTokenSubcommandRequiresSecret(t *testing.T) {
	getenv := mapGetenv(map[string]string{"JWT_SECRET": "too-short"})

	var stdout, stderr bytes.Buffer
	code := runToken([]string{"--tenant", "acme"}, &stdout, &stderr, getenv)
	if code != 2 {
		t.Fatalf("runToken exit = %d, want 2", code)
	}
	if stdout.Len() != 0 {
		t.Fatalf("stdout = %q, want empty", stdout.String())
	}
	if !strings.Contains(stderr.String(), "JWT_SECRET") {
		t.Fatalf("stderr = %q, want it to name JWT_SECRET", stderr.String())
	}
	if strings.Contains(stderr.String(), "too-short") {
		t.Fatalf("stderr leaked the secret value: %q", stderr.String())
	}
}
