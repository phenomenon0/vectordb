package security

import (
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

func TestAuthorizeTenantAccessCentralPolicy(t *testing.T) {
	tenantAdmin := &TenantContext{
		TenantID:      "acme",
		Permissions:   map[string]bool{"admin": true},
		Collections:   map[string]bool{"docs": true},
		IsAdmin:       true,
		IsServerAdmin: false,
	}
	serverAdmin := &TenantContext{IsServerAdmin: true}

	tests := []struct {
		name        string
		context     *TenantContext
		tenantID    string
		collection  string
		permission  string
		wantFailure AuthorizationFailure
		wantMessage string
	}{
		{
			name:        "missing context",
			tenantID:    "acme",
			collection:  "docs",
			permission:  "read",
			wantFailure: AuthorizationUnauthenticated,
			wantMessage: "authenticated tenant context required",
		},
		{
			name:       "server admin crosses scopes",
			context:    serverAdmin,
			tenantID:   "other",
			collection: "secret",
			permission: "admin",
		},
		{
			name: "tenant mismatch",
			context: &TenantContext{
				TenantID:    "acme",
				Permissions: map[string]bool{"read": true},
			},
			tenantID:    "other",
			collection:  "docs",
			permission:  "read",
			wantFailure: AuthorizationPermissionDenied,
			wantMessage: "cannot access tenant",
		},
		{
			name: "permission denied",
			context: &TenantContext{
				TenantID:    "acme",
				Permissions: map[string]bool{"read": true},
			},
			tenantID:    "acme",
			collection:  "docs",
			permission:  "write",
			wantFailure: AuthorizationPermissionDenied,
			wantMessage: "write permission required",
		},
		{
			name:        "collection scoped admin cannot use tenant wide endpoint",
			context:     tenantAdmin,
			tenantID:    "acme",
			permission:  "admin",
			wantFailure: AuthorizationPermissionDenied,
			wantMessage: "collection-scoped token",
		},
		{
			name:        "collection scoped admin cannot escape allowlist",
			context:     tenantAdmin,
			tenantID:    "acme",
			collection:  "secret",
			permission:  "admin",
			wantFailure: AuthorizationPermissionDenied,
			wantMessage: "cannot access collection",
		},
		{
			name:       "collection scoped admin can use allowed collection",
			context:    tenantAdmin,
			tenantID:   "acme",
			collection: "docs",
			permission: "admin",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := AuthorizeTenantAccess(test.context, test.tenantID, test.collection, test.permission)
			if test.wantFailure == 0 {
				if err != nil {
					t.Fatalf("authorization failed: %v", err)
				}
				return
			}
			if !IsAuthorizationFailure(err, test.wantFailure) {
				t.Fatalf("authorization error = %v, want failure %v", err, test.wantFailure)
			}
			if !strings.Contains(err.Error(), test.wantMessage) {
				t.Fatalf("authorization error = %q, want substring %q", err, test.wantMessage)
			}
		})
	}

	if err := AuthorizeTenantPermission(tenantAdmin, "acme", "admin"); err != nil {
		t.Fatalf("permission preflight rejected collection-scoped tenant admin: %v", err)
	}
}

func TestExtractTokenUsesAuthorizationHeaderOnly(t *testing.T) {
	queryOnly := httptest.NewRequest("GET", "/protected?token=query-secret", nil)
	if token := extractToken(queryOnly); token != "" {
		t.Fatalf("query-string credential was accepted: %q", token)
	}

	bearer := httptest.NewRequest("GET", "/protected?token=query-secret", nil)
	bearer.Header.Set("Authorization", "Bearer header-secret")
	if token := extractToken(bearer); token != "header-secret" {
		t.Fatalf("bearer token = %q, want header-secret", token)
	}

	raw := httptest.NewRequest("GET", "/protected", nil)
	raw.Header.Set("Authorization", "raw-header-secret")
	if token := extractToken(raw); token != "raw-header-secret" {
		t.Fatalf("raw header token = %q, want raw-header-secret", token)
	}
}

func TestServerAdminClaimRoundTrip(t *testing.T) {
	const secret = "0123456789abcdef0123456789abcdef" // gitleaks:allow -- deterministic test-only credential
	manager := NewJWTManager(secret, "deepdata-test")

	adminToken, err := manager.SignTenantClaims(TenantClaims{TenantID: "acme", ServerAdmin: true}, time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	adminCtx, err := manager.ValidateTenantToken(adminToken)
	if err != nil {
		t.Fatalf("ValidateTenantToken rejected a server_admin token: %v", err)
	}
	if !adminCtx.IsServerAdmin {
		t.Fatal("IsServerAdmin = false, want true for a server_admin claim")
	}

	plainToken, err := manager.GenerateTenantToken("acme", []string{"read"}, nil, time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	plainCtx, err := manager.ValidateTenantToken(plainToken)
	if err != nil {
		t.Fatal(err)
	}
	if plainCtx.IsServerAdmin {
		t.Fatal("IsServerAdmin = true, want false without a server_admin claim")
	}

	if err := AuthorizeServerAdmin(plainCtx); !IsAuthorizationFailure(err, AuthorizationPermissionDenied) {
		t.Fatalf("AuthorizeServerAdmin(plain tenant) = %v, want permission denied", err)
	}
	if err := AuthorizeServerAdmin(adminCtx); err != nil {
		t.Fatalf("AuthorizeServerAdmin(server admin) = %v, want nil", err)
	}
}

func TestJWTValidationAcceptsOnlyHS256(t *testing.T) {
	const secret = "0123456789abcdef0123456789abcdef" // gitleaks:allow -- deterministic test-only credential
	manager := NewJWTManager(secret, "deepdata-test")

	validTenantToken, err := manager.GenerateTenantToken("acme", []string{"read"}, nil, time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := manager.ValidateTenantToken(validTenantToken); err != nil {
		t.Fatalf("HS256 tenant token rejected: %v", err)
	}

	hs512Generic := jwt.NewWithClaims(jwt.SigningMethodHS512, jwt.MapClaims{
		"sub": "test-user",
		"exp": time.Now().Add(time.Hour).Unix(),
	})
	hs512GenericString, err := hs512Generic.SignedString([]byte(secret))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := manager.ValidateToken(hs512GenericString); err == nil {
		t.Fatal("ValidateToken accepted an HS512 token")
	}

	hs512Tenant := jwt.NewWithClaims(jwt.SigningMethodHS512, &TenantClaims{
		TenantID:    "acme",
		Permissions: []string{"read"},
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(time.Hour)),
			Issuer:    "deepdata-test",
		},
	})
	hs512TenantString, err := hs512Tenant.SignedString([]byte(secret))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := manager.ValidateTenantToken(hs512TenantString); err == nil {
		t.Fatal("ValidateTenantToken accepted an HS512 token")
	}
}
