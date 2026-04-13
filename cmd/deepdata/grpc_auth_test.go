package main

import (
	"context"
	"testing"
	"time"

	"github.com/phenomenon0/vectordb/internal/logging"
	"github.com/phenomenon0/vectordb/internal/security"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

// passHandler is a gRPC handler stub that returns the context's TenantContext.
func passThroughHandler(ctx context.Context, req any) (any, error) {
	tc, _ := security.GetTenantContextFromContext(ctx)
	return tc, nil
}

func dummyServerInfo(method string) *grpc.UnaryServerInfo {
	return &grpc.UnaryServerInfo{FullMethod: method}
}

func testLogger() *logging.Logger {
	return logging.Init(logging.Config{Level: logging.LevelError})
}

// ─── JWT auth tests ─────────────────────────────────────────────────────────

func TestGRPCAuth_JWT_NoToken_RequireAuth(t *testing.T) {
	jwtMgr := security.NewJWTManager("test-secret-key-for-grpc", "test-issuer")
	interceptor := grpcAuthInterceptor(jwtMgr, "", true, testLogger())

	ctx := context.Background()
	_, err := interceptor(ctx, nil, dummyServerInfo("/test.Service/Method"), passThroughHandler)
	if err == nil {
		t.Fatal("expected Unauthenticated error, got nil")
	}
	st, ok := status.FromError(err)
	if !ok {
		t.Fatalf("expected gRPC status error, got %v", err)
	}
	if st.Code() != codes.Unauthenticated {
		t.Fatalf("expected Unauthenticated, got %v", st.Code())
	}
}

func TestGRPCAuth_JWT_ValidToken(t *testing.T) {
	jwtMgr := security.NewJWTManager("test-secret-key-for-grpc", "test-issuer")
	interceptor := grpcAuthInterceptor(jwtMgr, "", true, testLogger())

	// Generate a valid tenant token
	token, err := jwtMgr.GenerateTenantToken("tenant-42", []string{"read", "write"}, []string{"my-collection"}, time.Hour)
	if err != nil {
		t.Fatalf("failed to generate token: %v", err)
	}

	md := metadata.Pairs("authorization", "Bearer "+token)
	ctx := metadata.NewIncomingContext(context.Background(), md)

	resp, err := interceptor(ctx, nil, dummyServerInfo("/test.Service/Method"), passThroughHandler)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	tc, ok := resp.(*security.TenantContext)
	if !ok {
		t.Fatalf("expected *TenantContext, got %T", resp)
	}
	if tc.TenantID != "tenant-42" {
		t.Errorf("expected tenant-42, got %s", tc.TenantID)
	}
	if !tc.Permissions["read"] || !tc.Permissions["write"] {
		t.Errorf("expected read+write permissions, got %v", tc.Permissions)
	}
	if !tc.Collections["my-collection"] {
		t.Errorf("expected my-collection in collections, got %v", tc.Collections)
	}
}

func TestGRPCAuth_JWT_InvalidToken(t *testing.T) {
	jwtMgr := security.NewJWTManager("test-secret-key-for-grpc", "test-issuer")
	interceptor := grpcAuthInterceptor(jwtMgr, "", true, testLogger())

	md := metadata.Pairs("authorization", "Bearer invalid-garbage-token")
	ctx := metadata.NewIncomingContext(context.Background(), md)

	_, err := interceptor(ctx, nil, dummyServerInfo("/test.Service/Method"), passThroughHandler)
	if err == nil {
		t.Fatal("expected Unauthenticated error, got nil")
	}
	st, _ := status.FromError(err)
	if st.Code() != codes.Unauthenticated {
		t.Fatalf("expected Unauthenticated, got %v", st.Code())
	}
}

func TestGRPCAuth_JWT_ExpiredToken(t *testing.T) {
	jwtMgr := security.NewJWTManager("test-secret-key-for-grpc", "test-issuer")
	interceptor := grpcAuthInterceptor(jwtMgr, "", true, testLogger())

	// Generate a token that expired 1 hour ago
	token, err := jwtMgr.GenerateTenantToken("tenant-expired", []string{"read"}, nil, -time.Hour)
	if err != nil {
		t.Fatalf("failed to generate token: %v", err)
	}

	md := metadata.Pairs("authorization", "Bearer "+token)
	ctx := metadata.NewIncomingContext(context.Background(), md)

	_, err = interceptor(ctx, nil, dummyServerInfo("/test.Service/Method"), passThroughHandler)
	if err == nil {
		t.Fatal("expected Unauthenticated error for expired token, got nil")
	}
	st, _ := status.FromError(err)
	if st.Code() != codes.Unauthenticated {
		t.Fatalf("expected Unauthenticated, got %v", st.Code())
	}
}

func TestGRPCAuth_JWT_WrongSecret(t *testing.T) {
	serverMgr := security.NewJWTManager("server-secret", "test-issuer")
	clientMgr := security.NewJWTManager("different-secret", "test-issuer")
	interceptor := grpcAuthInterceptor(serverMgr, "", true, testLogger())

	token, err := clientMgr.GenerateTenantToken("tenant-bad", []string{"read"}, nil, time.Hour)
	if err != nil {
		t.Fatalf("failed to generate token: %v", err)
	}

	md := metadata.Pairs("authorization", "Bearer "+token)
	ctx := metadata.NewIncomingContext(context.Background(), md)

	_, err = interceptor(ctx, nil, dummyServerInfo("/test.Service/Method"), passThroughHandler)
	if err == nil {
		t.Fatal("expected Unauthenticated error for wrong secret, got nil")
	}
	st, _ := status.FromError(err)
	if st.Code() != codes.Unauthenticated {
		t.Fatalf("expected Unauthenticated, got %v", st.Code())
	}
}

// ─── API token auth tests ───────────────────────────────────────────────────

func TestGRPCAuth_APIToken_Valid(t *testing.T) {
	interceptor := grpcAuthInterceptor(nil, "my-secret-token", true, testLogger())

	md := metadata.Pairs("authorization", "Bearer my-secret-token")
	ctx := metadata.NewIncomingContext(context.Background(), md)

	resp, err := interceptor(ctx, nil, dummyServerInfo("/test.Service/Method"), passThroughHandler)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	tc, ok := resp.(*security.TenantContext)
	if !ok {
		t.Fatalf("expected *TenantContext, got %T", resp)
	}
	if tc.TenantID != "default" {
		t.Errorf("expected default tenant, got %s", tc.TenantID)
	}
}

func TestGRPCAuth_APIToken_Invalid(t *testing.T) {
	interceptor := grpcAuthInterceptor(nil, "my-secret-token", true, testLogger())

	md := metadata.Pairs("authorization", "Bearer wrong-token")
	ctx := metadata.NewIncomingContext(context.Background(), md)

	_, err := interceptor(ctx, nil, dummyServerInfo("/test.Service/Method"), passThroughHandler)
	if err == nil {
		t.Fatal("expected Unauthenticated error, got nil")
	}
	st, _ := status.FromError(err)
	if st.Code() != codes.Unauthenticated {
		t.Fatalf("expected Unauthenticated, got %v", st.Code())
	}
}

func TestGRPCAuth_APIToken_Missing(t *testing.T) {
	interceptor := grpcAuthInterceptor(nil, "my-secret-token", true, testLogger())

	ctx := context.Background()
	_, err := interceptor(ctx, nil, dummyServerInfo("/test.Service/Method"), passThroughHandler)
	if err == nil {
		t.Fatal("expected Unauthenticated error, got nil")
	}
	st, _ := status.FromError(err)
	if st.Code() != codes.Unauthenticated {
		t.Fatalf("expected Unauthenticated, got %v", st.Code())
	}
}

// ─── No auth configured tests ───────────────────────────────────────────────

func TestGRPCAuth_NoAuth_PassThrough(t *testing.T) {
	interceptor := grpcAuthInterceptor(nil, "", false, testLogger())

	ctx := context.Background()
	resp, err := interceptor(ctx, nil, dummyServerInfo("/test.Service/Method"), passThroughHandler)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	tc, ok := resp.(*security.TenantContext)
	if !ok {
		t.Fatalf("expected *TenantContext, got %T", resp)
	}
	if tc.TenantID != "default" {
		t.Errorf("expected default tenant, got %s", tc.TenantID)
	}
	if !tc.IsAdmin {
		t.Error("expected IsAdmin=true when no auth is configured")
	}
	if !tc.Permissions["read"] || !tc.Permissions["write"] {
		t.Errorf("expected read+write permissions, got %v", tc.Permissions)
	}
}

// ─── Panic recovery test ────────────────────────────────────────────────────

func TestGRPCAuth_PanicRecovery(t *testing.T) {
	interceptor := grpcAuthInterceptor(nil, "", false, testLogger())

	panicHandler := func(ctx context.Context, req any) (any, error) {
		panic("test panic in gRPC handler")
	}

	ctx := context.Background()
	_, err := interceptor(ctx, nil, dummyServerInfo("/test.Service/PanicMethod"), panicHandler)
	if err == nil {
		t.Fatal("expected error from panic recovery, got nil")
	}
	st, ok := status.FromError(err)
	if !ok {
		t.Fatalf("expected gRPC status error, got %v", err)
	}
	if st.Code() != codes.Internal {
		t.Fatalf("expected Internal, got %v", st.Code())
	}
}

// ─── TenantContext injection test ───────────────────────────────────────────

func TestGRPCAuth_TenantContextInjected(t *testing.T) {
	jwtMgr := security.NewJWTManager("test-secret-key-for-grpc", "test-issuer")
	interceptor := grpcAuthInterceptor(jwtMgr, "", true, testLogger())

	token, err := jwtMgr.GenerateTenantToken("acme-corp", []string{"read", "write", "admin"}, []string{"products", "orders"}, time.Hour)
	if err != nil {
		t.Fatalf("failed to generate token: %v", err)
	}

	// Handler that verifies the context has the right TenantContext
	checkHandler := func(ctx context.Context, req any) (any, error) {
		tc, ok := security.GetTenantContextFromContext(ctx)
		if !ok {
			t.Error("TenantContext not found in context")
			return nil, nil
		}
		if tc.TenantID != "acme-corp" {
			t.Errorf("expected acme-corp, got %s", tc.TenantID)
		}
		if !tc.IsAdmin {
			t.Error("expected IsAdmin=true for admin permission")
		}
		if !tc.Collections["products"] || !tc.Collections["orders"] {
			t.Errorf("expected products+orders collections, got %v", tc.Collections)
		}
		return "ok", nil
	}

	md := metadata.Pairs("authorization", "Bearer "+token)
	ctx := metadata.NewIncomingContext(context.Background(), md)

	_, err = interceptor(ctx, nil, dummyServerInfo("/test.Service/Method"), checkHandler)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}
