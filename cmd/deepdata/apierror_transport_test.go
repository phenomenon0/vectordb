package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	deepdatav3 "github.com/phenomenon0/vectordb/api/gen/deepdata/v3"
	"github.com/phenomenon0/vectordb/internal/apierror"
	vcollection "github.com/phenomenon0/vectordb/internal/collection"
	"github.com/phenomenon0/vectordb/internal/logging"
	"google.golang.org/genproto/googleapis/rpc/errdetails"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

// TestCanonicalHTTPErrorEnvelope proves every canonical HTTP failure is the
// structured envelope (internal/collection/API.md, section Errors): an agent
// branches on the code, reads the hint to pick its next call, and gets back
// the request id it sent to correlate with server logs. Plain text gave it
// none of these.
func TestCanonicalHTTPErrorEnvelope(t *testing.T) {
	handler := newCanonicalSurfaceTestHandler(t)
	if response := canonicalHTTPCreateCollection(t, handler, "acme", "docs", ""); response.Code/100 != 2 {
		t.Fatalf("create collection returned %d: %s", response.Code, response.Body.String())
	}

	cases := []struct {
		name, method, path, body string
		status                   int
		code, mention            string
	}{
		{"missing collection", http.MethodGet, "/v3/tenants/acme/collections/missing", "", http.StatusNotFound, apierror.CodeNotFound, "missing"},
		{"missing document", http.MethodGet, "/v3/tenants/acme/collections/docs/docs/99", "", http.StatusNotFound, apierror.CodeNotFound, "99"},
		{"duplicate collection", http.MethodPost, "/v3/tenants/acme/collections", `{"name":"docs","fields":[{"name":"embedding","type":0,"dim":2,"index":{"type":"flat"}}]}`, http.StatusConflict, apierror.CodeAlreadyExists, "docs"},
		{"engine validation", http.MethodPost, "/v3/tenants/acme/collections/docs/search", `{"queries":{"embedding":[1,0]},"top_k":0}`, http.StatusBadRequest, apierror.CodeInvalidArgument, "top_k"},
		{"unknown route", http.MethodGet, "/v3/nope", "", http.StatusNotFound, apierror.CodeNotFound, "/v3/nope"},
	}
	for _, tc := range cases {
		request := httptest.NewRequest(tc.method, tc.path, strings.NewReader(tc.body))
		request.Header.Set("Content-Type", "application/json")
		request.Header.Set("X-Request-ID", "agent-req-7")
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		if response.Code != tc.status {
			t.Fatalf("%s: status %d, want %d: %s", tc.name, response.Code, tc.status, response.Body.String())
		}
		if ct := response.Header().Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
			t.Fatalf("%s: content-type %q, want application/json", tc.name, ct)
		}
		var envelope apierror.Error
		if err := json.Unmarshal(response.Body.Bytes(), &envelope); err != nil {
			t.Fatalf("%s: body is not the envelope: %v: %s", tc.name, err, response.Body.String())
		}
		if envelope.Code != tc.code || !strings.Contains(envelope.Message, tc.mention) {
			t.Fatalf("%s: envelope %+v, want code %s mentioning %q", tc.name, envelope, tc.code, tc.mention)
		}
		if envelope.RequestID != "agent-req-7" || envelope.Docs != apierror.Docs || envelope.Retryable {
			t.Fatalf("%s: envelope %+v must echo the request id, point at the docs and be non-retryable", tc.name, envelope)
		}
	}
}

// TestCanonicalHTTPRateLimitedErrorCarriesRetryAfter: a 429 is the one
// retryable 4xx, and the envelope has to say when — both as the standard
// Retry-After header and as retry_after_ms for clients that only read JSON.
func TestCanonicalHTTPRateLimitedErrorCarriesRetryAfter(t *testing.T) {
	t.Setenv("JWT_SECRET", "")
	t.Setenv("API_TOKEN", "")
	t.Setenv("REQUIRE_AUTH", "0")
	t.Setenv("TRUST_PROXY", "0")
	rt := newServerRuntime()
	rt.requireAuth = true
	rt.apiToken = "envelope-static-token"
	rt.rl = newRateLimiter(100, 100, 100, time.Hour)
	rt.canonicalTenantRL = newRateLimiter(100, 100, 100, time.Hour)
	rt.authFailureRL = newAuthFailureLimiter(1, 1, 100, time.Hour)
	handler, collections := newCanonicalHTTPHandler(rt, NewHashEmbedder(4), nil, filepath.Join(t.TempDir(), "index.gob"))
	t.Cleanup(func() { _ = collections.Close() })

	if response := authThrottleHTTPRequest(handler, "wrong"); response.Code != http.StatusUnauthorized {
		t.Fatalf("first failure returned %d: %s", response.Code, response.Body.String())
	}
	response := authThrottleHTTPRequest(handler, "wrong-again")
	if response.Code != http.StatusTooManyRequests {
		t.Fatalf("throttled failure returned %d: %s", response.Code, response.Body.String())
	}
	if got := response.Header().Get("Retry-After"); got != "1" {
		t.Fatalf("Retry-After = %q, want 1", got)
	}
	var envelope apierror.Error
	if err := json.Unmarshal(response.Body.Bytes(), &envelope); err != nil {
		t.Fatalf("body is not the envelope: %v: %s", err, response.Body.String())
	}
	if envelope.Code != apierror.CodeRateLimited || !envelope.Retryable || envelope.RetryAfterMS != 1000 {
		t.Fatalf("envelope %+v, want rate_limited, retryable, retry_after_ms 1000", envelope)
	}
}

// TestCanonicalGRPCErrorDetails proves the gRPC surface carries the same
// envelope as HTTP inside status details: ErrorInfo.Reason is the code an
// agent branches on, its metadata carries the hint, docs pointer and the
// request id, and RetryInfo appears only when a retry can succeed.
func TestCanonicalGRPCErrorDetails(t *testing.T) {
	server := &CollectionGRPCServer{tenants: vcollection.NewTenantManager(""), persistenceHealth: func() error { return nil }}
	ctx := context.WithValue(canonicalGRPCAdminContext("acme"), logging.RequestIDKey, "agent-req-9")

	_, err := server.Search(ctx, &deepdatav3.SearchRequest{TenantId: "acme", Collection: "missing", TopK: 1})
	st := status.Convert(err)
	if st.Code() != codes.NotFound {
		t.Fatalf("search on a missing collection = %v, want NotFound", err)
	}
	info := errorInfoOf(t, st)
	if info.Reason != apierror.CodeNotFound || info.Domain != "deepdata" {
		t.Fatalf("ErrorInfo = %+v, want reason not_found in domain deepdata", info)
	}
	if info.Metadata["request_id"] != "agent-req-9" || info.Metadata["docs"] != apierror.Docs {
		t.Fatalf("ErrorInfo metadata = %v, want the request id and docs pointer", info.Metadata)
	}
	if retryInfoOf(st) != nil {
		t.Fatal("a missing collection is not retryable; RetryInfo must be absent")
	}
}

// TestCanonicalGRPCInterceptorErrorPlumbing: the caller's x-request-id
// reaches the handler context (so every apierror on the call quotes it), one
// is minted when absent, and a throttled call carries RetryInfo so a client
// backs off instead of hammering the auth limiter.
func TestCanonicalGRPCInterceptorErrorPlumbing(t *testing.T) {
	t.Setenv("TRUST_PROXY", "0")
	interceptor := grpcAuthInterceptorWithRateLimiters(nil, "static-token", true, testLogger(), nil, newAuthFailureLimiter(1, 1, 100, time.Hour))
	info := dummyServerInfo("/deepdata.v3.DeepData/GetTenantInfo")
	peerCtx := authThrottleGRPCContext("")
	withMD := func(pairs ...string) context.Context {
		return metadata.NewIncomingContext(peerCtx, metadata.Pairs(pairs...))
	}
	var seen string
	capture := func(ctx context.Context, req any) (any, error) {
		seen, _ = ctx.Value(logging.RequestIDKey).(string)
		return nil, nil
	}

	if _, err := interceptor(withMD("authorization", "Bearer static-token", "x-request-id", " agent-req-3 "), nil, info, capture); err != nil {
		t.Fatalf("authenticated call failed: %v", err)
	}
	if seen != "agent-req-3" {
		t.Fatalf("handler saw request id %q, want the caller's agent-req-3", seen)
	}
	if _, err := interceptor(withMD("authorization", "Bearer static-token"), nil, info, capture); err != nil {
		t.Fatalf("authenticated call failed: %v", err)
	}
	if seen == "" {
		t.Fatal("a call without x-request-id must still get a minted request id")
	}

	if _, err := interceptor(withMD("authorization", "Bearer wrong"), nil, info, passThroughHandler); status.Code(err) != codes.Unauthenticated {
		t.Fatalf("first bad token = %v, want Unauthenticated", err)
	}
	_, err := interceptor(withMD("authorization", "Bearer wrong"), nil, info, passThroughHandler)
	st := status.Convert(err)
	if st.Code() != codes.ResourceExhausted {
		t.Fatalf("throttled call = %v, want ResourceExhausted", err)
	}
	if reason := errorInfoOf(t, st).Reason; reason != apierror.CodeRateLimited {
		t.Fatalf("ErrorInfo.Reason = %q, want rate_limited", reason)
	}
	retry := retryInfoOf(st)
	if retry == nil || retry.GetRetryDelay().AsDuration() != time.Second {
		t.Fatalf("RetryInfo = %v, want a one second delay", retry)
	}
}

func errorInfoOf(t *testing.T, st *status.Status) *errdetails.ErrorInfo {
	t.Helper()
	for _, detail := range st.Details() {
		if info, ok := detail.(*errdetails.ErrorInfo); ok {
			return info
		}
	}
	t.Fatalf("status %v carries no ErrorInfo detail", st.Err())
	return nil
}

func retryInfoOf(st *status.Status) *errdetails.RetryInfo {
	for _, detail := range st.Details() {
		if info, ok := detail.(*errdetails.RetryInfo); ok {
			return info
		}
	}
	return nil
}
