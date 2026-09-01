package apierror

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	vcollection "github.com/phenomenon0/vectordb/internal/collection"
	"github.com/phenomenon0/vectordb/internal/logging"
	"google.golang.org/genproto/googleapis/rpc/errdetails"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// The table is the contract: a client that switches on code must see the
// same HTTP status and gRPC code on every surface, and only rate limits and
// persistence faults are worth retrying.
func TestErrorCodeTable(t *testing.T) {
	cases := []struct {
		code      string
		http      int
		grpc      codes.Code
		retryable bool
	}{
		{CodeInvalidArgument, 400, codes.InvalidArgument, false},
		{CodeNotFound, 404, codes.NotFound, false},
		{CodeAlreadyExists, 409, codes.AlreadyExists, false},
		{CodeUnauthenticated, 401, codes.Unauthenticated, false},
		{CodePermissionDenied, 403, codes.PermissionDenied, false},
		{CodeQuotaExceeded, 409, codes.FailedPrecondition, false},
		{CodePayloadTooLarge, 413, codes.ResourceExhausted, false},
		{CodeRateLimited, 429, codes.ResourceExhausted, true},
		{CodeUnavailable, 503, codes.Unavailable, true},
		{CodeMethodNotAllowed, 405, codes.Unimplemented, false},
		{CodeInternal, 500, codes.Internal, false},
		{CodeEmbeddingMismatch, 409, codes.FailedPrecondition, false},
		{CodeEmbedderUnavailable, 503, codes.Unavailable, true},
	}
	for _, tc := range cases {
		e := New(tc.code, "m")
		if HTTPStatus(tc.code) != tc.http || GRPCCode(tc.code) != tc.grpc || e.Retryable != tc.retryable {
			t.Errorf("%s: got %d/%v/retryable=%v want %d/%v/%v", tc.code, HTTPStatus(tc.code), GRPCCode(tc.code), e.Retryable, tc.http, tc.grpc, tc.retryable)
		}
		if e.Hint == "" || e.Docs != Docs {
			t.Errorf("%s: every error carries a hint and the docs pointer: %+v", tc.code, e)
		}
	}
	if New(CodeRateLimited, "m").RetryAfterMS != 1000 {
		t.Fatalf("rate_limited must tell the client how long to wait")
	}
	if New("no_such_code", "m").Code != CodeInternal {
		t.Fatalf("an unknown code must not leak as a new vocabulary word")
	}
}

// FromEngine must classify wrapped sentinels (the engine always wraps with
// context) so that a permanent limit is never reported as retryable and a
// caller mistake is never a 500.
func TestErrorFromEngineClassifiesWrappedSentinels(t *testing.T) {
	cases := []struct {
		err  error
		code string
	}{
		{fmt.Errorf("%w: top_k must be in [1, 1000]", vcollection.ErrInvalidArgument), CodeInvalidArgument},
		{fmt.Errorf("%w: score_floor must be finite", vcollection.ErrInvalidSearchArgument), CodeInvalidArgument},
		{fmt.Errorf("%w: docs", vcollection.ErrCollectionNotFound), CodeNotFound},
		{fmt.Errorf("%w: 7 in collection docs", vcollection.ErrDocumentNotFound), CodeNotFound},
		{fmt.Errorf("%w: docs", vcollection.ErrCollectionExists), CodeAlreadyExists},
		{fmt.Errorf("%w: document 0 ID 7", vcollection.ErrDocumentExists), CodeAlreadyExists},
		{fmt.Errorf("create: %w", vcollection.ErrTenantLimitExceeded), CodeQuotaExceeded},
		{fmt.Errorf("create: %w", vcollection.ErrCollectionLimitExceeded), CodeQuotaExceeded},
		{fmt.Errorf("search: %w", vcollection.ErrSearchResponseBudgetExceeded), CodePayloadTooLarge},
		{vcollection.ErrDurableStoreClosed, CodeUnavailable},
		{vcollection.ErrDurableStoreFaulted, CodeUnavailable},
		{errors.New("disk on fire"), CodeInternal},
	}
	for _, tc := range cases {
		if got := FromEngine(tc.err, CodeInternal); got.Code != tc.code {
			t.Errorf("%v: got %s want %s", tc.err, got.Code, tc.code)
		}
	}
	if got := FromEngine(errors.New("bad json"), CodeInvalidArgument); got.Code != CodeInvalidArgument {
		t.Fatalf("unclassified errors take the operation's fallback, got %s", got.Code)
	}
	if got := FromEngine(New(CodeNotFound, "already classified"), CodeInternal); got.Code != CodeNotFound {
		t.Fatalf("an *Error passes through unchanged, got %s", got.Code)
	}
	if got := FromEngine(vcollection.ErrTenantLimitExceeded, CodeInternal); got.Retryable {
		t.Fatalf("quota limits are fixed for the process lifetime; reporting them retryable makes every SDK retry forever")
	}
}

// The HTTP projection: JSON body, status from the table, Retry-After on rate
// limits, and the request id the middleware already put on the response so a
// client can quote it without a second header lookup.
func TestErrorWriteHTTPEnvelope(t *testing.T) {
	rec := httptest.NewRecorder()
	rec.Header().Set("X-Request-ID", "req-123")
	WriteHTTP(rec, New(CodeRateLimited, "tenant rate limited"))
	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("status %d", rec.Code)
	}
	if got := rec.Header().Get("Retry-After"); got != "1" {
		t.Fatalf("Retry-After %q, want 1 second", got)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Fatalf("content-type %q", ct)
	}
	var body Error
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("body is not the JSON envelope: %v: %s", err, rec.Body.String())
	}
	if body.Code != CodeRateLimited || body.RequestID != "req-123" || !body.Retryable || body.RetryAfterMS != 1000 || body.Hint == "" || body.Docs != Docs {
		t.Fatalf("envelope %+v", body)
	}

	rec = httptest.NewRecorder()
	WriteHTTP(rec, New(CodeNotFound, "collection not found: docs"))
	if rec.Header().Get("Retry-After") != "" {
		t.Fatalf("non-retryable errors must not advertise Retry-After")
	}
	var m map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &m); err != nil {
		t.Fatal(err)
	}
	if _, ok := m["retryable"]; !ok {
		t.Fatalf("retryable is always present, even when false, so clients never guess: %v", m)
	}
}

// The gRPC projection: the code from the table, ErrorInfo.Reason carrying the
// same code string as HTTP, hint and request id in its metadata, and
// RetryInfo only when retrying makes sense.
func TestErrorGRPCStatusDetails(t *testing.T) {
	ctx := context.WithValue(context.Background(), logging.RequestIDKey, "req-abc")
	err := New(CodeRateLimited, "tenant rate limited").GRPC(ctx)
	st, ok := status.FromError(err)
	if !ok || st.Code() != codes.ResourceExhausted {
		t.Fatalf("status %v", err)
	}
	var info *errdetails.ErrorInfo
	var retry *errdetails.RetryInfo
	for _, d := range st.Details() {
		switch v := d.(type) {
		case *errdetails.ErrorInfo:
			info = v
		case *errdetails.RetryInfo:
			retry = v
		}
	}
	if info == nil || info.Reason != CodeRateLimited || info.Domain != "deepdata" {
		t.Fatalf("ErrorInfo %+v", info)
	}
	if info.Metadata["hint"] == "" || info.Metadata["request_id"] != "req-abc" || info.Metadata["docs"] != Docs {
		t.Fatalf("ErrorInfo metadata %v", info.Metadata)
	}
	if retry == nil || retry.RetryDelay.AsDuration().Milliseconds() != 1000 {
		t.Fatalf("RetryInfo %+v", retry)
	}

	st, _ = status.FromError(New(CodeQuotaExceeded, "tenant limit").GRPC(context.Background()))
	if st.Code() != codes.FailedPrecondition {
		t.Fatalf("quota is a precondition failure, not resource exhaustion: %v", st.Code())
	}
	for _, d := range st.Details() {
		if _, isRetry := d.(*errdetails.RetryInfo); isRetry {
			t.Fatalf("non-retryable errors must not carry RetryInfo")
		}
	}
}
