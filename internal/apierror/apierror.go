// Package apierror is the one error vocabulary of the API. An engine error is
// classified once, here, into a code with a hint and a docs pointer; the HTTP,
// gRPC and MCP surfaces only project the same Error. Gate CTL-01.
package apierror

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strconv"
	"time"

	vcollection "github.com/phenomenon0/vectordb/internal/collection"
	"github.com/phenomenon0/vectordb/internal/logging"
	"google.golang.org/genproto/googleapis/rpc/errdetails"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/protoadapt"
	"google.golang.org/protobuf/types/known/durationpb"
)

// Docs is the pointer every envelope carries: the error section of the contract.
const Docs = "internal/collection/API.md#errors"

// Codes. The table in API.md#errors is the same list; keep them in step.
const (
	CodeInvalidArgument  = "invalid_argument"
	CodeNotFound         = "not_found"
	CodeAlreadyExists    = "already_exists"
	CodeUnauthenticated  = "unauthenticated"
	CodePermissionDenied = "permission_denied"
	CodeQuotaExceeded    = "quota_exceeded"
	CodePayloadTooLarge  = "payload_too_large"
	CodeRateLimited      = "rate_limited"
	CodeUnavailable      = "unavailable"
	CodeMethodNotAllowed = "method_not_allowed"
	CodeInternal         = "internal"
)

// Error is the wire envelope. The same struct is the HTTP JSON body, the
// gRPC ErrorInfo metadata and the MCP structuredContent.
type Error struct {
	Code         string `json:"code"`
	Message      string `json:"message"`
	Hint         string `json:"hint,omitempty"`
	Field        string `json:"field,omitempty"`
	RequestID    string `json:"request_id,omitempty"`
	Retryable    bool   `json:"retryable"`
	RetryAfterMS int64  `json:"retry_after_ms,omitempty"`
	Docs         string `json:"docs"`
}

func (e *Error) Error() string { return e.Code + ": " + e.Message }

type spec struct {
	http      int
	grpc      codes.Code
	retryable bool
	hint      string
}

var specs = map[string]spec{
	CodeInvalidArgument:  {http.StatusBadRequest, codes.InvalidArgument, false, "fix the named value and resend; the limits are in the contract"},
	CodeNotFound:         {http.StatusNotFound, codes.NotFound, false, "list the tenant's collections to see what exists; document ids are the ones you inserted"},
	CodeAlreadyExists:    {http.StatusConflict, codes.AlreadyExists, false, "use the existing collection or choose another name; there is no create-or-get"},
	CodeUnauthenticated:  {http.StatusUnauthorized, codes.Unauthenticated, false, "send Authorization: Bearer <token> (the API_TOKEN or a JWT from cmd/gentoken)"},
	CodePermissionDenied: {http.StatusForbidden, codes.PermissionDenied, false, "the token lacks the permission or collection scope for this call; mint one with the needed read/write/admin claim"},
	CodeQuotaExceeded:    {http.StatusConflict, codes.FailedPrecondition, false, "a fixed deployment limit (max tenants or max collections) is reached; delete a collection or raise the limit; retrying does not help"},
	CodePayloadTooLarge:  {http.StatusRequestEntityTooLarge, codes.ResourceExhausted, false, "send fewer or smaller documents, or lower top_k and drop include_vectors"},
	CodeRateLimited:      {http.StatusTooManyRequests, codes.ResourceExhausted, true, "wait retry_after_ms and retry the same request"},
	CodeUnavailable:      {http.StatusServiceUnavailable, codes.Unavailable, true, "the server is refusing work (persistence fault or shutdown); check GET /readyz and retry later"},
	CodeMethodNotAllowed: {http.StatusMethodNotAllowed, codes.Unimplemented, false, "the path exists but not for this method; the route table is in the contract"},
	CodeInternal:         {http.StatusInternalServerError, codes.Internal, false, "unexpected server fault; report it with the request_id"},
}

// New builds an Error for a known code. An unknown code is an internal error.
func New(code, message string) *Error {
	sp, ok := specs[code]
	if !ok {
		code, sp = CodeInternal, specs[CodeInternal]
	}
	e := &Error{Code: code, Message: message, Hint: sp.hint, Retryable: sp.retryable, Docs: Docs}
	if code == CodeRateLimited {
		e.RetryAfterMS = 1000
	}
	return e
}

// FromEngine is the single errors.Is table from engine sentinels to codes.
// Errors no sentinel claims take fallback (the operation's default code).
func FromEngine(err error, fallback string) *Error {
	var e *Error
	if errors.As(err, &e) {
		return e
	}
	switch {
	case errors.Is(err, vcollection.ErrInvalidArgument), errors.Is(err, vcollection.ErrInvalidSearchArgument):
		return New(CodeInvalidArgument, err.Error())
	case errors.Is(err, vcollection.ErrCollectionNotFound), errors.Is(err, vcollection.ErrDocumentNotFound):
		return New(CodeNotFound, err.Error())
	case errors.Is(err, vcollection.ErrCollectionExists), errors.Is(err, vcollection.ErrDocumentExists):
		return New(CodeAlreadyExists, err.Error())
	case errors.Is(err, vcollection.ErrTenantLimitExceeded), errors.Is(err, vcollection.ErrCollectionLimitExceeded):
		return New(CodeQuotaExceeded, err.Error())
	case errors.Is(err, vcollection.ErrSearchResponseBudgetExceeded):
		return New(CodePayloadTooLarge, err.Error())
	case errors.Is(err, vcollection.ErrDurableStoreClosed), errors.Is(err, vcollection.ErrDurableStoreFaulted):
		return New(CodeUnavailable, "collection persistence unavailable")
	}
	return New(fallback, err.Error())
}

// HTTPStatus and GRPCCode expose the mapping for tests and callers that only
// need the status.
func HTTPStatus(code string) int { return lookup(code).http }

func GRPCCode(code string) codes.Code { return lookup(code).grpc }

func lookup(code string) spec {
	if sp, ok := specs[code]; ok {
		return sp
	}
	return specs[CodeInternal]
}

// WriteHTTP writes the envelope as JSON. The request id comes from the
// X-Request-ID response header the middleware set before the handler ran.
func WriteHTTP(w http.ResponseWriter, e *Error) {
	if e.RequestID == "" {
		e.RequestID = w.Header().Get("X-Request-ID")
	}
	h := w.Header()
	h.Set("Content-Type", "application/json")
	h.Set("X-Content-Type-Options", "nosniff")
	h.Del("Content-Length")
	if e.RetryAfterMS > 0 {
		h.Set("Retry-After", strconv.FormatInt((e.RetryAfterMS+999)/1000, 10))
	}
	w.WriteHeader(HTTPStatus(e.Code))
	_ = json.NewEncoder(w).Encode(e)
}

// GRPC returns the status error for e: the code from the table, the message,
// an ErrorInfo detail carrying code, hint, field, docs and request id, and a
// RetryInfo detail when the error is retryable.
func (e *Error) GRPC(ctx context.Context) error {
	if e.RequestID == "" && ctx != nil {
		if id, ok := ctx.Value(logging.RequestIDKey).(string); ok {
			e.RequestID = id
		}
	}
	st := status.New(GRPCCode(e.Code), e.Message)
	info := &errdetails.ErrorInfo{Reason: e.Code, Domain: "deepdata", Metadata: map[string]string{"docs": e.Docs}}
	for k, v := range map[string]string{"hint": e.Hint, "field": e.Field, "request_id": e.RequestID} {
		if v != "" {
			info.Metadata[k] = v
		}
	}
	details := []protoadapt.MessageV1{info}
	if e.RetryAfterMS > 0 {
		details = append(details, &errdetails.RetryInfo{RetryDelay: durationpb.New(time.Duration(e.RetryAfterMS) * time.Millisecond)})
	}
	if withDetails, err := st.WithDetails(details...); err == nil {
		st = withDetails
	}
	return st.Err()
}
