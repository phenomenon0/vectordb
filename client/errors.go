package client

import (
	"errors"
	"fmt"
)

// Sentinel errors for typed error handling, matching the TS/Python SDKs.
var (
	ErrTimeout            = errors.New("vectordb: request timed out")
	ErrConnection         = errors.New("vectordb: connection failed")
	ErrCollectionNotFound = errors.New("vectordb: collection not found")
	ErrCollectionExists   = errors.New("vectordb: collection already exists")
	ErrDimensionMismatch  = errors.New("vectordb: dimension mismatch")
	ErrValidation         = errors.New("vectordb: validation error")
	ErrUnauthorized       = errors.New("vectordb: unauthorized")
	ErrForbidden          = errors.New("vectordb: forbidden")
	ErrRateLimited        = errors.New("vectordb: rate limited")
)

// APIError wraps an HTTP error with typed classification.
type APIError struct {
	StatusCode int
	Message    string
	Retryable  bool
	Cause      error // underlying sentinel
}

func (e *APIError) Error() string {
	return fmt.Sprintf("vectordb api %d: %s", e.StatusCode, e.Message)
}

func (e *APIError) Unwrap() error {
	return e.Cause
}

// IsRetryable returns true if the error is transient and the request can be retried.
func IsRetryable(err error) bool {
	var apiErr *APIError
	if errors.As(err, &apiErr) {
		return apiErr.Retryable
	}
	// Connection/timeout errors are retryable
	return errors.Is(err, ErrTimeout) || errors.Is(err, ErrConnection)
}

// retryableStatusCodes matches the TS/Python SDKs.
var retryableStatusCodes = map[int]bool{
	408: true, // Request Timeout
	429: true, // Too Many Requests
	500: true, // Internal Server Error
	502: true, // Bad Gateway
	503: true, // Service Unavailable
	504: true, // Gateway Timeout
}

// classifyHTTPError converts an HTTPError into a typed APIError.
func classifyHTTPError(statusCode int, body string) *APIError {
	ae := &APIError{
		StatusCode: statusCode,
		Message:    body,
		Retryable:  retryableStatusCodes[statusCode],
	}

	switch {
	case statusCode == 401:
		ae.Cause = ErrUnauthorized
	case statusCode == 403:
		ae.Cause = ErrForbidden
	case statusCode == 404:
		ae.Cause = ErrCollectionNotFound
	case statusCode == 409:
		ae.Cause = ErrCollectionExists
	case statusCode == 400 || statusCode == 422:
		ae.Cause = ErrValidation
	case statusCode == 429:
		ae.Cause = ErrRateLimited
		ae.Retryable = true
	default:
		ae.Cause = fmt.Errorf("vectordb: http %d", statusCode)
	}

	return ae
}
