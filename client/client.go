package client

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Client is a lightweight HTTP wrapper for the vectordb service.
// It exposes typed helpers but keeps a generic Do for extensibility.
type Client struct {
	baseURL string
	http    *http.Client
	token   string
	headers http.Header
}

// Option mutates client configuration.
type Option func(*Client)

// WithHTTPClient sets a custom http.Client (useful for custom transports/timeouts).
func WithHTTPClient(hc *http.Client) Option {
	return func(c *Client) {
		if hc != nil {
			c.http = hc
		}
	}
}

// WithToken sets the bearer token (Authorization: Bearer <token>).
func WithToken(token string) Option {
	return func(c *Client) {
		c.token = token
	}
}

// WithHeader adds an extra header to every request.
func WithHeader(key, value string) Option {
	return func(c *Client) {
		if key == "" {
			return
		}
		c.headers.Add(key, value)
	}
}

// WithTimeout sets the client's timeout (creates a shallow copy of the default client).
func WithTimeout(d time.Duration) Option {
	return func(c *Client) {
		if c.http == nil {
			c.http = &http.Client{Timeout: d}
			return
		}
		cp := *c.http
		cp.Timeout = d
		c.http = &cp
	}
}

// New constructs a client. baseURL may be empty (defaults to http://localhost:8080).
func New(baseURL string, opts ...Option) *Client {
	if baseURL == "" {
		baseURL = "http://localhost:8080"
	}
	c := &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		http:    &http.Client{Timeout: 15 * time.Second},
		headers: make(http.Header),
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// InsertRequest mirrors POST /insert.
type InsertRequest struct {
	ID         string            `json:"id,omitempty"`
	Doc        string            `json:"doc"`
	Meta       map[string]string `json:"meta,omitempty"`
	Upsert     bool              `json:"upsert,omitempty"`
	Collection string            `json:"collection,omitempty"`
}

type InsertResponse struct {
	ID string `json:"id"`
}

// BatchInsertRequest mirrors POST /batch_insert.
type BatchInsertRequest struct {
	Docs   []BatchDoc `json:"docs"`
	Upsert bool       `json:"upsert,omitempty"`
}

type BatchDoc struct {
	ID         string            `json:"id,omitempty"`
	Doc        string            `json:"doc"`
	Meta       map[string]string `json:"meta,omitempty"`
	Collection string            `json:"collection,omitempty"`
}

type BatchInsertResponse struct {
	IDs []string `json:"ids"`
}

// QueryRequest mirrors POST /query.
type QueryRequest struct {
	Query       string              `json:"query"`
	TopK        int                 `json:"top_k,omitempty"`
	Mode        string              `json:"mode,omitempty"` // "ann" (default) or "scan"
	Meta        map[string]string   `json:"meta,omitempty"`
	MetaAny     []map[string]string `json:"meta_any,omitempty"`
	MetaNot     map[string]string   `json:"meta_not,omitempty"`
	IncludeMeta bool                `json:"include_meta,omitempty"`
	Collection  string              `json:"collection,omitempty"`
	Offset      int                 `json:"offset,omitempty"`
	Limit       int                 `json:"limit,omitempty"`
}

type QueryResponse struct {
	IDs    []string            `json:"ids"`
	Docs   []string            `json:"docs"`
	Scores []float32           `json:"scores"`
	Stats  string              `json:"stats"`
	Meta   []map[string]string `json:"meta,omitempty"`
}

// DeleteRequest mirrors POST /delete.
type DeleteRequest struct {
	ID string `json:"id"`
}

type DeleteResponse struct {
	Deleted string `json:"deleted"`
}

// HealthResponse mirrors GET /health.
type HealthResponse struct {
	OK            bool   `json:"ok"`
	Total         int    `json:"total"`
	Active        int    `json:"active"`
	Deleted       int    `json:"deleted"`
	HNSWIDs       int    `json:"hnsw_ids"`
	Checksum      string `json:"checksum"`
	WALBytes      int64  `json:"wal_bytes"`
	IndexBytes    int64  `json:"index_bytes,omitempty"`
	SnapshotAgeMS int64  `json:"snapshot_age_ms,omitempty"`
	WALAgeMS      int64  `json:"wal_age_ms,omitempty"`
}

// HTTPError is returned for non-2xx responses.
type HTTPError struct {
	StatusCode int
	Body       string
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("http %d: %s", e.StatusCode, e.Body)
}

// Insert calls POST /insert.
func (c *Client) Insert(ctx context.Context, req InsertRequest) (*InsertResponse, error) {
	var resp InsertResponse
	if err := c.doJSON(ctx, http.MethodPost, "/insert", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// BatchInsert calls POST /batch_insert.
func (c *Client) BatchInsert(ctx context.Context, req BatchInsertRequest) (*BatchInsertResponse, error) {
	var resp BatchInsertResponse
	if err := c.doJSON(ctx, http.MethodPost, "/batch_insert", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// Query calls POST /query.
func (c *Client) Query(ctx context.Context, req QueryRequest) (*QueryResponse, error) {
	var resp QueryResponse
	if err := c.doJSON(ctx, http.MethodPost, "/query", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// Delete calls POST /delete.
func (c *Client) Delete(ctx context.Context, req DeleteRequest) (*DeleteResponse, error) {
	var resp DeleteResponse
	if err := c.doJSON(ctx, http.MethodPost, "/delete", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// Health calls GET /health.
func (c *Client) Health(ctx context.Context) (*HealthResponse, error) {
	var resp HealthResponse
	if err := c.doJSON(ctx, http.MethodGet, "/health", nil, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// doJSON is the small generic core used by helpers.
func (c *Client) doJSON(ctx context.Context, method, path string, in any, out any) error {
	url := c.baseURL + path

	var body io.Reader
	if method == http.MethodPost || method == http.MethodPut || method == http.MethodPatch {
		b, err := json.Marshal(in)
		if err != nil {
			return fmt.Errorf("encode request: %w", err)
		}
		body = bytes.NewReader(b)
	}

	req, err := http.NewRequestWithContext(ctx, method, url, body)
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	for k, vals := range c.headers {
		for _, v := range vals {
			req.Header.Add(k, v)
		}
	}
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}

	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("do request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return &HTTPError{StatusCode: resp.StatusCode, Body: string(respBody)}
	}
	if out == nil || len(respBody) == 0 {
		return nil
	}
	if err := json.Unmarshal(respBody, out); err != nil {
		return fmt.Errorf("decode response: %w", err)
	}
	return nil
}
