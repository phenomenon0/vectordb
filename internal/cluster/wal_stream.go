package cluster

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"
)

// ===========================================================================================
// WAL STREAMING FOR REPLICATION
// Replaces snapshot-based replication with incremental WAL streaming
// ===========================================================================================

// WALStream manages streaming WAL entries to replicas
type WALStream struct {
	mu      sync.RWMutex
	entries []WalEntry // In-memory WAL buffer
	nextSeq uint64     // Next sequence number to assign

	maxEntries int    // Maximum entries to keep in memory (default: 10000)
	minSeq     uint64 // Minimum sequence number available
}

// NewWALStream creates a new WAL stream
func NewWALStream() *WALStream {
	return &WALStream{
		entries:    make([]WalEntry, 0, 1000),
		nextSeq:    1,
		minSeq:     1,
		maxEntries: 10000,
	}
}

// Append adds a new entry to the WAL stream
func (ws *WALStream) Append(entry WalEntry) uint64 {
	ws.mu.Lock()
	defer ws.mu.Unlock()

	if entry.Time == 0 {
		entry.Time = time.Now().Unix()
	}
	entry.Seq = ws.nextSeq

	ws.entries = append(ws.entries, entry)
	seq := ws.nextSeq
	ws.nextSeq++

	// Trim old entries if buffer gets too large
	if len(ws.entries) > ws.maxEntries {
		trimCount := len(ws.entries) - ws.maxEntries
		ws.entries = ws.entries[trimCount:]
		ws.minSeq = ws.entries[0].Seq
	}

	return seq
}

// GetSince returns all WAL entries since the given sequence number
func (ws *WALStream) GetSince(sinceSeq uint64) ([]WalEntry, error) {
	ws.mu.RLock()
	defer ws.mu.RUnlock()

	// Check if requested sequence is too old (already trimmed)
	// Allow sinceSeq=0 to mean "all available entries"
	if sinceSeq > 0 && sinceSeq < ws.minSeq && len(ws.entries) > 0 {
		return nil, fmt.Errorf("WAL entries before seq %d have been trimmed (min available: %d)",
			sinceSeq, ws.minSeq)
	}

	// Find start index
	result := make([]WalEntry, 0)
	for _, entry := range ws.entries {
		if entry.Seq > sinceSeq {
			result = append(result, entry)
		}
	}

	return result, nil
}

// GetLatestSeq returns the latest sequence number
func (ws *WALStream) GetLatestSeq() uint64 {
	ws.mu.RLock()
	defer ws.mu.RUnlock()
	return ws.nextSeq - 1
}

// WALStreamResponse is the response format for /wal/stream endpoint
type WALStreamResponse struct {
	Entries   []WalEntry `json:"entries"`
	LatestSeq uint64     `json:"latest_seq"`
	MinSeq    uint64     `json:"min_seq"`
	Count     int        `json:"count"`
}

// HandleWALStream serves WAL entries to replicas (GET /wal/stream?since=N)
func (s *ShardServer) HandleWALStream(w http.ResponseWriter, r *http.Request) {
	// Only primaries can serve WAL stream
	if s.role != RolePrimary {
		http.Error(w, "only primary can serve WAL stream", http.StatusForbidden)
		return
	}

	if err := AuthorizeWALStream(s.store, r); err != nil {
		http.Error(w, err.Error(), http.StatusUnauthorized)
		return
	}

	// Get since parameter
	sinceStr := r.URL.Query().Get("since")
	var since uint64
	if sinceStr != "" {
		fmt.Sscanf(sinceStr, "%d", &since)
	}

	// Get WAL entries
	entries, err := s.walStream.GetSince(since)
	if err != nil {
		http.Error(w, err.Error(), http.StatusGone) // 410 Gone = data too old
		return
	}

	// Build response
	response := WALStreamResponse{
		Entries:   entries,
		LatestSeq: s.walStream.GetLatestSeq(),
		MinSeq:    s.walStream.minSeq,
		Count:     len(entries),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// AuthorizeWALStream protects the WAL stream endpoint to prevent unauthenticated replication.
func AuthorizeWALStream(store Store, r *http.Request) error {
	token := r.Header.Get("Authorization")
	if token == "" {
		token = r.URL.Query().Get("token")
	}

	// If auth is required or tokens are configured, enforce them
	if store.StoreRequireAuth() || store.StoreAPIToken() != "" || store.StoreJWTMgr() != nil {
		if token == "" {
			return fmt.Errorf("unauthorized: missing token")
		}
	}

	apiToken := store.StoreAPIToken()
	if apiToken != "" {
		if token == apiToken || token == "Bearer "+apiToken {
			return nil
		}
		if token != "" {
			return fmt.Errorf("unauthorized")
		}
	}

	jwtMgr := store.StoreJWTMgr()
	if jwtMgr != nil && token != "" {
		jwtToken := strings.TrimPrefix(token, "Bearer ")
		if ctx, err := jwtMgr.ValidateTenantToken(jwtToken); err == nil && ctx.IsAdmin {
			return nil
		}
		return fmt.Errorf("unauthorized")
	}

	if store.StoreRequireAuth() {
		return fmt.Errorf("unauthorized")
	}

	return nil
}

// WALStreamClient pulls WAL entries from primary
type WALStreamClient struct {
	primaryAddr string
	lastSeq     uint64
	httpClient  *http.Client
	authToken   string
}

// NewWALStreamClient creates a new WAL stream client
func NewWALStreamClient(primaryAddr string, authToken string) *WALStreamClient {
	return &WALStreamClient{
		primaryAddr: primaryAddr,
		lastSeq:     0,
		httpClient:  &http.Client{Timeout: 10 * time.Second},
		authToken:   authToken,
	}
}

// PullLatest pulls the latest WAL entries from primary
func (wsc *WALStreamClient) PullLatest() ([]WalEntry, error) {
	url := fmt.Sprintf("%s/wal/stream?since=%d", wsc.primaryAddr, wsc.lastSeq)

	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to build WAL request: %w", err)
	}
	if wsc.authToken != "" {
		req.Header.Set("Authorization", "Bearer "+wsc.authToken)
	}

	resp, err := wsc.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to pull WAL: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusGone {
		// WAL entries too old, need full snapshot
		return nil, fmt.Errorf("WAL gap detected: entries before seq %d no longer available", wsc.lastSeq)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("WAL stream returned status %d", resp.StatusCode)
	}

	var response WALStreamResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("failed to decode WAL response: %w", err)
	}

	return response.Entries, nil
}

// Advance records the latest WAL sequence that was durably applied locally.
func (wsc *WALStreamClient) Advance(seq uint64) {
	if seq > wsc.lastSeq {
		wsc.lastSeq = seq
	}
}

// ApplyEntries applies WAL entries to the local store
func (s *ShardServer) ApplyEntries(entries []WalEntry) error {
	for _, entry := range entries {
		switch entry.Op {
		case "insert":
			// Add to store
			_, err := s.store.Add(entry.Vec, entry.Doc, entry.ID, entry.Meta, entry.Coll, entry.Tenant)
			if err != nil {
				return fmt.Errorf("failed to apply insert (seq %d): %w", entry.Seq, err)
			}

		case "upsert":
			_, err := s.store.Upsert(entry.Vec, entry.Doc, entry.ID, entry.Meta, entry.Coll, entry.Tenant)
			if err != nil {
				return fmt.Errorf("failed to apply upsert (seq %d): %w", entry.Seq, err)
			}

		case "delete":
			// Delete from store
			s.store.DeleteByID(entry.ID)

		default:
			return fmt.Errorf("unknown WAL operation: %s", entry.Op)
		}
	}

	return nil
}
