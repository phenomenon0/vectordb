package main

import (
	"encoding/json"
	"fmt"
	"net/http"
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
	entries []walEntry // In-memory WAL buffer
	nextSeq uint64     // Next sequence number to assign

	maxEntries int      // Maximum entries to keep in memory (default: 10000)
	minSeq     uint64   // Minimum sequence number available
}

// NewWALStream creates a new WAL stream
func NewWALStream() *WALStream {
	return &WALStream{
		entries:    make([]walEntry, 0, 1000),
		nextSeq:    1,
		minSeq:     1,
		maxEntries: 10000,
	}
}

// Append adds a new entry to the WAL stream
func (ws *WALStream) Append(op, id, doc, coll string, vec []float32, meta map[string]string) uint64 {
	ws.mu.Lock()
	defer ws.mu.Unlock()

	entry := walEntry{
		Seq:  ws.nextSeq,
		Op:   op,
		ID:   id,
		Doc:  doc,
		Vec:  vec,
		Meta: meta,
		Coll: coll,
		Time: time.Now().Unix(),
	}

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
func (ws *WALStream) GetSince(sinceSeq uint64) ([]walEntry, error) {
	ws.mu.RLock()
	defer ws.mu.RUnlock()

	// Check if requested sequence is too old (already trimmed)
	// Allow sinceSeq=0 to mean "all available entries"
	if sinceSeq > 0 && sinceSeq < ws.minSeq && len(ws.entries) > 0 {
		return nil, fmt.Errorf("WAL entries before seq %d have been trimmed (min available: %d)",
			sinceSeq, ws.minSeq)
	}

	// Find start index
	result := make([]walEntry, 0)
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
	Entries    []walEntry `json:"entries"`
	LatestSeq  uint64     `json:"latest_seq"`
	MinSeq     uint64     `json:"min_seq"`
	Count      int        `json:"count"`
}

// handleWALStream serves WAL entries to replicas (GET /wal/stream?since=N)
func (s *ShardServer) handleWALStream(w http.ResponseWriter, r *http.Request) {
	// Only primaries can serve WAL stream
	if s.role != RolePrimary {
		http.Error(w, "only primary can serve WAL stream", http.StatusForbidden)
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

// WALStreamClient pulls WAL entries from primary
type WALStreamClient struct {
	primaryAddr string
	lastSeq     uint64
	httpClient  *http.Client
}

// NewWALStreamClient creates a new WAL stream client
func NewWALStreamClient(primaryAddr string) *WALStreamClient {
	return &WALStreamClient{
		primaryAddr: primaryAddr,
		lastSeq:     0,
		httpClient:  &http.Client{Timeout: 10 * time.Second},
	}
}

// PullLatest pulls the latest WAL entries from primary
func (wsc *WALStreamClient) PullLatest() ([]walEntry, error) {
	url := fmt.Sprintf("%s/wal/stream?since=%d", wsc.primaryAddr, wsc.lastSeq)

	resp, err := wsc.httpClient.Get(url)
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

	// Update last seq
	if len(response.Entries) > 0 {
		wsc.lastSeq = response.Entries[len(response.Entries)-1].Seq
	}

	return response.Entries, nil
}

// ApplyEntries applies WAL entries to the local store
func (s *ShardServer) ApplyEntries(entries []walEntry) error {
	for _, entry := range entries {
		switch entry.Op {
		case "insert", "upsert":
			// Add to store
			_, err := s.store.Add(entry.Vec, entry.Doc, entry.ID, entry.Meta, entry.Coll, "")
			if err != nil {
				return fmt.Errorf("failed to apply insert (seq %d): %w", entry.Seq, err)
			}

		case "delete":
			// Delete from store
			s.store.Delete(entry.ID)

		default:
			return fmt.Errorf("unknown WAL operation: %s", entry.Op)
		}
	}

	return nil
}
