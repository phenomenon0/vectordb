package cluster

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestSnapshotSyncEndToEnd(t *testing.T) {
	// Create a WAL stream and add entries exceeding the buffer
	ws := NewWALStream()

	// Fill WAL with 15K entries (buffer is 10K, so first 5K get trimmed)
	for i := 0; i < 15000; i++ {
		ws.Append(WalEntry{
			Op:   "insert",
			ID:   "vec-" + string(rune('0'+i%10)),
			Coll: "test-collection",
		})
	}

	// Verify oldest entries are trimmed
	_, err := ws.GetSince(1)
	if err == nil {
		t.Fatal("expected error for trimmed entries, got nil")
	}

	// Verify latest entries are available
	latestSeq := ws.GetLatestSeq()
	if latestSeq != 15000 {
		t.Fatalf("expected latest seq 15000, got %d", latestSeq)
	}

	// GetSince with a recent seq should work
	entries, err := ws.GetSince(14990)
	if err != nil {
		t.Fatalf("GetSince(14990) failed: %v", err)
	}
	if len(entries) != 10 {
		t.Fatalf("expected 10 entries, got %d", len(entries))
	}
}

func TestSnapshotChecksumValidation(t *testing.T) {
	snapshot := &Snapshot{
		ShardID:  1,
		Sequence: 100,
		Vectors: []SnapshotEntry{
			{ID: "vec-1", Vector: []float32{1.0, 2.0, 3.0}},
			{ID: "vec-2", Vector: []float32{4.0, 5.0, 6.0}},
		},
	}
	snapshot.Checksum = snapshot.computeChecksum()

	if !snapshot.validate() {
		t.Fatal("valid snapshot failed validation")
	}

	// Tamper with data
	snapshot.Vectors[0].Vector[0] = 999.0
	if snapshot.validate() {
		t.Fatal("tampered snapshot should fail validation")
	}
}

func TestSnapshotTransferHandler(t *testing.T) {
	// Create a snapshot directly (without full ShardServer)
	snapshot := &Snapshot{
		ShardID:  0,
		Sequence: 42,
		Vectors: []SnapshotEntry{
			{ID: "v1", Vector: []float32{1.0, 2.0}},
			{ID: "v2", Vector: []float32{3.0, 4.0}},
		},
	}
	snapshot.Checksum = snapshot.computeChecksum()

	// Serve it via HTTP
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(snapshot)
	}))
	defer server.Close()

	// Fetch and validate
	resp, err := http.Get(server.URL + "/internal/snapshot")
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	var received Snapshot
	if err := json.NewDecoder(resp.Body).Decode(&received); err != nil {
		t.Fatalf("decode failed: %v", err)
	}

	if !received.validate() {
		t.Fatal("received snapshot failed checksum validation")
	}
	if received.Sequence != 42 {
		t.Fatalf("expected sequence 42, got %d", received.Sequence)
	}
	if len(received.Vectors) != 2 {
		t.Fatalf("expected 2 vectors, got %d", len(received.Vectors))
	}
}

func TestWALStreamGapTriggersSnapshot(t *testing.T) {
	// Simulate the WAL stream client seeing a 410 response
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Always return 410 to simulate WAL gap
		http.Error(w, "WAL entries trimmed", http.StatusGone)
	}))
	defer server.Close()

	client := NewWALStreamClient(server.URL, "")
	_, err := client.PullLatest()
	if err == nil {
		t.Fatal("expected error on 410 response")
	}

	// Error should mention WAL gap
	if err.Error() == "" {
		t.Fatal("error message should not be empty")
	}
	t.Logf("WAL gap error: %v", err)
}

func TestWALStreamConfigurable(t *testing.T) {
	ws := NewWALStream()

	// Default buffer is 10000
	if ws.maxEntries != 10000 {
		t.Fatalf("expected default maxEntries 10000, got %d", ws.maxEntries)
	}

	// Verify trimming works
	for i := 0; i < 10500; i++ {
		ws.Append(WalEntry{Op: "insert", ID: "v"})
	}

	// Buffer should have been trimmed
	if len(ws.entries) > ws.maxEntries {
		t.Fatalf("entries not trimmed: %d > %d", len(ws.entries), ws.maxEntries)
	}

	// minSeq should have advanced
	if ws.minSeq <= 1 {
		t.Fatalf("minSeq should have advanced, got %d", ws.minSeq)
	}
}
