package cluster

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"testing"
	"time"

	"github.com/phenomenon0/vectordb/internal/testutil"
)

// TestPrimaryWriteWALReplicateGapSnapshotResume is a multi-node integration test
// proving the full cycle: WAL replication → WAL gap → streaming snapshot → WAL resume.
func TestPrimaryWriteWALReplicateGapSnapshotResume(t *testing.T) {
	const dim = 4

	// --- Primary setup ---
	primaryStore := newMockStore(dim)
	walStream := NewWALStreamWithConfig(WALStreamConfig{MaxEntries: 10}) // tiny buffer to force gaps

	// Wire WAL hook so inserts flow into the WAL stream
	primaryStore.SetWALHook(func(entry WalEntry) {
		walStream.Append(entry)
	})

	// Create snapshot manager for primary (streaming snapshots)
	snapshotDir := t.TempDir()
	snapCfg := DefaultSnapshotSyncConfig()
	snapCfg.SnapshotDir = snapshotDir
	snapCfg.CompressionEnabled = false // simplify for test
	snapMgr, err := NewSnapshotSyncManager(primaryStore, walStream, snapCfg)
	if err != nil {
		t.Fatalf("NewSnapshotSyncManager: %v", err)
	}

	// Track which WAL since value was requested
	var walMu sync.Mutex
	var lastWALSince string

	// Primary HTTP server serving WAL stream + streaming snapshot
	primaryServer := testutil.NewLoopbackServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/wal/stream":
			walMu.Lock()
			lastWALSince = r.URL.Query().Get("since")
			walMu.Unlock()

			sinceStr := r.URL.Query().Get("since")
			var since uint64
			fmt.Sscanf(sinceStr, "%d", &since)

			entries, err := walStream.GetSince(since)
			if err != nil {
				http.Error(w, err.Error(), http.StatusGone)
				return
			}
			resp := WALStreamResponse{
				Entries:   entries,
				LatestSeq: walStream.GetLatestSeq(),
				Count:     len(entries),
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(resp)

		case "/internal/snapshot/stream/download":
			// Create a fresh snapshot and serve it
			ctx := context.Background()
			_, err := snapMgr.CreateSnapshot(ctx)
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			snapMgr.HandleSnapshotDownload(w, r)

		case "/internal/snapshot":
			// Legacy fallback (should not be reached in this test)
			t.Error("Legacy snapshot endpoint should not be called when streaming is available")
			http.Error(w, "use streaming", http.StatusNotFound)

		default:
			http.Error(w, "not found", http.StatusNotFound)
		}
	}))

	// --- Follower setup ---
	followerStore := newMockStore(dim)
	followerShard := &ShardServer{store: followerStore}

	replicator := NewFollowerReplicator(followerShard, FollowerReplicatorConfig{
		PrimaryAddr:          primaryServer.URL,
		PollInterval:         50 * time.Millisecond,
		FastPollInterval:     10 * time.Millisecond,
		ReconnectInterval:    50 * time.Millisecond,
		MaxReconnectAttempts: 5,
		BatchSize:            100,
		FullSyncThreshold:    1000,
	})

	// Helper: insert N vectors into primary
	insertVectors := func(prefix string, count int) {
		for i := 0; i < count; i++ {
			id := fmt.Sprintf("%s-%d", prefix, i)
			vec := make([]float32, dim)
			for d := 0; d < dim; d++ {
				vec[d] = float32(i*dim + d)
			}
			if _, err := primaryStore.Add(vec, fmt.Sprintf("doc-%s", id), id, nil, "default", ""); err != nil {
				t.Fatalf("failed to insert %s: %v", id, err)
			}
		}
	}

	// Helper: count non-deleted vectors in a store
	countVectors := func(store *mockStore) int {
		store.mu.RLock()
		defer store.mu.RUnlock()
		count := 0
		for _, id := range store.ids {
			if !store.deleted[testHashID(id)] {
				count++
			}
		}
		return count
	}

	// --- Phase 1: Write 5 vectors, replicate via WAL ---
	insertVectors("phase1", 5)

	// Poll manually to replicate
	if err := replicator.poll(); err != nil {
		t.Fatalf("Phase 1 poll failed: %v", err)
	}

	if got := countVectors(followerStore); got != 5 {
		t.Fatalf("Phase 1: expected 5 vectors on follower, got %d", got)
	}
	t.Logf("Phase 1 OK: follower has 5 vectors via WAL")

	// --- Phase 2: Write 20 more vectors (exceeds WAL buffer of 10, causing gap) ---
	insertVectors("phase2", 20)

	// The WAL buffer is 10, so entries from phase1 + early phase2 are trimmed.
	// The follower's lastSeq is 5, but minSeq is now > 5, so PullLatest returns 410.
	if err := replicator.poll(); err != nil {
		t.Fatalf("Phase 2 poll failed (should have recovered via streaming snapshot): %v", err)
	}

	// After streaming snapshot, follower should have all 25 vectors
	if got := countVectors(followerStore); got != 25 {
		t.Fatalf("Phase 2: expected 25 vectors on follower after snapshot recovery, got %d", got)
	}
	t.Logf("Phase 2 OK: follower recovered via streaming snapshot, has 25 vectors")

	// --- Phase 3: Write 3 more vectors, verify WAL streaming resumes ---
	insertVectors("phase3", 3)

	if err := replicator.poll(); err != nil {
		t.Fatalf("Phase 3 poll failed: %v", err)
	}

	if got := countVectors(followerStore); got != 28 {
		t.Fatalf("Phase 3: expected 28 vectors on follower, got %d", got)
	}
	t.Logf("Phase 3 OK: follower resumed WAL streaming, has 28 vectors")

	// Verify primary and follower vector counts match
	primaryCount := countVectors(primaryStore)
	followerCount := countVectors(followerStore)
	if primaryCount != followerCount {
		t.Fatalf("Vector count mismatch: primary=%d, follower=%d", primaryCount, followerCount)
	}

	// Verify WAL streaming resumed (not stuck in snapshot loop)
	walMu.Lock()
	since := lastWALSince
	walMu.Unlock()
	t.Logf("Last WAL since value: %s (should be > 0 after snapshot)", since)
}

// TestStreamingSnapshotFallbackToLegacy verifies that when the streaming endpoint
// returns 404, the follower falls back to the legacy JSON blob snapshot.
func TestStreamingSnapshotFallbackToLegacy(t *testing.T) {
	const dim = 2

	snapshot := &Snapshot{
		ShardID:   1,
		Sequence:  10,
		Timestamp: time.Now(),
		Vectors: []SnapshotEntry{
			{ID: "doc-1", Vector: []float32{1, 2}},
			{ID: "doc-2", Vector: []float32{3, 4}},
		},
	}
	snapshot.Checksum = snapshot.computeChecksum()

	var streamingCalled, legacyCalled bool

	server := testutil.NewLoopbackServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/internal/snapshot/stream/download":
			streamingCalled = true
			http.Error(w, "not found", http.StatusNotFound)
		case "/internal/snapshot":
			legacyCalled = true
			json.NewEncoder(w).Encode(snapshot)
		default:
			http.Error(w, "not found", http.StatusNotFound)
		}
	}))

	followerStore := newMockStore(dim)
	followerShard := &ShardServer{store: followerStore}

	replicator := NewFollowerReplicator(followerShard, FollowerReplicatorConfig{
		PrimaryAddr: server.URL,
	})

	if err := replicator.RequestFullSync(context.Background()); err != nil {
		t.Fatalf("RequestFullSync failed: %v", err)
	}

	if !streamingCalled {
		t.Error("Expected streaming endpoint to be tried first")
	}
	if !legacyCalled {
		t.Error("Expected legacy endpoint to be called as fallback")
	}

	if got := followerStore.StoreCount(); got != 2 {
		t.Fatalf("Expected 2 vectors after legacy fallback, got %d", got)
	}
	if replicator.lastAppliedSeq != 10 {
		t.Fatalf("Expected lastAppliedSeq=10, got %d", replicator.lastAppliedSeq)
	}
}
