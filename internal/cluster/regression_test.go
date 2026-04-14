package cluster

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"
	"time"

	"github.com/phenomenon0/vectordb/internal/testutil"
)

func TestStreamingSnapshotRestorePreservesHashedState(t *testing.T) {
	const dim = 2

	primary := newMockStore(dim)
	if _, err := primary.Add([]float32{1, 2}, "doc-1", "doc-1", map[string]string{"topic": "alpha"}, "coll-a", ""); err != nil {
		t.Fatalf("seed doc-1: %v", err)
	}
	if _, err := primary.Add([]float32{3, 4}, "doc-2", "doc-2", map[string]string{"topic": "beta"}, "coll-b", ""); err != nil {
		t.Fatalf("seed doc-2: %v", err)
	}
	primary.DeleteByID("doc-2")

	cfg := DefaultSnapshotSyncConfig()
	cfg.SnapshotDir = t.TempDir()
	cfg.CompressionEnabled = false

	primaryMgr, err := NewSnapshotSyncManager(primary, NewWALStream(), cfg)
	if err != nil {
		t.Fatalf("create primary snapshot manager: %v", err)
	}

	server := testutil.NewLoopbackServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/internal/snapshot/stream/download":
			if _, err := primaryMgr.CreateSnapshot(r.Context()); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			primaryMgr.HandleSnapshotDownload(w, r)
		default:
			http.NotFound(w, r)
		}
	}))

	restored := newMockStore(dim)
	replicator := NewFollowerReplicator(&ShardServer{store: restored}, FollowerReplicatorConfig{
		PrimaryAddr: server.URL,
	})
	if err := replicator.RequestFullSync(context.Background()); err != nil {
		t.Fatalf("streaming full sync: %v", err)
	}

	if len(restored.ids) != 1 || restored.ids[0] != "doc-1" {
		t.Fatalf("expected only live doc-1 after restore, got ids=%v", restored.ids)
	}

	hid := testHashID("doc-1")
	if got, ok := restored.idToIx[hid]; !ok || got != 0 {
		t.Fatalf("expected IDToIx[%d]=0 after restore, got (%d,%v)", hid, got, ok)
	}
	if restored.meta[hid]["topic"] != "alpha" {
		t.Fatalf("expected metadata keyed by hashed id, got %+v", restored.meta)
	}
	if restored.coll[hid] != "coll-a" {
		t.Fatalf("expected collection keyed by hashed id, got %+v", restored.coll)
	}
	if _, ok := restored.idToIx[restored.seqs[0]]; ok && restored.seqs[0] != hid {
		t.Fatalf("IDToIx should not be rebuilt from sequence numbers: %+v", restored.idToIx)
	}

	if _, err := restored.Upsert([]float32{9, 9}, "updated", "doc-1", map[string]string{"topic": "updated"}, "coll-a", ""); err != nil {
		t.Fatalf("upsert restored doc: %v", err)
	}
	if len(restored.ids) != 1 {
		t.Fatalf("expected upsert to update in place, got ids=%v", restored.ids)
	}
	if restored.docs[0] != "updated" {
		t.Fatalf("expected updated doc text, got %q", restored.docs[0])
	}
}

func TestTriggerElectionIgnoresStaleVoteResult(t *testing.T) {
	releaseVote := make(chan struct{})
	peer := testutil.NewLoopbackServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/internal/vote" {
			http.NotFound(w, r)
			return
		}
		<-releaseVote
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(VoteResponse{
			RequestID: "vote-1",
			VoterID:   "peer-1",
			Approve:   true,
		})
	}))

	coordinator := &DistributedVectorDB{
		quorum: NewQuorumVoter("node-1", []string{peer.URL}),
	}
	coordinator.quorum.httpClient = peer.Client()
	coordinator.quorum.voteTimeout = time.Second

	le := NewLeaderElector("node-1", coordinator, DefaultLeaderElectionConfig())
	state := &LeadershipState{
		ShardID:       0,
		ElectionState: PhaseStable,
	}

	done := make(chan struct{})
	state.mu.Lock()
	go func() {
		le.triggerElection(state)
		state.mu.Unlock()
		close(done)
	}()

	deadline := time.After(2 * time.Second)
	for {
		state.mu.RLock()
		epoch := state.LeaderEpoch
		phase := state.ElectionState
		state.mu.RUnlock()
		if epoch == 1 && phase == PhaseElecting {
			break
		}
		select {
		case <-deadline:
			t.Fatal("timed out waiting for election to start")
		default:
			time.Sleep(10 * time.Millisecond)
		}
	}

	le.mu.Lock()
	le.shardLeadership[0] = state
	le.mu.Unlock()
	le.RecordHeartbeat(0, "other-node", 2)

	close(releaseVote)

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for election to finish")
	}

	state.mu.RLock()
	defer state.mu.RUnlock()
	if state.CurrentLeader != "other-node" {
		t.Fatalf("expected higher-epoch heartbeat leader to win, got %q", state.CurrentLeader)
	}
	if state.LeaderEpoch != 2 {
		t.Fatalf("expected epoch 2 to be preserved, got %d", state.LeaderEpoch)
	}
	if state.IsLeader {
		t.Fatal("node should not become leader from stale election result")
	}
}
