package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"
)

// ===========================================================================================
// QUORUM VOTER TESTS
// Tests for distributed voting, vote request/response, and consensus mechanisms
// ===========================================================================================

// TestNewQuorumVoter tests quorum voter creation
func TestNewQuorumVoter(t *testing.T) {
	tests := []struct {
		name          string
		coordinatorID string
		peers         []string
	}{
		{
			name:          "single coordinator no peers",
			coordinatorID: "coord-1",
			peers:         []string{},
		},
		{
			name:          "with peers",
			coordinatorID: "coord-1",
			peers:         []string{"http://peer1:9000", "http://peer2:9000"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			qv := NewQuorumVoter(tt.coordinatorID, tt.peers)
			if qv == nil {
				t.Fatal("expected non-nil QuorumVoter")
			}
			if qv.coordinatorID != tt.coordinatorID {
				t.Errorf("coordinatorID = %v, want %v", qv.coordinatorID, tt.coordinatorID)
			}
			if len(qv.peerCoordinators) != len(tt.peers) {
				t.Errorf("peers len = %d, want %d", len(qv.peerCoordinators), len(tt.peers))
			}
			if qv.activeVotes == nil {
				t.Error("expected activeVotes map to be initialized")
			}
			if qv.voteTimeout != 10*time.Second {
				t.Errorf("voteTimeout = %v, want 10s", qv.voteTimeout)
			}
		})
	}
}

// TestRequestQuorumNoPeers tests quorum request with no peers (auto-approve)
func TestRequestQuorumNoPeers(t *testing.T) {
	qv := NewQuorumVoter("coord-1", []string{})

	req := &VoteRequest{
		RequestID:    "test-1",
		DecisionType: DecisionFailover,
		ShardID:      0,
		Payload: map[string]any{
			"replica_id":      "replica-1",
			"replication_lag": float64(10),
		},
	}

	ctx := context.Background()
	approved, err := qv.RequestQuorum(ctx, req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !approved {
		t.Error("expected auto-approval with no peers")
	}
}

// TestEvaluateVoteBasicValidation tests basic vote validation
func TestEvaluateVoteBasicValidation(t *testing.T) {
	qv := NewQuorumVoter("coord-1", []string{})

	tests := []struct {
		name    string
		req     *VoteRequest
		approve bool
	}{
		{
			name: "empty request ID",
			req: &VoteRequest{
				RequestID:    "",
				DecisionType: DecisionFailover,
				ShardID:      0,
			},
			approve: false,
		},
		{
			name: "negative shard ID",
			req: &VoteRequest{
				RequestID:    "test-1",
				DecisionType: DecisionFailover,
				ShardID:      -1,
			},
			approve: false,
		},
		{
			name: "unknown decision type",
			req: &VoteRequest{
				RequestID:    "test-1",
				DecisionType: "unknown",
				ShardID:      0,
			},
			approve: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := qv.evaluateVote(tt.req)
			if result != tt.approve {
				t.Errorf("evaluateVote() = %v, want %v", result, tt.approve)
			}
		})
	}
}

// TestEvaluateFailoverVote tests failover vote evaluation
func TestEvaluateFailoverVote(t *testing.T) {
	qv := NewQuorumVoter("coord-1", []string{})

	tests := []struct {
		name    string
		payload map[string]any
		approve bool
	}{
		{
			name:    "missing replica_id",
			payload: map[string]any{},
			approve: false,
		},
		{
			name: "empty replica_id",
			payload: map[string]any{
				"replica_id": "",
			},
			approve: false,
		},
		{
			name: "valid with low lag",
			payload: map[string]any{
				"replica_id":      "replica-1",
				"replication_lag": float64(50),
			},
			approve: true,
		},
		{
			name: "high replication lag (>1000)",
			payload: map[string]any{
				"replica_id":      "replica-1",
				"replication_lag": float64(1500),
			},
			approve: false,
		},
		{
			name: "exactly 1000 lag (borderline)",
			payload: map[string]any{
				"replica_id":      "replica-1",
				"replication_lag": float64(1000),
			},
			approve: true, // <= 1000 is acceptable
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &VoteRequest{
				RequestID:    "test-1",
				DecisionType: DecisionFailover,
				ShardID:      0,
				Payload:      tt.payload,
			}
			result := qv.evaluateFailoverVote(req)
			if result != tt.approve {
				t.Errorf("evaluateFailoverVote() = %v, want %v", result, tt.approve)
			}
		})
	}
}

// TestEvaluateLeaderElectionVote tests leader election vote evaluation
func TestEvaluateLeaderElectionVote(t *testing.T) {
	qv := NewQuorumVoter("coord-1", []string{})

	tests := []struct {
		name    string
		payload map[string]any
		approve bool
	}{
		{
			name:    "missing candidate_id",
			payload: map[string]any{"epoch": float64(1)},
			approve: false,
		},
		{
			name:    "empty candidate_id",
			payload: map[string]any{"candidate_id": "", "epoch": float64(1)},
			approve: false,
		},
		{
			name:    "missing epoch",
			payload: map[string]any{"candidate_id": "node-1"},
			approve: false,
		},
		{
			name:    "zero epoch",
			payload: map[string]any{"candidate_id": "node-1", "epoch": float64(0)},
			approve: false,
		},
		{
			name:    "negative epoch",
			payload: map[string]any{"candidate_id": "node-1", "epoch": float64(-1)},
			approve: false,
		},
		{
			name:    "valid vote",
			payload: map[string]any{"candidate_id": "node-1", "epoch": float64(5)},
			approve: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &VoteRequest{
				RequestID:    "test-1",
				DecisionType: DecisionLeaderElection,
				ShardID:      0,
				Payload:      tt.payload,
			}
			result := qv.evaluateLeaderElectionVote(req)
			if result != tt.approve {
				t.Errorf("evaluateLeaderElectionVote() = %v, want %v", result, tt.approve)
			}
		})
	}
}

// TestEvaluateLeaderRenewalVote tests leader renewal vote evaluation
func TestEvaluateLeaderRenewalVote(t *testing.T) {
	qv := NewQuorumVoter("coord-1", []string{})

	tests := []struct {
		name    string
		payload map[string]any
		approve bool
	}{
		{
			name:    "missing leader_id",
			payload: map[string]any{"epoch": float64(1)},
			approve: false,
		},
		{
			name:    "empty leader_id",
			payload: map[string]any{"leader_id": "", "epoch": float64(1)},
			approve: false,
		},
		{
			name:    "missing epoch",
			payload: map[string]any{"leader_id": "node-1"},
			approve: false,
		},
		{
			name:    "valid renewal",
			payload: map[string]any{"leader_id": "node-1", "epoch": float64(3)},
			approve: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &VoteRequest{
				RequestID:    "test-1",
				DecisionType: DecisionLeaderRenewal,
				ShardID:      0,
				Payload:      tt.payload,
			}
			result := qv.evaluateLeaderRenewalVote(req)
			if result != tt.approve {
				t.Errorf("evaluateLeaderRenewalVote() = %v, want %v", result, tt.approve)
			}
		})
	}
}

// TestEvaluateLeaderTransferVote tests leader transfer vote evaluation
func TestEvaluateLeaderTransferVote(t *testing.T) {
	qv := NewQuorumVoter("coord-1", []string{})

	tests := []struct {
		name    string
		payload map[string]any
		approve bool
	}{
		{
			name:    "missing old_leader_id",
			payload: map[string]any{"new_leader_id": "node-2", "epoch": float64(1)},
			approve: false,
		},
		{
			name:    "missing new_leader_id",
			payload: map[string]any{"old_leader_id": "node-1", "epoch": float64(1)},
			approve: false,
		},
		{
			name:    "missing epoch",
			payload: map[string]any{"old_leader_id": "node-1", "new_leader_id": "node-2"},
			approve: false,
		},
		{
			name: "valid transfer",
			payload: map[string]any{
				"old_leader_id": "node-1",
				"new_leader_id": "node-2",
				"epoch":         float64(5),
			},
			approve: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &VoteRequest{
				RequestID:    "test-1",
				DecisionType: DecisionLeaderTransfer,
				ShardID:      0,
				Payload:      tt.payload,
			}
			result := qv.evaluateLeaderTransferVote(req)
			if result != tt.approve {
				t.Errorf("evaluateLeaderTransferVote() = %v, want %v", result, tt.approve)
			}
		})
	}
}

// TestHandleVoteRequest tests incoming vote request handling
func TestHandleVoteRequest(t *testing.T) {
	qv := NewQuorumVoter("coord-1", []string{})

	req := &VoteRequest{
		RequestID:    "test-1",
		DecisionType: DecisionFailover,
		ShardID:      0,
		Payload: map[string]any{
			"replica_id":      "replica-1",
			"replication_lag": float64(10),
		},
		RequestedBy: "coord-2",
	}

	resp := qv.HandleVoteRequest(req)
	if resp == nil {
		t.Fatal("expected non-nil response")
	}
	if resp.RequestID != req.RequestID {
		t.Errorf("RequestID = %v, want %v", resp.RequestID, req.RequestID)
	}
	if resp.VoterID != qv.coordinatorID {
		t.Errorf("VoterID = %v, want %v", resp.VoterID, qv.coordinatorID)
	}
	if !resp.Approve {
		t.Error("expected approval for valid failover request")
	}
}

// TestGetVoteReason tests vote reason generation
func TestGetVoteReason(t *testing.T) {
	qv := NewQuorumVoter("coord-1", []string{})

	t.Run("approved", func(t *testing.T) {
		req := &VoteRequest{DecisionType: DecisionFailover}
		reason := qv.getVoteReason(req, true)
		if reason != "approved" {
			t.Errorf("reason = %v, want 'approved'", reason)
		}
	})

	t.Run("failover high lag", func(t *testing.T) {
		req := &VoteRequest{
			DecisionType: DecisionFailover,
			Payload:      map[string]any{"replication_lag": float64(2000)},
		}
		reason := qv.getVoteReason(req, false)
		if reason != "replica lag too high" {
			t.Errorf("reason = %v, want 'replica lag too high'", reason)
		}
	})

	t.Run("default rejection", func(t *testing.T) {
		req := &VoteRequest{DecisionType: DecisionMigration}
		reason := qv.getVoteReason(req, false)
		if reason != "rejected" {
			t.Errorf("reason = %v, want 'rejected'", reason)
		}
	})
}

// TestVoteState tests vote state tracking
func TestVoteState(t *testing.T) {
	state := &voteState{
		request:     &VoteRequest{RequestID: "test-1"},
		votes:       make(map[string]bool),
		totalVoters: 5,
		quorumSize:  3,
		startedAt:   time.Now(),
	}

	t.Run("record votes", func(t *testing.T) {
		state.recordVote("voter-1", true)
		state.recordVote("voter-2", false)
		state.recordVote("voter-3", true)

		if state.getVoteCount() != 3 {
			t.Errorf("vote count = %d, want 3", state.getVoteCount())
		}
	})

	t.Run("is approved - not yet", func(t *testing.T) {
		// 2 approvals, need 3
		approved, count := state.isApproved()
		if approved {
			t.Error("should not be approved yet")
		}
		if count != 2 {
			t.Errorf("approval count = %d, want 2", count)
		}
	})

	t.Run("can reach quorum", func(t *testing.T) {
		// 2 approvals, 1 rejection, 2 remaining = can reach 4 max = still possible
		if !state.canReachQuorum() {
			t.Error("should still be able to reach quorum")
		}
	})

	t.Run("is approved - reached", func(t *testing.T) {
		state.recordVote("voter-4", true)
		approved, count := state.isApproved()
		if !approved {
			t.Error("should be approved")
		}
		if count != 3 {
			t.Errorf("approval count = %d, want 3", count)
		}
	})

	t.Run("cannot reach quorum after too many rejections", func(t *testing.T) {
		// Reset state
		state2 := &voteState{
			votes:       make(map[string]bool),
			totalVoters: 5,
			quorumSize:  3,
		}
		state2.recordVote("voter-1", false)
		state2.recordVote("voter-2", false)
		state2.recordVote("voter-3", false)

		// 0 approvals, 3 rejections, 2 remaining = max 2 = cannot reach 3
		if state2.canReachQuorum() {
			t.Error("should not be able to reach quorum")
		}
	})
}

// TestRequestQuorumWithMockedPeers tests quorum with mock HTTP peers
func TestRequestQuorumWithMockedPeers(t *testing.T) {
	// Create mock peer servers
	approveServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req VoteRequest
		json.NewDecoder(r.Body).Decode(&req)
		resp := VoteResponse{
			RequestID: req.RequestID,
			VoterID:   "mock-peer",
			Approve:   true,
			Reason:    "approved",
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer approveServer.Close()

	rejectServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req VoteRequest
		json.NewDecoder(r.Body).Decode(&req)
		resp := VoteResponse{
			RequestID: req.RequestID,
			VoterID:   "mock-peer-2",
			Approve:   false,
			Reason:    "rejected",
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer rejectServer.Close()

	t.Run("quorum reached with approvals", func(t *testing.T) {
		qv := NewQuorumVoter("coord-1", []string{approveServer.URL, approveServer.URL})

		req := &VoteRequest{
			RequestID:    "test-quorum-1",
			DecisionType: DecisionFailover,
			ShardID:      0,
			Payload: map[string]any{
				"replica_id":      "replica-1",
				"replication_lag": float64(10),
			},
		}

		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		approved, err := qv.RequestQuorum(ctx, req)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !approved {
			t.Error("expected quorum to be approved")
		}
	})

	t.Run("quorum not reached with rejections", func(t *testing.T) {
		qv := NewQuorumVoter("coord-1", []string{rejectServer.URL, rejectServer.URL})

		req := &VoteRequest{
			RequestID:    "test-quorum-2",
			DecisionType: DecisionFailover,
			ShardID:      0,
			Payload: map[string]any{
				"replica_id":      "replica-1",
				"replication_lag": float64(10),
			},
		}

		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		approved, err := qv.RequestQuorum(ctx, req)
		if err == nil {
			t.Fatal("expected error for rejected quorum")
		}
		if approved {
			t.Error("expected quorum to be rejected")
		}
	})
}

// ===========================================================================================
// FENCING MANAGER TESTS
// Tests for fencing tokens and split-brain prevention
// ===========================================================================================

// TestNewFencingManager tests fencing manager creation
func TestNewFencingManager(t *testing.T) {
	qv := NewQuorumVoter("coord-1", []string{})
	fm := NewFencingManager(qv)

	if fm == nil {
		t.Fatal("expected non-nil FencingManager")
	}
	if fm.shardFencing == nil {
		t.Error("expected shardFencing map to be initialized")
	}
	if fm.quorum != qv {
		t.Error("expected quorum voter to be set")
	}
}

// TestIssueFencingToken tests fencing token issuance
func TestIssueFencingToken(t *testing.T) {
	qv := NewQuorumVoter("coord-1", []string{}) // No peers = auto-approve
	fm := NewFencingManager(qv)

	ctx := context.Background()

	t.Run("issue first token", func(t *testing.T) {
		epoch, err := fm.IssueFencingToken(ctx, 0, "node-1")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if epoch != 1 {
			t.Errorf("epoch = %d, want 1", epoch)
		}
	})

	t.Run("issue second token - increments epoch", func(t *testing.T) {
		epoch, err := fm.IssueFencingToken(ctx, 0, "node-2")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if epoch != 2 {
			t.Errorf("epoch = %d, want 2", epoch)
		}
	})

	t.Run("issue token for different shard", func(t *testing.T) {
		epoch, err := fm.IssueFencingToken(ctx, 1, "node-1")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if epoch != 1 {
			t.Errorf("epoch = %d, want 1 (new shard)", epoch)
		}
	})
}

// TestValidateFencingToken tests fencing token validation
func TestValidateFencingToken(t *testing.T) {
	qv := NewQuorumVoter("coord-1", []string{})
	fm := NewFencingManager(qv)
	ctx := context.Background()

	// Issue a token
	epoch, err := fm.IssueFencingToken(ctx, 0, "node-1")
	if err != nil {
		t.Fatalf("failed to issue token: %v", err)
	}

	t.Run("valid token", func(t *testing.T) {
		err := fm.ValidateFencingToken(0, "node-1", epoch)
		if err != nil {
			t.Errorf("unexpected error for valid token: %v", err)
		}
	})

	t.Run("wrong epoch", func(t *testing.T) {
		err := fm.ValidateFencingToken(0, "node-1", epoch+1)
		if err == nil {
			t.Error("expected error for wrong epoch")
		}
	})

	t.Run("wrong owner", func(t *testing.T) {
		err := fm.ValidateFencingToken(0, "node-2", epoch)
		if err == nil {
			t.Error("expected error for wrong owner")
		}
	})

	t.Run("unknown shard", func(t *testing.T) {
		err := fm.ValidateFencingToken(99, "node-1", 1)
		if err == nil {
			t.Error("expected error for unknown shard")
		}
	})
}

// TestRefreshFencingToken tests fencing token refresh
func TestRefreshFencingToken(t *testing.T) {
	qv := NewQuorumVoter("coord-1", []string{})
	fm := NewFencingManager(qv)
	ctx := context.Background()

	epoch, _ := fm.IssueFencingToken(ctx, 0, "node-1")

	t.Run("valid refresh", func(t *testing.T) {
		err := fm.RefreshFencingToken(0, "node-1", epoch)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	})

	t.Run("refresh with wrong epoch", func(t *testing.T) {
		err := fm.RefreshFencingToken(0, "node-1", epoch+1)
		if err == nil {
			t.Error("expected error for wrong epoch")
		}
	})

	t.Run("refresh with wrong owner", func(t *testing.T) {
		err := fm.RefreshFencingToken(0, "node-2", epoch)
		if err == nil {
			t.Error("expected error for wrong owner")
		}
	})

	t.Run("refresh unknown shard", func(t *testing.T) {
		err := fm.RefreshFencingToken(99, "node-1", 1)
		if err == nil {
			t.Error("expected error for unknown shard")
		}
	})
}

// TestGetFencingEpoch tests epoch retrieval
func TestGetFencingEpoch(t *testing.T) {
	qv := NewQuorumVoter("coord-1", []string{})
	fm := NewFencingManager(qv)
	ctx := context.Background()

	t.Run("unknown shard returns 0", func(t *testing.T) {
		epoch := fm.GetFencingEpoch(99)
		if epoch != 0 {
			t.Errorf("epoch = %d, want 0", epoch)
		}
	})

	t.Run("returns current epoch after token issuance", func(t *testing.T) {
		issuedEpoch, _ := fm.IssueFencingToken(ctx, 0, "node-1")
		epoch := fm.GetFencingEpoch(0)
		if epoch != issuedEpoch {
			t.Errorf("epoch = %d, want %d", epoch, issuedEpoch)
		}
	})
}

// TestFencingTokenExpiration tests token expiration
func TestFencingTokenExpiration(t *testing.T) {
	qv := NewQuorumVoter("coord-1", []string{})
	fm := NewFencingManager(qv)
	ctx := context.Background()

	epoch, _ := fm.IssueFencingToken(ctx, 0, "node-1")

	// Manually expire the token
	fm.mu.Lock()
	state := fm.shardFencing[0]
	state.mu.Lock()
	state.ExpiresAt = time.Now().Add(-1 * time.Second) // Expired
	state.mu.Unlock()
	fm.mu.Unlock()

	err := fm.ValidateFencingToken(0, "node-1", epoch)
	if err == nil {
		t.Error("expected error for expired token")
	}
}

// TestConcurrentFencingOperations tests concurrent access safety
func TestConcurrentFencingOperations(t *testing.T) {
	qv := NewQuorumVoter("coord-1", []string{})
	fm := NewFencingManager(qv)
	ctx := context.Background()

	var wg sync.WaitGroup
	operations := 50

	// Concurrent token issuance
	wg.Add(operations)
	for i := 0; i < operations; i++ {
		go func(shardID int) {
			defer wg.Done()
			_, _ = fm.IssueFencingToken(ctx, shardID%5, "node-1")
		}(i)
	}

	// Concurrent epoch reads
	wg.Add(operations)
	for i := 0; i < operations; i++ {
		go func(shardID int) {
			defer wg.Done()
			_ = fm.GetFencingEpoch(shardID % 5)
		}(i)
	}

	wg.Wait()
}

// TestConcurrentQuorumVoting tests concurrent voting
func TestConcurrentQuorumVoting(t *testing.T) {
	qv := NewQuorumVoter("coord-1", []string{}) // No peers = instant approval

	var wg sync.WaitGroup
	requests := 20

	wg.Add(requests)
	for i := 0; i < requests; i++ {
		go func(id int) {
			defer wg.Done()
			req := &VoteRequest{
				RequestID:    "test-" + string(rune('0'+id%10)),
				DecisionType: DecisionMigration,
				ShardID:      id % 5,
				Payload:      map[string]any{},
			}
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			_, _ = qv.RequestQuorum(ctx, req)
		}(i)
	}

	wg.Wait()
}
