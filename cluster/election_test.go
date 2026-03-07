package cluster

import (
	"testing"
	"time"
)

func TestLeaderElectionConfig(t *testing.T) {
	cfg := DefaultLeaderElectionConfig()

	if cfg.LeaderLeaseDuration != 30*time.Second {
		t.Errorf("expected lease duration 30s, got %v", cfg.LeaderLeaseDuration)
	}
	if cfg.LeaderRenewalInterval != 10*time.Second {
		t.Errorf("expected renewal interval 10s, got %v", cfg.LeaderRenewalInterval)
	}
	if cfg.ElectionTimeout != 5*time.Second {
		t.Errorf("expected election timeout 5s, got %v", cfg.ElectionTimeout)
	}
}

func TestLeaderElector_SingleNode(t *testing.T) {
	// Create coordinator without quorum (single-node mode)
	coordinator := NewDistributedVectorDB(DistributedConfig{
		NumShards:         3,
		ReplicationFactor: 1,
	})
	defer coordinator.Shutdown()

	// Create leader elector
	le := NewLeaderElector("node-1", coordinator, DefaultLeaderElectionConfig())

	// Start elector
	le.Start()
	defer le.Stop()

	// In single-node mode, IsLeader should work after checking
	// (Since there's no quorum, elections auto-approve)
	time.Sleep(100 * time.Millisecond)

	// Check stats
	stats := le.GetLeadershipStats()
	if stats["node_id"] != "node-1" {
		t.Errorf("expected node_id 'node-1', got %v", stats["node_id"])
	}
}

func TestLeadershipState_Phases(t *testing.T) {
	tests := []struct {
		phase    ElectionPhase
		expected string
	}{
		{PhaseStable, "stable"},
		{PhaseElecting, "electing"},
		{PhaseTransfer, "transfer"},
		{ElectionPhase(99), "unknown"},
	}

	for _, tt := range tests {
		if got := tt.phase.String(); got != tt.expected {
			t.Errorf("ElectionPhase(%d).String() = %s, want %s", tt.phase, got, tt.expected)
		}
	}
}

func TestLeaderElector_IsLeader(t *testing.T) {
	coordinator := NewDistributedVectorDB(DistributedConfig{
		NumShards:         2,
		ReplicationFactor: 1,
	})
	defer coordinator.Shutdown()

	le := NewLeaderElector("test-node", coordinator, DefaultLeaderElectionConfig())

	// Before election, should not be leader
	if le.IsLeader(0) {
		t.Error("should not be leader before election")
	}

	// Manually set up leadership state for testing
	le.mu.Lock()
	le.shardLeadership[0] = &LeadershipState{
		ShardID:       0,
		CurrentLeader: "test-node",
		LeaderEpoch:   1,
		LeaseExpiry:   time.Now().Add(30 * time.Second),
		IsLeader:      true,
		ElectionState: PhaseStable,
	}
	le.mu.Unlock()

	// Now should be leader
	if !le.IsLeader(0) {
		t.Error("should be leader after manual state setup")
	}

	// Expired lease should not be leader
	le.mu.Lock()
	le.shardLeadership[0].LeaseExpiry = time.Now().Add(-1 * time.Second)
	le.mu.Unlock()

	if le.IsLeader(0) {
		t.Error("should not be leader with expired lease")
	}
}

func TestLeaderElector_GetLeader(t *testing.T) {
	coordinator := NewDistributedVectorDB(DistributedConfig{
		NumShards:         2,
		ReplicationFactor: 1,
	})
	defer coordinator.Shutdown()

	le := NewLeaderElector("test-node", coordinator, DefaultLeaderElectionConfig())

	// No leader initially
	_, _, err := le.GetLeader(0)
	if err == nil {
		t.Error("expected error when no leader")
	}

	// Set up leader
	le.mu.Lock()
	le.shardLeadership[0] = &LeadershipState{
		ShardID:       0,
		CurrentLeader: "leader-node",
		LeaderEpoch:   5,
		LeaseExpiry:   time.Now().Add(30 * time.Second),
		IsLeader:      false, // We're not the leader
		ElectionState: PhaseStable,
	}
	le.mu.Unlock()

	// Get leader
	leader, epoch, err := le.GetLeader(0)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if leader != "leader-node" {
		t.Errorf("expected leader 'leader-node', got %s", leader)
	}
	if epoch != 5 {
		t.Errorf("expected epoch 5, got %d", epoch)
	}
}

func TestLeaderElector_RecordHeartbeat(t *testing.T) {
	coordinator := NewDistributedVectorDB(DistributedConfig{
		NumShards:         2,
		ReplicationFactor: 1,
	})
	defer coordinator.Shutdown()

	le := NewLeaderElector("follower-node", coordinator, DefaultLeaderElectionConfig())

	// Set up initial state
	le.mu.Lock()
	le.shardLeadership[0] = &LeadershipState{
		ShardID:       0,
		CurrentLeader: "leader-node",
		LeaderEpoch:   1,
		LeaseExpiry:   time.Now().Add(30 * time.Second),
		IsLeader:      false,
		ElectionState: PhaseStable,
	}
	le.mu.Unlock()

	// Record heartbeat from current leader
	le.RecordHeartbeat(0, "leader-node", 1)

	le.mu.RLock()
	state := le.shardLeadership[0]
	state.mu.RLock()
	hb := state.LastHeartbeat
	state.mu.RUnlock()
	le.mu.RUnlock()

	if hb.IsZero() {
		t.Error("heartbeat should be recorded")
	}

	// Heartbeat with higher epoch should update leader
	le.RecordHeartbeat(0, "new-leader", 2)

	le.mu.RLock()
	state = le.shardLeadership[0]
	state.mu.RLock()
	leader := state.CurrentLeader
	epoch := state.LeaderEpoch
	state.mu.RUnlock()
	le.mu.RUnlock()

	if leader != "new-leader" {
		t.Errorf("expected new leader 'new-leader', got %s", leader)
	}
	if epoch != 2 {
		t.Errorf("expected epoch 2, got %d", epoch)
	}
}

func TestLeaderElector_Stats(t *testing.T) {
	coordinator := NewDistributedVectorDB(DistributedConfig{
		NumShards:         3,
		ReplicationFactor: 1,
	})
	defer coordinator.Shutdown()

	le := NewLeaderElector("stat-node", coordinator, DefaultLeaderElectionConfig())

	// Set up multiple shards with different states
	le.mu.Lock()
	le.shardLeadership[0] = &LeadershipState{
		ShardID:       0,
		CurrentLeader: "stat-node",
		LeaderEpoch:   1,
		LeaseExpiry:   time.Now().Add(30 * time.Second),
		IsLeader:      true,
		ElectionState: PhaseStable,
	}
	le.shardLeadership[1] = &LeadershipState{
		ShardID:       1,
		CurrentLeader: "other-node",
		LeaderEpoch:   2,
		LeaseExpiry:   time.Now().Add(30 * time.Second),
		IsLeader:      false,
		ElectionState: PhaseStable,
	}
	le.mu.Unlock()

	stats := le.GetLeadershipStats()

	if stats["node_id"] != "stat-node" {
		t.Errorf("expected node_id 'stat-node', got %v", stats["node_id"])
	}
	if stats["num_shards"] != 2 {
		t.Errorf("expected num_shards 2, got %v", stats["num_shards"])
	}
	if stats["leading_shards"] != 1 {
		t.Errorf("expected leading_shards 1, got %v", stats["leading_shards"])
	}

	shards := stats["shards"].([]map[string]any)
	if len(shards) != 2 {
		t.Errorf("expected 2 shards in stats, got %d", len(shards))
	}
}

// TestQuorumDecisionTypes tests that leader election decision types are recognized
func TestQuorumDecisionTypes(t *testing.T) {
	qv := NewQuorumVoter("test-coord", nil)

	// Test leader election vote
	req := &VoteRequest{
		RequestID:    "test-election",
		DecisionType: DecisionLeaderElection,
		ShardID:      0,
		Payload: map[string]any{
			"candidate_id": "candidate-node",
			"epoch":        float64(1),
		},
	}

	resp := qv.HandleVoteRequest(req)
	if !resp.Approve {
		t.Errorf("expected leader election to be approved, got rejected: %s", resp.Reason)
	}

	// Test leader renewal vote
	renewReq := &VoteRequest{
		RequestID:    "test-renewal",
		DecisionType: DecisionLeaderRenewal,
		ShardID:      0,
		Payload: map[string]any{
			"leader_id": "current-leader",
			"epoch":     float64(1),
		},
	}

	resp = qv.HandleVoteRequest(renewReq)
	if !resp.Approve {
		t.Errorf("expected leader renewal to be approved, got rejected: %s", resp.Reason)
	}

	// Test leader transfer vote
	transferReq := &VoteRequest{
		RequestID:    "test-transfer",
		DecisionType: DecisionLeaderTransfer,
		ShardID:      0,
		Payload: map[string]any{
			"old_leader_id": "old-leader",
			"new_leader_id": "new-leader",
			"epoch":         float64(2),
		},
	}

	resp = qv.HandleVoteRequest(transferReq)
	if !resp.Approve {
		t.Errorf("expected leader transfer to be approved, got rejected: %s", resp.Reason)
	}
}

// TestQuorumDecisionValidation tests validation logic for leader decisions
func TestQuorumDecisionValidation(t *testing.T) {
	qv := NewQuorumVoter("test-coord", nil)

	// Missing candidate_id should reject
	req := &VoteRequest{
		RequestID:    "test-invalid-1",
		DecisionType: DecisionLeaderElection,
		ShardID:      0,
		Payload:      map[string]any{},
	}

	resp := qv.HandleVoteRequest(req)
	if resp.Approve {
		t.Error("expected rejection for missing candidate_id")
	}

	// Invalid epoch should reject
	req = &VoteRequest{
		RequestID:    "test-invalid-2",
		DecisionType: DecisionLeaderElection,
		ShardID:      0,
		Payload: map[string]any{
			"candidate_id": "candidate",
			"epoch":        float64(0), // Invalid: epoch must be > 0
		},
	}

	resp = qv.HandleVoteRequest(req)
	if resp.Approve {
		t.Error("expected rejection for invalid epoch")
	}
}
