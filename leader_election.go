package main

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// ===========================================================================================
// LEADER ELECTION
// Lightweight leader election for shard primaries using quorum voting
// Supports: initial election, lease renewal, leadership transfer
// ===========================================================================================

// LeaderElectionConfig configures the leader election system
type LeaderElectionConfig struct {
	// LeaderLeaseDuration is how long a leader holds the lease
	LeaderLeaseDuration time.Duration

	// LeaderRenewalInterval is how often to renew the lease
	LeaderRenewalInterval time.Duration

	// ElectionTimeout is how long to wait for election votes
	ElectionTimeout time.Duration

	// MinHealthyReplicas required before starting election
	MinHealthyReplicas int
}

// DefaultLeaderElectionConfig returns sensible defaults
func DefaultLeaderElectionConfig() LeaderElectionConfig {
	return LeaderElectionConfig{
		LeaderLeaseDuration:   30 * time.Second,
		LeaderRenewalInterval: 10 * time.Second,
		ElectionTimeout:       5 * time.Second,
		MinHealthyReplicas:    0, // Can elect with just primary
	}
}

// LeaderElector manages leader election for shards
type LeaderElector struct {
	mu sync.RWMutex

	nodeID      string
	coordinator *DistributedVectorDB
	config      LeaderElectionConfig

	// Per-shard election state
	shardLeadership map[int]*LeadershipState

	// Control
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// LeadershipState tracks leadership for a single shard
type LeadershipState struct {
	mu sync.RWMutex

	ShardID       int
	CurrentLeader string        // NodeID of current leader (empty if none)
	LeaderEpoch   uint64        // Monotonic epoch - higher epoch wins
	LeaseExpiry   time.Time     // When current leader's lease expires
	IsLeader      bool          // Are we the leader?
	LastHeartbeat time.Time     // Last heartbeat from leader
	Candidates    []string      // Nodes that can become leader
	ElectionState ElectionPhase // Current election phase
}

// ElectionPhase represents the current state of an election
type ElectionPhase int

const (
	PhaseStable   ElectionPhase = iota // No election in progress, leader is active
	PhaseElecting                      // Election in progress
	PhaseTransfer                      // Leadership transfer in progress
)

func (p ElectionPhase) String() string {
	switch p {
	case PhaseStable:
		return "stable"
	case PhaseElecting:
		return "electing"
	case PhaseTransfer:
		return "transfer"
	default:
		return "unknown"
	}
}

// NewLeaderElector creates a new leader elector
func NewLeaderElector(nodeID string, coordinator *DistributedVectorDB, config LeaderElectionConfig) *LeaderElector {
	ctx, cancel := context.WithCancel(context.Background())

	le := &LeaderElector{
		nodeID:          nodeID,
		coordinator:     coordinator,
		config:          config,
		shardLeadership: make(map[int]*LeadershipState),
		ctx:             ctx,
		cancel:          cancel,
	}

	return le
}

// Start begins leader election monitoring
func (le *LeaderElector) Start() {
	le.wg.Add(1)
	go le.electionLoop()
}

// Stop stops the leader elector
func (le *LeaderElector) Stop() {
	le.cancel()
	le.wg.Wait()
}

// electionLoop monitors shard leadership and triggers elections when needed
func (le *LeaderElector) electionLoop() {
	defer le.wg.Done()

	ticker := time.NewTicker(le.config.LeaderRenewalInterval / 2)
	defer ticker.Stop()

	for {
		select {
		case <-le.ctx.Done():
			return
		case <-ticker.C:
			le.checkAllShards()
		}
	}
}

// checkAllShards checks all shards for leader health
func (le *LeaderElector) checkAllShards() {
	le.coordinator.mu.RLock()
	shardIDs := make([]int, 0, len(le.coordinator.shards))
	for shardID := range le.coordinator.shards {
		shardIDs = append(shardIDs, shardID)
	}
	le.coordinator.mu.RUnlock()

	for _, shardID := range shardIDs {
		le.checkShard(shardID)
	}
}

// checkShard checks a single shard's leadership status
func (le *LeaderElector) checkShard(shardID int) {
	le.mu.Lock()
	state := le.shardLeadership[shardID]
	if state == nil {
		state = &LeadershipState{
			ShardID:       shardID,
			ElectionState: PhaseStable,
		}
		le.shardLeadership[shardID] = state
	}
	le.mu.Unlock()

	state.mu.Lock()
	defer state.mu.Unlock()

	now := time.Now()

	// Check if current leader's lease has expired
	if state.CurrentLeader != "" && now.After(state.LeaseExpiry) {
		fmt.Printf("[LeaderElection] Shard %d: Leader %s lease expired\n", shardID, state.CurrentLeader)
		le.triggerElection(state)
		return
	}

	// If we're the leader, renew our lease
	if state.IsLeader && state.CurrentLeader == le.nodeID {
		if now.After(state.LeaseExpiry.Add(-le.config.LeaderRenewalInterval)) {
			le.renewLease(state)
		}
	}

	// Check if there's no leader at all
	if state.CurrentLeader == "" && state.ElectionState != PhaseElecting {
		fmt.Printf("[LeaderElection] Shard %d: No leader, starting election\n", shardID)
		le.triggerElection(state)
	}
}

// triggerElection initiates a leader election for a shard
func (le *LeaderElector) triggerElection(state *LeadershipState) {
	if state.ElectionState == PhaseElecting {
		return // Already electing
	}

	state.ElectionState = PhaseElecting
	state.LeaderEpoch++
	newEpoch := state.LeaderEpoch

	fmt.Printf("[LeaderElection] Shard %d: Starting election (epoch %d)\n", state.ShardID, newEpoch)

	// Release lock for election (to allow other operations)
	state.mu.Unlock()
	defer state.mu.Lock()

	// Request quorum approval for becoming leader
	ctx, cancel := context.WithTimeout(le.ctx, le.config.ElectionTimeout)
	defer cancel()

	// Build vote request
	approved, err := le.requestLeadershipVote(ctx, state.ShardID, newEpoch)

	// Re-acquire lock and update state
	state.mu.Lock()

	if err != nil || !approved {
		fmt.Printf("[LeaderElection] Shard %d: Election failed (epoch %d): %v\n",
			state.ShardID, newEpoch, err)
		state.ElectionState = PhaseStable
		state.mu.Unlock()
		return
	}

	// We won the election!
	state.CurrentLeader = le.nodeID
	state.IsLeader = true
	state.LeaseExpiry = time.Now().Add(le.config.LeaderLeaseDuration)
	state.ElectionState = PhaseStable
	state.LastHeartbeat = time.Now()

	fmt.Printf("[LeaderElection] Shard %d: Elected as leader (epoch %d, lease until %v)\n",
		state.ShardID, newEpoch, state.LeaseExpiry.Format(time.RFC3339))

	// Update coordinator routing
	le.updateCoordinatorRole(state.ShardID, le.nodeID, RolePrimary)

	state.mu.Unlock()
}

// requestLeadershipVote requests quorum approval for leadership
func (le *LeaderElector) requestLeadershipVote(ctx context.Context, shardID int, epoch uint64) (bool, error) {
	if le.coordinator.quorum == nil {
		// No quorum configured, auto-approve
		return true, nil
	}

	req := &VoteRequest{
		RequestID:    fmt.Sprintf("leader-elect-%d-%d-%d", shardID, epoch, time.Now().UnixNano()),
		DecisionType: DecisionLeaderElection,
		ShardID:      shardID,
		Payload: map[string]any{
			"candidate_id": le.nodeID,
			"epoch":        epoch,
			"timestamp":    time.Now().Unix(),
		},
	}

	return le.coordinator.quorum.RequestQuorum(ctx, req)
}

// renewLease extends the current leader's lease
func (le *LeaderElector) renewLease(state *LeadershipState) {
	ctx, cancel := context.WithTimeout(le.ctx, le.config.ElectionTimeout)
	defer cancel()

	// Request quorum for lease renewal (lighter weight than full election)
	if le.coordinator.quorum != nil {
		req := &VoteRequest{
			RequestID:    fmt.Sprintf("leader-renew-%d-%d-%d", state.ShardID, state.LeaderEpoch, time.Now().UnixNano()),
			DecisionType: DecisionLeaderRenewal,
			ShardID:      state.ShardID,
			Payload: map[string]any{
				"leader_id": le.nodeID,
				"epoch":     state.LeaderEpoch,
			},
		}

		approved, err := le.coordinator.quorum.RequestQuorum(ctx, req)
		if err != nil || !approved {
			fmt.Printf("[LeaderElection] Shard %d: Lease renewal failed, stepping down\n", state.ShardID)
			le.stepDown(state)
			return
		}
	}

	// Extend lease
	state.LeaseExpiry = time.Now().Add(le.config.LeaderLeaseDuration)
	state.LastHeartbeat = time.Now()

	fmt.Printf("[LeaderElection] Shard %d: Lease renewed (until %v)\n",
		state.ShardID, state.LeaseExpiry.Format(time.RFC3339))
}

// stepDown voluntarily gives up leadership
func (le *LeaderElector) stepDown(state *LeadershipState) {
	if !state.IsLeader {
		return
	}

	fmt.Printf("[LeaderElection] Shard %d: Stepping down as leader\n", state.ShardID)

	state.IsLeader = false
	state.CurrentLeader = ""
	state.LeaseExpiry = time.Time{}
	state.ElectionState = PhaseStable

	// Update coordinator routing
	le.updateCoordinatorRole(state.ShardID, le.nodeID, RoleReplica)
}

// TransferLeadership initiates a controlled leadership transfer
func (le *LeaderElector) TransferLeadership(shardID int, targetNodeID string) error {
	le.mu.RLock()
	state := le.shardLeadership[shardID]
	le.mu.RUnlock()

	if state == nil {
		return fmt.Errorf("shard %d not found", shardID)
	}

	state.mu.Lock()
	defer state.mu.Unlock()

	if !state.IsLeader {
		return fmt.Errorf("not the leader of shard %d", shardID)
	}

	if state.CurrentLeader != le.nodeID {
		return fmt.Errorf("leadership mismatch")
	}

	fmt.Printf("[LeaderElection] Shard %d: Initiating leadership transfer to %s\n",
		shardID, targetNodeID)

	state.ElectionState = PhaseTransfer

	// Request quorum for leadership transfer
	ctx, cancel := context.WithTimeout(le.ctx, le.config.ElectionTimeout*2)
	defer cancel()

	if le.coordinator.quorum != nil {
		req := &VoteRequest{
			RequestID:    fmt.Sprintf("leader-transfer-%d-%d-%d", shardID, state.LeaderEpoch, time.Now().UnixNano()),
			DecisionType: DecisionLeaderTransfer,
			ShardID:      shardID,
			Payload: map[string]any{
				"old_leader_id": le.nodeID,
				"new_leader_id": targetNodeID,
				"epoch":         state.LeaderEpoch + 1,
			},
		}

		approved, err := le.coordinator.quorum.RequestQuorum(ctx, req)
		if err != nil || !approved {
			state.ElectionState = PhaseStable
			return fmt.Errorf("transfer not approved by quorum: %w", err)
		}
	}

	// Update state for transfer
	state.LeaderEpoch++
	state.CurrentLeader = targetNodeID
	state.IsLeader = false
	state.LeaseExpiry = time.Now().Add(le.config.LeaderLeaseDuration)
	state.ElectionState = PhaseStable

	// Update coordinator routing
	le.updateCoordinatorRole(shardID, le.nodeID, RoleReplica)
	le.updateCoordinatorRole(shardID, targetNodeID, RolePrimary)

	fmt.Printf("[LeaderElection] Shard %d: Leadership transferred to %s (epoch %d)\n",
		shardID, targetNodeID, state.LeaderEpoch)

	return nil
}

// updateCoordinatorRole updates the node role in the coordinator
func (le *LeaderElector) updateCoordinatorRole(shardID int, nodeID string, role ReplicaRole) {
	le.coordinator.mu.Lock()
	defer le.coordinator.mu.Unlock()

	nodes := le.coordinator.shards[shardID]
	for _, node := range nodes {
		if node.NodeID == nodeID {
			node.Role = role
			return
		}
	}
}

// IsLeader checks if this node is the leader for a shard
func (le *LeaderElector) IsLeader(shardID int) bool {
	le.mu.RLock()
	state := le.shardLeadership[shardID]
	le.mu.RUnlock()

	if state == nil {
		return false
	}

	state.mu.RLock()
	defer state.mu.RUnlock()

	return state.IsLeader && state.CurrentLeader == le.nodeID && time.Now().Before(state.LeaseExpiry)
}

// GetLeader returns the current leader for a shard
func (le *LeaderElector) GetLeader(shardID int) (string, uint64, error) {
	le.mu.RLock()
	state := le.shardLeadership[shardID]
	le.mu.RUnlock()

	if state == nil {
		return "", 0, fmt.Errorf("shard %d not found", shardID)
	}

	state.mu.RLock()
	defer state.mu.RUnlock()

	if state.CurrentLeader == "" {
		return "", state.LeaderEpoch, fmt.Errorf("no leader for shard %d", shardID)
	}

	if time.Now().After(state.LeaseExpiry) {
		return "", state.LeaderEpoch, fmt.Errorf("leader lease expired for shard %d", shardID)
	}

	return state.CurrentLeader, state.LeaderEpoch, nil
}

// GetLeadershipStats returns statistics about leader election
func (le *LeaderElector) GetLeadershipStats() map[string]any {
	le.mu.RLock()
	defer le.mu.RUnlock()

	stats := map[string]any{
		"node_id":    le.nodeID,
		"num_shards": len(le.shardLeadership),
		"shards":     make([]map[string]any, 0),
	}

	leadingCount := 0
	for shardID, state := range le.shardLeadership {
		state.mu.RLock()
		shardInfo := map[string]any{
			"shard_id":       shardID,
			"current_leader": state.CurrentLeader,
			"epoch":          state.LeaderEpoch,
			"is_leader":      state.IsLeader,
			"election_state": state.ElectionState.String(),
		}

		if !state.LeaseExpiry.IsZero() {
			shardInfo["lease_remaining_seconds"] = time.Until(state.LeaseExpiry).Seconds()
		}

		if state.IsLeader {
			leadingCount++
		}

		stats["shards"] = append(stats["shards"].([]map[string]any), shardInfo)
		state.mu.RUnlock()
	}

	stats["leading_shards"] = leadingCount

	return stats
}

// RecordHeartbeat records a heartbeat from the current leader
func (le *LeaderElector) RecordHeartbeat(shardID int, leaderID string, epoch uint64) {
	le.mu.RLock()
	state := le.shardLeadership[shardID]
	le.mu.RUnlock()

	if state == nil {
		return
	}

	state.mu.Lock()
	defer state.mu.Unlock()

	// Only accept heartbeats from current leader with matching epoch
	if state.CurrentLeader == leaderID && state.LeaderEpoch == epoch {
		state.LastHeartbeat = time.Now()
	} else if epoch > state.LeaderEpoch {
		// New leader with higher epoch
		state.CurrentLeader = leaderID
		state.LeaderEpoch = epoch
		state.LastHeartbeat = time.Now()
		state.IsLeader = (leaderID == le.nodeID)
		state.LeaseExpiry = time.Now().Add(le.config.LeaderLeaseDuration)

		if state.IsLeader {
			fmt.Printf("[LeaderElection] Shard %d: Became leader via heartbeat (epoch %d)\n",
				shardID, epoch)
		} else {
			fmt.Printf("[LeaderElection] Shard %d: Acknowledged new leader %s (epoch %d)\n",
				shardID, leaderID, epoch)
		}
	}
}

// Note: DecisionLeaderElection, DecisionLeaderRenewal, DecisionLeaderTransfer
// are defined in quorum.go alongside other QuorumDecision types
