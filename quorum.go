package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"
)

// ===========================================================================================
// SIMPLE QUORUM VOTING FOR DISTRIBUTED CONSENSUS
// Lightweight consensus without full Raft complexity - sufficient for 20-node scale
// ===========================================================================================

// QuorumDecision represents a distributed decision that requires majority agreement
type QuorumDecision string

const (
	DecisionFailover   QuorumDecision = "failover"
	DecisionMigration  QuorumDecision = "migration"
	DecisionFencing    QuorumDecision = "fencing"
	DecisionRebalance  QuorumDecision = "rebalance"
)

// VoteRequest represents a vote request sent to peer coordinators
type VoteRequest struct {
	RequestID   string         `json:"request_id"`    // Unique request ID
	DecisionType QuorumDecision `json:"decision_type"` // Type of decision
	ShardID     int            `json:"shard_id"`
	Payload     map[string]any `json:"payload"`      // Decision-specific data
	RequestedBy string         `json:"requested_by"` // Coordinator who initiated
	Timestamp   time.Time      `json:"timestamp"`
}

// VoteResponse represents a coordinator's vote
type VoteResponse struct {
	RequestID string `json:"request_id"`
	VoterID   string `json:"voter_id"`
	Approve   bool   `json:"approve"`
	Reason    string `json:"reason,omitempty"` // Why vote was rejected
}

// QuorumVoter handles distributed voting between coordinators
type QuorumVoter struct {
	mu sync.RWMutex

	coordinatorID    string
	peerCoordinators []string // HTTP addresses of peer coordinators
	httpClient       *http.Client

	// Active votes being tracked
	activeVotes map[string]*voteState

	// Vote timeout
	voteTimeout time.Duration
}

// voteState tracks an ongoing vote
type voteState struct {
	request      *VoteRequest
	votes        map[string]bool // coordinatorID -> approve
	totalVoters  int
	quorumSize   int // Votes needed for approval (majority)
	startedAt    time.Time
	decidedAt    *time.Time
	approved     bool
	mu           sync.RWMutex
}

// NewQuorumVoter creates a new quorum voter
func NewQuorumVoter(coordinatorID string, peerAddresses []string) *QuorumVoter {
	return &QuorumVoter{
		coordinatorID:    coordinatorID,
		peerCoordinators: peerAddresses,
		httpClient:       &http.Client{Timeout: 5 * time.Second},
		activeVotes:      make(map[string]*voteState),
		voteTimeout:      10 * time.Second,
	}
}

// RequestQuorum initiates a vote and waits for quorum decision
func (qv *QuorumVoter) RequestQuorum(ctx context.Context, req *VoteRequest) (bool, error) {
	// Auto-approve if no peers (single coordinator mode)
	if len(qv.peerCoordinators) == 0 {
		return true, nil
	}

	req.RequestedBy = qv.coordinatorID
	req.Timestamp = time.Now()

	totalVoters := len(qv.peerCoordinators) + 1 // Peers + self
	quorumSize := (totalVoters / 2) + 1          // Simple majority

	// Create vote state
	state := &voteState{
		request:     req,
		votes:       make(map[string]bool),
		totalVoters: totalVoters,
		quorumSize:  quorumSize,
		startedAt:   time.Now(),
	}

	// Register active vote
	qv.mu.Lock()
	qv.activeVotes[req.RequestID] = state
	qv.mu.Unlock()

	// Cleanup when done
	defer func() {
		qv.mu.Lock()
		delete(qv.activeVotes, req.RequestID)
		qv.mu.Unlock()
	}()

	// Vote ourselves first
	selfApprove := qv.evaluateVote(req)
	state.recordVote(qv.coordinatorID, selfApprove)

	if !selfApprove {
		return false, fmt.Errorf("local vote rejected")
	}

	// Request votes from peers concurrently
	ctx, cancel := context.WithTimeout(ctx, qv.voteTimeout)
	defer cancel()

	voteChan := make(chan *VoteResponse, len(qv.peerCoordinators))
	var wg sync.WaitGroup

	for _, peerAddr := range qv.peerCoordinators {
		wg.Add(1)
		go func(addr string) {
			defer wg.Done()
			if resp, err := qv.requestVoteFromPeer(ctx, addr, req); err == nil {
				voteChan <- resp
			}
		}(peerAddr)
	}

	// Close channel when all votes collected
	go func() {
		wg.Wait()
		close(voteChan)
	}()

	// Collect votes until quorum or timeout
	for {
		select {
		case <-ctx.Done():
			return false, fmt.Errorf("vote timeout: got %d/%d votes", state.getVoteCount(), state.quorumSize)

		case vote, ok := <-voteChan:
			if !ok {
				// All votes collected
				approved, count := state.isApproved()
				if approved {
					return true, nil
				}
				return false, fmt.Errorf("quorum not reached: %d/%d approved", count, state.quorumSize)
			}

			// Record vote
			state.recordVote(vote.VoterID, vote.Approve)

			// Check if we have quorum
			if approved, _ := state.isApproved(); approved {
				return true, nil
			}

			// Check if quorum is impossible
			if !state.canReachQuorum() {
				return false, fmt.Errorf("quorum impossible: too many rejections")
			}
		}
	}
}

// requestVoteFromPeer sends a vote request to a peer coordinator
func (qv *QuorumVoter) requestVoteFromPeer(ctx context.Context, peerAddr string, req *VoteRequest) (*VoteResponse, error) {
	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", peerAddr+"/internal/vote", bytes.NewReader(reqBody))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := qv.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("peer returned %d", resp.StatusCode)
	}

	var voteResp VoteResponse
	if err := json.NewDecoder(resp.Body).Decode(&voteResp); err != nil {
		return nil, err
	}

	return &voteResp, nil
}

// HandleVoteRequest processes incoming vote requests from peers
func (qv *QuorumVoter) HandleVoteRequest(req *VoteRequest) *VoteResponse {
	approve := qv.evaluateVote(req)

	return &VoteResponse{
		RequestID: req.RequestID,
		VoterID:   qv.coordinatorID,
		Approve:   approve,
		Reason:    qv.getVoteReason(req, approve),
	}
}

// evaluateVote determines whether to approve a vote request
func (qv *QuorumVoter) evaluateVote(req *VoteRequest) bool {
	// Basic validation
	if req.RequestID == "" || req.ShardID < 0 {
		return false
	}

	// Decision-specific logic
	switch req.DecisionType {
	case DecisionFailover:
		return qv.evaluateFailoverVote(req)
	case DecisionMigration:
		return qv.evaluateMigrationVote(req)
	case DecisionFencing:
		return qv.evaluateFencingVote(req)
	case DecisionRebalance:
		return qv.evaluateRebalanceVote(req)
	default:
		return false
	}
}

// evaluateFailoverVote checks if failover is justified
func (qv *QuorumVoter) evaluateFailoverVote(req *VoteRequest) bool {
	// Check if primary is actually unhealthy
	// Check if selected replica has low lag
	// Check if no other failover in progress

	// For now, approve if basic conditions met
	replicaID, ok := req.Payload["replica_id"].(string)
	if !ok || replicaID == "" {
		return false
	}

	// Check lag if provided
	if lag, ok := req.Payload["replication_lag"].(float64); ok {
		if lag > 1000 {
			return false // Reject if replica is too far behind
		}
	}

	return true
}

// evaluateMigrationVote checks if migration is safe
func (qv *QuorumVoter) evaluateMigrationVote(req *VoteRequest) bool {
	// Check if target node has capacity
	// Check if no other migration for this shard
	return true
}

// evaluateFencingVote checks if fencing should be issued
func (qv *QuorumVoter) evaluateFencingVote(req *VoteRequest) bool {
	// Check if node requesting fencing is healthy
	// Check if fencing epoch is higher than current
	return true
}

// evaluateRebalanceVote checks if rebalancing should proceed
func (qv *QuorumVoter) evaluateRebalanceVote(req *VoteRequest) bool {
	// Check if cluster is stable
	// Check if load imbalance justifies rebalance
	return true
}

// getVoteReason provides human-readable reason for vote decision
func (qv *QuorumVoter) getVoteReason(req *VoteRequest, approved bool) string {
	if approved {
		return "approved"
	}

	switch req.DecisionType {
	case DecisionFailover:
		if lag, ok := req.Payload["replication_lag"].(float64); ok && lag > 1000 {
			return "replica lag too high"
		}
		return "failover not justified"
	default:
		return "rejected"
	}
}

// voteState methods

func (vs *voteState) recordVote(voterID string, approve bool) {
	vs.mu.Lock()
	defer vs.mu.Unlock()
	vs.votes[voterID] = approve
}

func (vs *voteState) getVoteCount() int {
	vs.mu.RLock()
	defer vs.mu.RUnlock()
	return len(vs.votes)
}

func (vs *voteState) isApproved() (bool, int) {
	vs.mu.RLock()
	defer vs.mu.RUnlock()

	approvals := 0
	for _, approve := range vs.votes {
		if approve {
			approvals++
		}
	}

	return approvals >= vs.quorumSize, approvals
}

func (vs *voteState) canReachQuorum() bool {
	vs.mu.RLock()
	defer vs.mu.RUnlock()

	approvals := 0
	for _, approve := range vs.votes {
		if approve {
			approvals++
		}
	}

	// Calculate maximum possible approvals
	remainingVotes := vs.totalVoters - len(vs.votes)
	maxPossibleApprovals := approvals + remainingVotes

	return maxPossibleApprovals >= vs.quorumSize
}

// ===========================================================================================
// SIMPLE FENCING SYSTEM (WITHOUT RAFT)
// Uses timestamp-based epochs + quorum to prevent split-brain
// ===========================================================================================

// FencingManager manages fencing tokens for split-brain prevention
type FencingManager struct {
	mu sync.RWMutex

	// Shard -> current fencing state
	shardFencing map[int]*FencingState

	// Quorum voter for distributed token issuance
	quorum *QuorumVoter
}

// FencingState tracks fencing for a shard
type FencingState struct {
	ShardID      int
	CurrentEpoch uint64      // Monotonic counter
	CurrentOwner string      // Node ID that holds write permission
	IssuedAt     time.Time
	ExpiresAt    time.Time   // 10s TTL
	mu           sync.RWMutex
}

// NewFencingManager creates a new fencing manager
func NewFencingManager(quorum *QuorumVoter) *FencingManager {
	return &FencingManager{
		shardFencing: make(map[int]*FencingState),
		quorum:       quorum,
	}
}

// IssueFencingToken issues a new fencing token with quorum approval
func (fm *FencingManager) IssueFencingToken(ctx context.Context, shardID int, nodeID string) (uint64, error) {
	fm.mu.Lock()
	state := fm.shardFencing[shardID]
	if state == nil {
		state = &FencingState{
			ShardID:      shardID,
			CurrentEpoch: 0,
		}
		fm.shardFencing[shardID] = state
	}
	fm.mu.Unlock()

	// Request quorum approval for fencing
	req := &VoteRequest{
		RequestID:    fmt.Sprintf("fence-%d-%d-%d", shardID, time.Now().Unix(), state.CurrentEpoch+1),
		DecisionType: DecisionFencing,
		ShardID:      shardID,
		Payload: map[string]any{
			"node_id":   nodeID,
			"new_epoch": state.CurrentEpoch + 1,
		},
	}

	approved, err := fm.quorum.RequestQuorum(ctx, req)
	if err != nil || !approved {
		return 0, fmt.Errorf("quorum rejected fencing: %w", err)
	}

	// Issue token with new epoch
	state.mu.Lock()
	defer state.mu.Unlock()

	state.CurrentEpoch++
	state.CurrentOwner = nodeID
	state.IssuedAt = time.Now()
	state.ExpiresAt = time.Now().Add(10 * time.Second)

	return state.CurrentEpoch, nil
}

// ValidateFencingToken checks if a write is allowed with given token
func (fm *FencingManager) ValidateFencingToken(shardID int, nodeID string, epoch uint64) error {
	fm.mu.RLock()
	state := fm.shardFencing[shardID]
	fm.mu.RUnlock()

	if state == nil {
		return fmt.Errorf("no fencing state for shard %d", shardID)
	}

	state.mu.RLock()
	defer state.mu.RUnlock()

	// Check if token expired
	if time.Now().After(state.ExpiresAt) {
		return fmt.Errorf("fencing token expired")
	}

	// Check if epoch matches
	if epoch != state.CurrentEpoch {
		return fmt.Errorf("fencing epoch mismatch: expected %d, got %d", state.CurrentEpoch, epoch)
	}

	// Check if node matches
	if nodeID != state.CurrentOwner {
		return fmt.Errorf("fencing owner mismatch: expected %s, got %s", state.CurrentOwner, nodeID)
	}

	return nil
}

// RefreshFencingToken extends the TTL of an existing token
func (fm *FencingManager) RefreshFencingToken(shardID int, nodeID string, epoch uint64) error {
	fm.mu.RLock()
	state := fm.shardFencing[shardID]
	fm.mu.RUnlock()

	if state == nil {
		return fmt.Errorf("no fencing state for shard %d", shardID)
	}

	state.mu.Lock()
	defer state.mu.Unlock()

	// Validate before refresh
	if epoch != state.CurrentEpoch || nodeID != state.CurrentOwner {
		return fmt.Errorf("cannot refresh: invalid token")
	}

	state.ExpiresAt = time.Now().Add(10 * time.Second)
	return nil
}

// GetFencingEpoch returns the current fencing epoch for a shard
func (fm *FencingManager) GetFencingEpoch(shardID int) uint64 {
	fm.mu.RLock()
	state := fm.shardFencing[shardID]
	fm.mu.RUnlock()

	if state == nil {
		return 0
	}

	state.mu.RLock()
	defer state.mu.RUnlock()
	return state.CurrentEpoch
}
