package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"
)

// ===========================================================================================
// WAL ENTRY TYPES
// ===========================================================================================

// WALEntry represents a write-ahead log entry for replication
type WALEntry struct {
	Op     string
	ID     string
	Doc    string
	Meta   map[string]string
	Vec    []float32
	Coll   string
	Tenant string
	Time   int64
}

// ===========================================================================================
// SYNCHRONOUS WAL REPLICATION
// Quorum-based replication for zero data loss
// ===========================================================================================

// ReplicationMode defines the replication consistency level
type ReplicationMode int

const (
	// AsyncReplication: Fire-and-forget (fast, but potential data loss)
	AsyncReplication ReplicationMode = iota

	// SyncReplication: Wait for quorum (majority) before returning (balanced)
	SyncReplication

	// StrongConsistency: Wait for ALL replicas before returning (slow, but safest)
	StrongConsistency
)

func (rm ReplicationMode) String() string {
	switch rm {
	case AsyncReplication:
		return "async"
	case SyncReplication:
		return "sync_quorum"
	case StrongConsistency:
		return "strong_consistency"
	default:
		return "unknown"
	}
}

// ReplicationConfig configures replication behavior
type ReplicationConfig struct {
	Mode           ReplicationMode // Replication consistency mode
	QuorumSize     int             // Required ACKs (default: majority for sync mode)
	ReplicaTimeout time.Duration   // Timeout waiting for replica ACKs (default: 100ms)
	RetryAttempts  int             // Number of retry attempts on failure (default: 3)
	RetryDelay     time.Duration   // Delay between retries (default: 10ms)
}

// DefaultReplicationConfig returns default replication configuration
func DefaultReplicationConfig() ReplicationConfig {
	return ReplicationConfig{
		Mode:           AsyncReplication, // Safe default for dev
		QuorumSize:     0,                // Auto-calculate based on replica count
		ReplicaTimeout: 100 * time.Millisecond,
		RetryAttempts:  3,
		RetryDelay:     10 * time.Millisecond,
	}
}

// ReplicationManager handles WAL replication to replicas
type ReplicationManager struct {
	mu      sync.RWMutex
	config  ReplicationConfig
	metrics *MetricsCollector // Optional metrics
}

// NewReplicationManager creates a new replication manager
func NewReplicationManager(config ReplicationConfig, metrics *MetricsCollector) *ReplicationManager {
	// Validate config
	if config.ReplicaTimeout == 0 {
		config.ReplicaTimeout = 100 * time.Millisecond
	}
	if config.RetryAttempts == 0 {
		config.RetryAttempts = 3
	}
	if config.RetryDelay == 0 {
		config.RetryDelay = 10 * time.Millisecond
	}

	return &ReplicationManager{
		config:  config,
		metrics: metrics,
	}
}

// ReplicateEntry replicates a WAL entry to replicas based on configured mode
func (rm *ReplicationManager) ReplicateEntry(
	entry *WALEntry,
	replicas []*ReplicaNode,
	shardID int,
) error {
	if len(replicas) == 0 {
		return nil // No replicas, nothing to do
	}

	start := time.Now()
	defer func() {
		if rm.metrics != nil {
			duration := time.Since(start)
			rm.metrics.RecordOperation("replicate", shardID, duration, nil)
		}
	}()

	switch rm.config.Mode {
	case AsyncReplication:
		return rm.replicateAsync(entry, replicas, shardID)
	case SyncReplication:
		return rm.replicateSync(entry, replicas, shardID)
	case StrongConsistency:
		return rm.replicateStrong(entry, replicas, shardID)
	default:
		return fmt.Errorf("unknown replication mode: %v", rm.config.Mode)
	}
}

// replicateAsync performs async replication (fire-and-forget)
func (rm *ReplicationManager) replicateAsync(
	entry *WALEntry,
	replicas []*ReplicaNode,
	shardID int,
) error {
	// Fire-and-forget: spawn goroutines and return immediately
	for _, replica := range replicas {
		go func(r *ReplicaNode) {
			if err := rm.sendToReplica(r, entry, shardID); err != nil {
				// Log async replication failure (best-effort replication)
				fmt.Printf("[WARN] async replication to %s failed: %v\n", r.NodeID, err)
			}
		}(replica)
	}

	return nil
}

// replicateSync performs synchronous replication with quorum
func (rm *ReplicationManager) replicateSync(
	entry *WALEntry,
	replicas []*ReplicaNode,
	shardID int,
) error {
	ctx, cancel := context.WithTimeout(context.Background(), rm.config.ReplicaTimeout)
	defer cancel()

	// Calculate required quorum size
	quorumSize := rm.config.QuorumSize
	if quorumSize == 0 {
		// Default: majority (N/2 + 1)
		quorumSize = len(replicas)/2 + 1
	}

	// Channel to collect ACKs
	type ackResult struct {
		replica *ReplicaNode
		err     error
	}
	acks := make(chan ackResult, len(replicas))

	// Send to all replicas in parallel
	for _, replica := range replicas {
		go func(r *ReplicaNode) {
			err := rm.sendToReplicaWithRetry(ctx, r, entry, shardID)
			acks <- ackResult{replica: r, err: err}
		}(replica)
	}

	// Wait for quorum
	successCount := 0
	var lastErr error

	for i := 0; i < len(replicas); i++ {
		select {
		case result := <-acks:
			if result.err == nil {
				successCount++
				if successCount >= quorumSize {
					// Quorum reached!
					return nil
				}
			} else {
				lastErr = result.err
				if rm.metrics != nil {
					rm.metrics.operationErrors.WithLabelValues(
						"replicate",
						fmt.Sprintf("%d", shardID),
						"replica_failed",
					).Inc()
				}
			}
		case <-ctx.Done():
			// Timeout waiting for quorum
			err := fmt.Errorf(
				"replication timeout: got %d/%d ACKs (required: %d)",
				successCount,
				len(replicas),
				quorumSize,
			)
			if rm.metrics != nil {
				rm.metrics.operationErrors.WithLabelValues(
					"replicate",
					fmt.Sprintf("%d", shardID),
					"timeout",
				).Inc()
			}
			return err
		}
	}

	// Failed to reach quorum
	return fmt.Errorf(
		"failed to reach quorum: got %d/%d ACKs (required: %d): %w",
		successCount,
		len(replicas),
		quorumSize,
		lastErr,
	)
}

// replicateStrong performs strong consistency replication (wait for all)
func (rm *ReplicationManager) replicateStrong(
	entry *WALEntry,
	replicas []*ReplicaNode,
	shardID int,
) error {
	ctx, cancel := context.WithTimeout(context.Background(), rm.config.ReplicaTimeout)
	defer cancel()

	// Channel to collect results
	results := make(chan error, len(replicas))

	// Send to all replicas in parallel
	for _, replica := range replicas {
		go func(r *ReplicaNode) {
			results <- rm.sendToReplicaWithRetry(ctx, r, entry, shardID)
		}(replica)
	}

	// Wait for ALL replicas
	var errors []error
	for i := 0; i < len(replicas); i++ {
		select {
		case err := <-results:
			if err != nil {
				errors = append(errors, err)
			}
		case <-ctx.Done():
			return fmt.Errorf("replication timeout waiting for all replicas")
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("failed to replicate to all replicas: %v", errors)
	}

	return nil
}

// sendToReplicaWithRetry sends entry to replica with retry logic
func (rm *ReplicationManager) sendToReplicaWithRetry(
	ctx context.Context,
	replica *ReplicaNode,
	entry *WALEntry,
	shardID int,
) error {
	var lastErr error

	for attempt := 0; attempt < rm.config.RetryAttempts; attempt++ {
		if attempt > 0 {
			// Wait before retry
			select {
			case <-time.After(rm.config.RetryDelay):
			case <-ctx.Done():
				return ctx.Err()
			}
		}

		// Attempt to send
		err := rm.sendToReplica(replica, entry, shardID)
		if err == nil {
			return nil // Success!
		}

		lastErr = err
	}

	return fmt.Errorf(
		"failed after %d attempts: %w",
		rm.config.RetryAttempts,
		lastErr,
	)
}

// sendToReplica sends a WAL entry to a single replica
func (rm *ReplicationManager) sendToReplica(
	replica *ReplicaNode,
	entry *WALEntry,
	shardID int,
) error {
	if replica == nil {
		return errors.New("nil replica")
	}

	if !replica.Healthy {
		return fmt.Errorf("replica %s is unhealthy", replica.NodeID)
	}

	// Build replication endpoint URL
	endpoint := fmt.Sprintf("%s/replicate", replica.BaseURL)

	// Serialize WAL entry to JSON
	payload, err := json.Marshal(struct {
		ShardID int       `json:"shard_id"`
		Entry   *WALEntry `json:"entry"`
	}{
		ShardID: shardID,
		Entry:   entry,
	})
	if err != nil {
		return fmt.Errorf("failed to marshal WAL entry: %w", err)
	}

	// Create HTTP request with timeout
	ctx, cancel := context.WithTimeout(context.Background(), rm.config.ReplicaTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewReader(payload))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	// Send request
	client := &http.Client{
		Timeout: rm.config.ReplicaTimeout,
	}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send to replica %s: %w", replica.NodeID, err)
	}
	defer resp.Body.Close()

	// Check response status
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return fmt.Errorf("replica %s returned status %d: %s", replica.NodeID, resp.StatusCode, string(body))
	}

	return nil
}

// ReplicaNode represents a replica node
type ReplicaNode struct {
	NodeID   string
	BaseURL  string
	Healthy  bool
	ShardID  int
	Priority int // For replica selection (lower = higher priority)
}

// ===========================================================================================
// QUORUM CALCULATOR
// ===========================================================================================

// CalculateQuorum calculates the required quorum size based on replica count
func CalculateQuorum(replicaCount int) int {
	if replicaCount == 0 {
		return 0
	}
	// Majority: N/2 + 1
	return replicaCount/2 + 1
}

// CalculateStrongConsistency returns the count needed for strong consistency
func CalculateStrongConsistency(replicaCount int) int {
	// All replicas must ACK
	return replicaCount
}

// ===========================================================================================
// REPLICATION STATISTICS
// ===========================================================================================

// ReplicationStats tracks replication statistics
type ReplicationStats struct {
	mu sync.RWMutex

	TotalReplications int64
	SuccessfulQuorums int64
	FailedQuorums     int64
	Timeouts          int64
	ReplicaFailures   map[string]int64 // replica ID -> failure count
	AverageLatencyMs  float64
	TotalLatencyMs    int64
	LatencyCount      int64
}

// NewReplicationStats creates new replication statistics tracker
func NewReplicationStats() *ReplicationStats {
	return &ReplicationStats{
		ReplicaFailures: make(map[string]int64),
	}
}

// RecordSuccess records a successful replication
func (rs *ReplicationStats) RecordSuccess(latency time.Duration) {
	rs.mu.Lock()
	defer rs.mu.Unlock()

	rs.TotalReplications++
	rs.SuccessfulQuorums++
	rs.TotalLatencyMs += latency.Milliseconds()
	rs.LatencyCount++
	rs.AverageLatencyMs = float64(rs.TotalLatencyMs) / float64(rs.LatencyCount)
}

// RecordFailure records a failed replication
func (rs *ReplicationStats) RecordFailure(timeout bool, replicaID string) {
	rs.mu.Lock()
	defer rs.mu.Unlock()

	rs.TotalReplications++
	rs.FailedQuorums++

	if timeout {
		rs.Timeouts++
	}

	if replicaID != "" {
		rs.ReplicaFailures[replicaID]++
	}
}

// GetStats returns current statistics
func (rs *ReplicationStats) GetStats() map[string]any {
	rs.mu.RLock()
	defer rs.mu.RUnlock()

	return map[string]any{
		"total_replications": rs.TotalReplications,
		"successful_quorums": rs.SuccessfulQuorums,
		"failed_quorums":     rs.FailedQuorums,
		"timeouts":           rs.Timeouts,
		"average_latency_ms": rs.AverageLatencyMs,
		"replica_failures":   rs.ReplicaFailures,
		"success_rate":       float64(rs.SuccessfulQuorums) / float64(rs.TotalReplications),
	}
}

// ===========================================================================================
// USAGE EXAMPLE
// ===========================================================================================

/*
Example usage:

// Create replication manager with sync mode
config := ReplicationConfig{
    Mode:           SyncReplication,  // Wait for quorum
    QuorumSize:     0,                // Auto-calculate (majority)
    ReplicaTimeout: 100 * time.Millisecond,
    RetryAttempts:  3,
}

replMgr := NewReplicationManager(config, metricsCollector)

// Define replicas
replicas := []*ReplicaNode{
    {NodeID: "replica-1", BaseURL: "http://host1:9001", Healthy: true},
    {NodeID: "replica-2", BaseURL: "http://host2:9001", Healthy: true},
    {NodeID: "replica-3", BaseURL: "http://host3:9001", Healthy: true},
}

// Replicate WAL entry
entry := &WALEntry{
    OpType: OpInsert,
    Doc:    "Sample document",
    Vector: []float32{0.1, 0.2, 0.3},
}

// This will wait for 2/3 replicas (quorum) before returning
err := replMgr.ReplicateEntry(entry, replicas, shardID)
if err != nil {
    // Failed to reach quorum - write not safe!
    return err
}

// Success! Write is durable (replicated to majority)
*/
